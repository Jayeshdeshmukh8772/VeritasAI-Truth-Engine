"""
VeritasAI Dual-Signal Hallucination Isolation Engine.
Combines DBSCAN semantic token vector spatial graphs with peer evaluation tracks.
Supports SymPy mathematical equivalence validation.
"""

import re
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from core.result import LLMResult, DetectionResult


def check_math_equivalence(text1: str, text2: str) -> bool:
    """Check if two texts contain mathematically equivalent assertions using SymPy."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    # Helper to clean and parse expression
    def parse_math(expr_str: str):
        # Clean latex and standard expressions
        expr_str = expr_str.strip().replace('`', '')
        # Replace LaTeX-style powers and roots
        expr_str = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', expr_str)
        expr_str = expr_str.replace(r'\sqrt', 'sqrt')
        expr_str = expr_str.replace('^', '**')
        # Replace curly braces with parentheses
        expr_str = expr_str.replace('{', '(').replace('}', ')')
        # Clean up any unsupported characters
        expr_str = re.sub(r'[^a-zA-Z0-9\+\-\*/\(\)\s\.\*,]', '', expr_str)
        try:
            return parse_expr(expr_str, evaluate=True)
        except Exception:
            return None

    # 1. Extract variable assignments like "x = ..." or "y = ..."
    def extract_equations(text: str) -> dict:
        eqs = {}
        matches = re.findall(r'\b([a-zA-Z])\s*=\s*([^.\n\$;]+)', text)
        for var, expr in matches:
            v_name = var.lower().strip()
            parsed = parse_math(expr)
            if parsed is not None:
                eqs[v_name] = parsed
        return eqs

    eqs1 = extract_equations(text1)
    eqs2 = extract_equations(text2)

    # If both found some equations, check if they share variables and agree
    shared_vars = set(eqs1.keys()) & set(eqs2.keys())
    if shared_vars:
        all_equivalent = True
        for var in shared_vars:
            try:
                diff = sympy.simplify(eqs1[var] - eqs2[var])
                if diff != 0:
                    all_equivalent = False
                    break
            except Exception:
                all_equivalent = False
                break
        if all_equivalent:
            return True

    # 2. Fallback: look for isolated equations/numbers in LaTeX blocks
    def extract_latex_exprs(text: str) -> list:
        exprs = []
        blocks = re.findall(r'\$\$(.*?)\$\$|\$(.*?)\$', text, re.DOTALL)
        for b1, b2 in blocks:
            b = b1 or b2
            if not b:
                continue
            if '=' in b:
                parts = b.split('=')
                p1 = parse_math(parts[0])
                p2 = parse_math(parts[1])
                if p1 is not None and p2 is not None:
                    exprs.append((p1, p2))
            else:
                p = parse_math(b)
                if p is not None:
                    exprs.append(p)
        return exprs

    latex1 = extract_latex_exprs(text1)
    latex2 = extract_latex_exprs(text2)

    if latex1 and latex2:
        for item1 in latex1:
            for item2 in latex2:
                try:
                    if isinstance(item1, tuple) and isinstance(item2, tuple):
                        diff1 = sympy.simplify(item1[0] - item1[1])
                        diff2 = sympy.simplify(item2[0] - item2[1])
                        if sympy.simplify(diff1 - diff2) == 0 or sympy.simplify(diff1 + diff2) == 0:
                            return True
                    elif not isinstance(item1, tuple) and not isinstance(item2, tuple):
                        if sympy.simplify(item1 - item2) == 0:
                            return True
                except Exception:
                    pass

    return False


class HallucinationDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", eps: float = 0.25, min_samples: int = 2):
        # Model initializes lazily and uses Streamlit caching if running in server context
        try:
            import streamlit as st
            if st.runtime.exists():
                @st.cache_resource
                def _load_model(name):
                    return SentenceTransformer(name, device='cpu')
                self.encoder = _load_model(model_name)
            else:
                self.encoder = SentenceTransformer(model_name)
        except Exception:
            self.encoder = SentenceTransformer(model_name)
        self.eps = eps
        self.min_samples = min_samples

    def analyze(self, successful_results: List[LLMResult], semantic_weight: float = 0.6, peer_weight: float = 0.4, query_type: Optional[str] = None) -> DetectionResult:
        """Performs localized semantic clustering and flags outliers."""
        if not successful_results:
            return DetectionResult([], [], {}, 0.0, False, False)

        N = len(successful_results)
        
        # Determine weights dynamically based on available model count (Section 2 specifications)
        if N >= 4:
            sem_w = semantic_weight
            peer_w = peer_weight
        elif N == 3:
            sem_w = 0.35
            peer_w = 0.65
        else:  # N <= 2
            sem_w = 0.20
            peer_w = 0.80

        # 1. Transform raw text into sentence vectors
        corpus = [res.response for res in successful_results]
        embeddings = self.encoder.encode(corpus)

        # 2. Compute the cosine similarity matrix
        normalized_vectors = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

        # Boost similarity for mathematically equivalent responses (SymPy Verification)
        if query_type and query_type.lower() in ["mathematical", "code"]:
            for i in range(N):
                for j in range(i + 1, N):
                    if check_math_equivalence(successful_results[i].response, successful_results[j].response):
                        similarity_matrix[i, j] = 1.0
                        similarity_matrix[j, i] = 1.0

        high_dissent = False
        outlier_flags = [False] * N

        # 3. Clustering / Evaluation logic
        if N >= 4:
            # Standard DBSCAN spatial clustering
            distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 1.0)
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)
            
            # Determine consensus track
            active_labels = [label for label in labels if label != -1]
            consensus_label = max(set(active_labels), key=active_labels.count) if active_labels else -1
            
            if consensus_label != -1:
                consensus_indices = [idx for idx, lbl in enumerate(labels) if lbl == consensus_label]
                centroid = np.mean(embeddings[consensus_indices], axis=0)
                centroid_norm = centroid / np.linalg.norm(centroid)
                semantic_scores = np.dot(normalized_vectors, centroid_norm)
            else:
                semantic_scores = np.array([0.5] * N)
                
            outlier_flags = [labels[idx] != consensus_label for idx in range(N)]
            
        elif N >= 2:
            # Pairwise mean similarity fallback for N=2 or N=3
            semantic_scores = []
            for i in range(N):
                sum_sim = sum(similarity_matrix[i, j] for j in range(N) if j != i)
                semantic_scores.append(sum_sim / (N - 1))
            semantic_scores = np.array(semantic_scores)
            
            # Check high dissent for exactly 2 models
            high_dissent = (N == 2 and similarity_matrix[0, 1] < 0.55)
            
            # In N=3, flag a model if it diverges strongly from other two (mean sim < 0.60)
            # In N=2, if they dissent, flag both
            for idx in range(N):
                if N == 3:
                    outlier_flags[idx] = (semantic_scores[idx] < 0.60)
                else:
                    outlier_flags[idx] = high_dissent
        else:
            # N == 1: Single model response
            semantic_scores = np.array([1.0])
            sem_w = 0.5
            peer_w = 0.5

        trusted_pool: List[LLMResult] = []
        outlier_pool: List[LLMResult] = []
        trust_scores: dict[str, float] = {}

        # 4. Synthesize dual-signals into balanced rating
        for idx, res in enumerate(successful_results):
            sem_score = float(semantic_scores[idx])
            peer_score = res.peer_rank_score
            
            # Compute composite trust index
            composite_trust = round((sem_score * sem_w) + (peer_score * peer_w), 3)
            res.trust_score = composite_trust
            res.semantic_score = round(sem_score, 3)
            trust_scores[res.model] = composite_trust

            is_outlier = outlier_flags[idx] or (composite_trust < 0.45)
            if is_outlier:
                res.is_outlier = True
                outlier_pool.append(res)
            else:
                res.is_outlier = False
                trusted_pool.append(res)

        total_valid = len(successful_results)
        consensus_ratio = round(len(trusted_pool) / total_valid, 3) if total_valid > 0 else 0.0
        low_consensus_triggered = (consensus_ratio < 0.5) or high_dissent

        return DetectionResult(
            trusted=trusted_pool,
            outliers=outlier_pool,
            trust_scores=trust_scores,
            consensus_ratio=consensus_ratio,
            low_consensus=low_consensus_triggered,
            high_dissent=high_dissent
        )