"""
VeritasAI Dual-Signal Hallucination Isolation Engine.
Combines DBSCAN semantic token vector spatial graphs with peer evaluation tracks.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from core.result import LLMResult, DetectionResult

class HallucinationDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", eps: float = 0.25, min_samples: int = 2):
        # Model initializes lazily inside system state checkpoints
        self.encoder = SentenceTransformer(model_name)
        self.eps = eps
        self.min_samples = min_samples

    def analyze(self, successful_results: List[LLMResult], semantic_weight: float = 0.6, peer_weight: float = 0.4) -> DetectionResult:
        """Performs localized semantic clustering and flags outliers."""
        if not successful_results:
            return DetectionResult([], [], {}, 0.0, False, False)

        N = len(successful_results)
        
        # Determine weights dynamically based on available model count
        if N >= 4:
            sem_w = semantic_weight
            peer_w = peer_weight
        elif N == 3:
            sem_w = 0.40
            peer_w = 0.60
        else:  # N <= 2
            sem_w = 0.30
            peer_w = 0.70

        # 1. Transform raw text into sentence vectors
        corpus = [res.response for res in successful_results]
        embeddings = self.encoder.encode(corpus)

        # 2. Compute the cosine similarity matrix
        normalized_vectors = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

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
            high_dissent = (N == 2 and similarity_matrix[0, 1] < 0.5)
            
            # In N=3, flag a model if it diverges strongly from other two (mean sim < 0.5)
            # In N=2, if they dissent, flag both
            for idx in range(N):
                if N == 3:
                    outlier_flags[idx] = (semantic_scores[idx] < 0.5)
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