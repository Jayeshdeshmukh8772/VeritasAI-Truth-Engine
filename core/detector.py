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
            return DetectionResult([], [], {}, 0.0, False)

        # 1. Transform raw text into sentence vectors
        corpus = [res.response for res in successful_results]
        embeddings = self.encoder.encode(corpus)

        # 2. Compute the cosine distance matrix (1.0 - Cosine Similarity)
        normalized_vectors = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 1.0)

        # 3. Apply DBSCAN spatial clustering to separate signals from noise
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        # Determine the primary consensus track (the largest cluster excluding outliers/noise label -1)
        active_labels = [label for label in labels if label != -1]
        consensus_label = max(set(active_labels), key=active_labels.count) if active_labels else -1

        # Calculate semantic centroids to measure individual model alignment
        if consensus_label != -1:
            consensus_indices = [idx for idx, lbl in enumerate(labels) if lbl == consensus_label]
            centroid = np.mean(embeddings[consensus_indices], axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            semantic_scores = np.dot(normalized_vectors, centroid_norm)
        else:
            semantic_scores = np.array([0.5] * len(successful_results))

        trusted_pool: List[LLMResult] = []
        outlier_pool: List[LLMResult] = []
        trust_scores: dict[str, float] = {}

        # 4. Synthesize dual-signals into a balanced composite rating
        for idx, res in enumerate(successful_results):
            sem_score = float(semantic_scores[idx])
            peer_score = res.peer_rank_score
            
            # Compute composite trust index
            composite_trust = round((sem_score * semantic_weight) + (peer_score * peer_weight), 3)
            res.trust_score = composite_trust
            trust_scores[res.model] = composite_trust

            # Model is flagged if it falls outside the consensus cluster AND drops below the minimum confidence threshold
            is_semantic_outlier = (labels[idx] != consensus_label)
            if composite_trust < 0.45 and is_semantic_outlier:
                res.is_outlier = True
                outlier_pool.append(res)
            else:
                res.is_outlier = False
                trusted_pool.append(res)

        total_valid = len(successful_results)
        consensus_ratio = round(len(trusted_pool) / total_valid, 3) if total_valid > 0 else 0.0
        low_consensus_triggered = (consensus_ratio < 0.5)

        return DetectionResult(
            trusted=trusted_pool,
            outliers=outlier_pool,
            trust_scores=trust_scores,
            consensus_ratio=consensus_ratio,
            low_consensus=low_consensus_triggered
        )