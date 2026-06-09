"""
VeritasAI Advanced Mathematical Sensor Fusion Engine.
Implements PageRank-style Eigenvector Centrality, Shannon Information Entropy,
1D Kalman filtering for state updates, and Dempster-Shafer evidence combinations.
"""

import numpy as np
from typing import List, Dict, Tuple


class MathematicalFusionEngine:
    """Rigorous sensor-fusion math utilities for LLM consensus estimation."""

    @staticmethod
    def compute_shannon_entropy(similarity_matrix: np.ndarray) -> float:
        """
        Compute the Spectral Entropy (Von Neumann Entropy) of the similarity matrix.
        If all models agree, H = 0.0. If they completely disagree, H is maximized (log2(N)).
        
        Args:
            similarity_matrix: Pairwise cosine similarity matrix (N x N)
            
        Returns:
            Spectral entropy value in bits (float)
        """
        N = len(similarity_matrix)
        if N <= 1:
            return 0.0
            
        # Ensure similarity matrix is symmetric and non-negative
        A = np.clip(similarity_matrix, 0.0, 1.0)
        A = (A + A.T) / 2.0
        
        # Normalize by trace (which is N for cosine similarity with 1s on diagonal)
        trace = np.trace(A)
        if trace == 0:
            return float(np.log2(N))
            
        rho = A / trace
        
        # Compute eigenvalues of symmetric density matrix
        try:
            eigenvalues = np.linalg.eigvalsh(rho)
            # Clip to positive values to avoid numerical log issues
            eigenvalues = np.clip(eigenvalues, 1e-12, 1.0)
            # Compute Von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            return float(entropy)
        except Exception:
            return float(np.log2(N))

    @staticmethod
    def compute_eigenvector_centrality(similarity_matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Compute PageRank-style Eigenvector Centrality of the similarity network.
        A x = lambda x (using Power Iteration).
        
        Args:
            similarity_matrix: Pairwise cosine similarity matrix (N x N)
            max_iter: Maximum iterations for power method convergence
            tol: Tolerance criteria for convergence
            
        Returns:
            1D NumPy array representing the centrality score of each node
        """
        N = len(similarity_matrix)
        if N == 0:
            return np.array([])
        if N == 1:
            return np.array([1.0])

        # Ensure similarity matrix is non-negative and symmetric
        A = np.clip(similarity_matrix, 0.0, 1.0)
        
        # Power iteration to find principal eigenvector
        x = np.ones(N) / np.sqrt(N)
        for _ in range(max_iter):
            x_next = np.dot(A, x)
            norm = np.linalg.norm(x_next)
            if norm == 0:
                break
            x_next = x_next / norm
            if np.linalg.norm(x_next - x) < tol:
                x = x_next
                break
            x = x_next
            
        # Ensure outputs are normalized to sum to 1.0 (relative proportions)
        sum_x = np.sum(x)
        if sum_x > 0:
            x = x / sum_x
        return x

    @staticmethod
    def run_kalman_filter(x_prev: float, P_prev: float, measurement: float, Q: float = 0.02, R: float = 0.08) -> Tuple[float, float]:
        """
        Update model trust using a 1D Kalman Filter.
        
        Args:
            x_prev: Previous estimated trust score state
            P_prev: Previous covariance (uncertainty) state
            measurement: Current observed rating/score
            Q: Process noise covariance (representing gradual model trust drift)
            R: Measurement noise covariance (representing query-specific noise/hallucination)
            
        Returns:
            Tuple of (x_updated, P_updated)
        """
        # 1. Prediction State
        x_pred = x_prev
        P_pred = P_prev + Q
        
        # 2. Measurement Update
        K = P_pred / (P_pred + R)  # Kalman Gain
        x_updated = x_pred + K * (measurement - x_pred)
        P_updated = (1.0 - K) * P_pred
        
        # Clip trust to [0.0, 1.0] range
        x_updated = float(np.clip(x_updated, 0.0, 1.0))
        return x_updated, float(P_updated)

    @staticmethod
    def combine_dempster_shafer(beliefs: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine evidence masses using Dempster-Shafer theory of combination.
        Frame of Discernment: {Correct (C), Incorrect (I)}
        Input Dict items: {"C": mass_correct, "I": mass_incorrect, "U": mass_uncertain}
        
        Args:
            beliefs: List of dictionaries mapping elements (C, I, U) to belief masses
            
        Returns:
            Fused belief dictionary
        """
        if not beliefs:
            return {"C": 0.5, "I": 0.0, "U": 0.5}
            
        current = beliefs[0].copy()
        for next_b in beliefs[1:]:
            # Joint masses calculation
            m_c = current["C"] * next_b["C"] + current["C"] * next_b["U"] + current["U"] * next_b["C"]
            m_i = current["I"] * next_b["I"] + current["I"] * next_b["U"] + current["U"] * next_b["I"]
            m_u = current["U"] * next_b["U"]
            
            # Conflict coefficient K (sum of cross-disagreements)
            K = current["C"] * next_b["I"] + current["I"] * next_b["C"]
            
            # Division normalization
            if K < 1.0:
                current = {
                    "C": round(m_c / (1.0 - K), 4),
                    "I": round(m_i / (1.0 - K), 4),
                    "U": round(m_u / (1.0 - K), 4)
                }
            else:
                # Absolute conflict: assign completely to Uncertainty
                current = {"C": 0.0, "I": 0.0, "U": 1.0}
                
        return current
