"""
Unit tests for the VeritasAI Mathematical Fusion Engine.
"""

import numpy as np
import pytest
from core.math_engine import MathematicalFusionEngine


def test_shannon_entropy():
    # 1. Identical outputs (maximum similarity matrix of ones) -> should result in 0 entropy
    sim_ones = np.ones((3, 3))
    entropy = MathematicalFusionEngine.compute_shannon_entropy(sim_ones)
    assert abs(entropy) < 1e-9

    # 2. Random matrix check
    sim_random = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0]
    ])
    entropy_val = MathematicalFusionEngine.compute_shannon_entropy(sim_random)
    assert entropy_val > 0.0
    assert entropy_val < 3.0


def test_eigenvector_centrality():
    # Symmetric fully connected network: all nodes should have equal centrality (1/N = 0.25)
    sim_matrix = np.ones((4, 4))
    centrality = MathematicalFusionEngine.compute_eigenvector_centrality(sim_matrix)
    assert len(centrality) == 4
    for val in centrality:
        assert abs(val - 0.25) < 1e-3

    # Unbalanced matrix where node 0 is highly connected
    unbalanced_sim = np.array([
        [1.0, 0.9, 0.9, 0.9],
        [0.9, 1.0, 0.1, 0.1],
        [0.9, 0.1, 1.0, 0.1],
        [0.9, 0.1, 0.1, 1.0]
    ])
    centrality_unbalanced = MathematicalFusionEngine.compute_eigenvector_centrality(unbalanced_sim)
    # Node 0 must have the highest centrality score
    assert centrality_unbalanced[0] > centrality_unbalanced[1]
    assert centrality_unbalanced[0] > centrality_unbalanced[2]
    assert centrality_unbalanced[0] > centrality_unbalanced[3]
    # Sum of relative centralities must equal 1.0
    assert abs(np.sum(centrality_unbalanced) - 1.0) < 1e-5


def test_kalman_filter():
    # Baseline update check
    x_prev = 0.80
    P_prev = 0.10
    measurement = 0.90
    
    x_updated, P_updated = MathematicalFusionEngine.run_kalman_filter(x_prev, P_prev, measurement)
    
    # Updated trust should be between prior and measurement
    assert x_updated > 0.80
    assert x_updated < 0.90
    # Covariance (uncertainty) should decrease
    assert P_updated < 0.10
    
    # Boundary check (clipping to [0.0, 1.0])
    x_up, P_up = MathematicalFusionEngine.run_kalman_filter(0.99, 0.05, 2.5)
    assert x_up == 1.0


def test_dempster_shafer():
    # Setup two identical models agreeing on Correctness (0.8) and minor uncertainty (0.2)
    b1 = {"C": 0.8, "I": 0.0, "U": 0.2}
    b2 = {"C": 0.7, "I": 0.1, "U": 0.2}
    
    fused = MathematicalFusionEngine.combine_dempster_shafer([b1, b2])
    
    # Belief in correctness should be high
    assert fused["C"] > 0.80
    # Belief in uncertainty should decrease
    assert fused["U"] < 0.20
    # Beliefs must sum to 1.0
    assert abs(fused["C"] + fused["I"] + fused["U"] - 1.0) < 1e-3

    # Absolute conflict case (division by zero fallback)
    b_conflict1 = {"C": 1.0, "I": 0.0, "U": 0.0}
    b_conflict2 = {"C": 0.0, "I": 1.0, "U": 0.0}
    fused_conflict = MathematicalFusionEngine.combine_dempster_shafer([b_conflict1, b_conflict2])
    assert fused_conflict["U"] == 1.0
