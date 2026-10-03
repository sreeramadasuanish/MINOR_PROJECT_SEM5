"""
Phase 2: FED4FED Statistical Fairness Evaluation Framework.
Provides Algorithm 1 (Chi-Square Homogeneity), Algorithm 2 (Welch's F-test),
and Algorithm 3 (Bootstrap Confidence Intervals & Pairwise Disparity).
"""

from .bootstrap import run_client_bootstrap_evaluation
from .fed4fed_algorithms import (
    compute_algorithm1_chisquare,
    compute_algorithm2_welch_f,
    compute_algorithm3_confidence_intervals,
)
from .evaluate_fairness import run_fairness_evaluation

__all__ = [
    "run_client_bootstrap_evaluation",
    "compute_algorithm1_chisquare",
    "compute_algorithm2_welch_f",
    "compute_algorithm3_confidence_intervals",
    "run_fairness_evaluation",
]
