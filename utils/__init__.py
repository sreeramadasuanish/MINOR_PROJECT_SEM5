"""
Utility functions for metrics calculation and cryptographic primitives in Federated Learning.
"""

from .metrics import compute_classification_metrics, compute_client_metrics_table
from .crypto_utils import (
    simulate_smpc_masking,
    aggregate_masked_parameters,
    compute_state_hash,
    verify_model_integrity,
)

__all__ = [
    "compute_classification_metrics",
    "compute_client_metrics_table",
    "simulate_smpc_masking",
    "aggregate_masked_parameters",
    "compute_state_hash",
    "verify_model_integrity",
]
