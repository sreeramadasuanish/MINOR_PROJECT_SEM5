"""
Cryptographic and Secure Multiparty Computation (SMPC) simulation utilities for Federated Learning.
Provides additive parameter masking wrappers and cryptographic state integrity verification.
"""

import hashlib
from typing import Dict, List, Tuple
import torch
import numpy as np


def compute_state_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Computes a deterministic SHA-256 hash of a PyTorch state_dict to verify parameter integrity.
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor_bytes = state_dict[key].detach().cpu().numpy().tobytes()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor_bytes)
    return hasher.hexdigest()


def simulate_smpc_masking(
    client_weights_dict: Dict[str, Dict[str, torch.Tensor]],
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
    """
    Simulates additive pairwise masking (SMPC Secure Aggregation).
    For each pair of clients (u, v), a symmetric pseudo-random mask r_{u,v} is generated.
    Client u adds r_{u,v}, and client v subtracts r_{u,v}.
    Consequently, the sum of masks across all participating clients is identically zero:
        sum_u m_u = 0,
    allowing the server to aggregate private updates without ever observing unmasked weights.

    Args:
        client_weights_dict: Dictionary mapping client_id -> state_dict
        seed: Random seed for deterministic mask generation

    Returns:
        masked_updates: Dict mapping client_id -> masked state_dict
        mask_hashes: Dict mapping client_id -> SHA-256 hash of masked state
    """
    client_ids = sorted(list(client_weights_dict.keys()))
    K = len(client_ids)
    
    if K < 2:
        # Cannot form pairwise masks with fewer than 2 clients
        return client_weights_dict, {cid: compute_state_hash(w) for cid, w in client_weights_dict.items()}

    # Initialize empty masks for each client
    template_state = client_weights_dict[client_ids[0]]
    client_masks: Dict[str, Dict[str, torch.Tensor]] = {
        cid: {k: torch.zeros_like(v, dtype=torch.float32) for k, v in template_state.items()}
        for cid in client_ids
    }

    # Generate pairwise canceling masks
    pair_seed = seed
    for i in range(K):
        for j in range(i + 1, K):
            c_i = client_ids[i]
            c_j = client_ids[j]
            generator = torch.Generator().manual_seed(pair_seed)
            pair_seed += 1

            for key, val in template_state.items():
                if val.dtype.is_floating_point:
                    # Generate random mask bounded by small standard deviation
                    mask = torch.randn(val.shape, generator=generator, dtype=val.dtype) * 0.05
                    client_masks[c_i][key] += mask
                    client_masks[c_j][key] -= mask

    # Add masks to client weights
    masked_updates: Dict[str, Dict[str, torch.Tensor]] = {}
    mask_hashes: Dict[str, str] = {}

    for cid in client_ids:
        masked_state = {}
        for k, v in client_weights_dict[cid].items():
            if v.dtype.is_floating_point:
                masked_state[k] = v.clone() + client_masks[cid][k]
            else:
                masked_state[k] = v.clone()
        masked_updates[cid] = masked_state
        mask_hashes[cid] = compute_state_hash(masked_state)

    return masked_updates, mask_hashes


def aggregate_masked_parameters(
    masked_client_weights: Dict[str, Dict[str, torch.Tensor]],
    sample_weights: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """
    Aggregates masked client model parameters.
    When uniform or calibrated SMPC weighting is applied, pairwise masks cancel out.

    Args:
        masked_client_weights: Dictionary mapping client_id -> masked state_dict
        sample_weights: Sample counts per client

    Returns:
        Aggregated state_dict
    """
    client_ids = list(masked_client_weights.keys())
    total_samples = sum(sample_weights[cid] for cid in client_ids)
    
    first_client = client_ids[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in masked_client_weights[first_client].keys():
        val = masked_client_weights[first_client][key]
        if val.dtype.is_floating_point:
            # FedAvg weighted accumulation
            acc = torch.zeros_like(val, dtype=torch.float32)
            for cid in client_ids:
                weight_fraction = sample_weights[cid] / total_samples
                acc += weight_fraction * masked_client_weights[cid][key].float()
            aggregated_state[key] = acc.to(val.dtype)
        else:
            aggregated_state[key] = val.clone()

    return aggregated_state


def verify_model_integrity(state_dict: Dict[str, torch.Tensor], expected_hash: str) -> bool:
    """
    Verifies that the SHA-256 hash of the received model matches the sender's expected hash.
    """
    actual_hash = compute_state_hash(state_dict)
    return actual_hash == expected_hash
