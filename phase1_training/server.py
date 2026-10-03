"""
Central Federated Server implementation.
Coordinates model broadcasting, secure global aggregation (FedAvg),
cryptographic verification, and global evaluation.
"""

from typing import Dict, List, Any, Optional
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import MNISTCNN, get_model_parameters, set_model_parameters
from utils.metrics import compute_classification_metrics
from utils.crypto_utils import (
    compute_state_hash,
    simulate_smpc_masking,
    aggregate_masked_parameters,
)


class FederatedServer:
    """
    Central Server orchestrating federated rounds and aggregating parameters via FedAvg.
    """

    def __init__(
        self,
        test_dataset: Optional[Dataset] = None,
        device: str = "cpu",
        use_smpc: bool = True,
    ):
        self.device = torch.device(device)
        self.use_smpc = use_smpc
        self.global_model = MNISTCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        self.round_history: List[Dict[str, Any]] = []

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Broadcasts current global model parameters to participating clients.
        """
        return get_model_parameters(self.global_model)

    def aggregate_fedavg(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        sample_counts: Dict[str, int],
        round_num: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates client model updates using Federated Averaging (FedAvg),
        optionally through simulated SMPC additive parameter masking.

        Formula:
            w_{global} = sum_{k=1}^K (n_k / n_{total}) * w_k
        """
        client_ids = list(client_updates.keys())
        total_samples = sum(sample_counts[cid] for cid in client_ids)

        if self.use_smpc:
            # Simulate SMPC pairwise additive masking
            masked_updates, mask_hashes = simulate_smpc_masking(client_updates, seed=42 + round_num)
            aggregated_dict = aggregate_masked_parameters(masked_updates, sample_counts)
        else:
            first_client = client_ids[0]
            aggregated_dict: Dict[str, torch.Tensor] = {}

            for key in client_updates[first_client].keys():
                template_val = client_updates[first_client][key]
                if template_val.dtype.is_floating_point:
                    acc = torch.zeros_like(template_val, dtype=torch.float32)
                    for cid in client_ids:
                        fraction = sample_counts[cid] / total_samples
                        acc += fraction * client_updates[cid][key].float()
                    aggregated_dict[key] = acc.to(template_val.dtype)
                else:
                    aggregated_dict[key] = template_val.clone()

        # Update global model
        set_model_parameters(self.global_model, aggregated_dict)
        return aggregated_dict

    def evaluate_global(self, custom_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluates the global model on the central test set.
        """
        loader = custom_loader or self.test_loader
        if loader is None:
            return {"accuracy": 0.0, "loss": 0.0, "auc": 0.0, "f1_macro": 0.0}

        self.global_model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.global_model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        avg_loss = total_loss / max(1, n_batches)
        metrics = compute_classification_metrics(all_targets, all_preds, all_probs, num_classes=10)
        metrics["loss"] = avg_loss
        metrics["state_hash"] = compute_state_hash(get_model_parameters(self.global_model))
        return metrics

    def save_checkpoint(self, filepath: str):
        """
        Saves current global model weights to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.global_model.state_dict(), filepath)
        print(f"[Server] Checkpoint saved successfully to '{filepath}'.")

    def load_checkpoint(self, filepath: str):
        """
        Loads model weights from disk into the global model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")
        state_dict = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(state_dict)
        print(f"[Server] Loaded global model from '{filepath}'.")
