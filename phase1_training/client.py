"""
Client Node implementation for Federated Learning on MNIST.
Includes local training, class-imbalance oversampling, gradient/update clipping,
and Differential Privacy (DP) Gaussian noise injection.
"""

from typing import Dict, Any, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .model import MNISTCNN, get_model_parameters, set_model_parameters
from utils.metrics import compute_classification_metrics
from utils.crypto_utils import compute_state_hash


class ClientNode:
    """
    Simulates a federated client node holding local, non-IID MNIST data.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: Subset,
        test_dataset: Subset,
        batch_size: int = 64,
        lr: float = 0.01,
        device: str = "cpu",
        use_oversampling: bool = True,
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.use_oversampling = use_oversampling

        self.model = MNISTCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Build dataloaders
        self.train_loader = self._build_train_loader()
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        # Track telemetry
        self.last_train_loss = 0.0
        self.last_clipped_norm = 0.0
        self.last_noise_std = 0.0

    def _build_train_loader(self) -> DataLoader:
        """
        Builds DataLoader with class-imbalance oversampling via WeightedRandomSampler
        if enabled, or standard random shuffle.
        """
        if not self.use_oversampling or len(self.train_dataset) == 0:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Extract targets to calculate inverse class frequencies
        targets = []
        for i in range(len(self.train_dataset)):
            _, y = self.train_dataset[i]
            targets.append(int(y))

        targets = np.array(targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        count_dict = dict(zip(classes, class_counts))

        # Weight per sample = 1 / class_frequency
        sample_weights = np.array([1.0 / (count_dict[t] + 1e-6) for t in targets], dtype=np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

    def train_epoch(
        self,
        global_parameters: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        clip_c: float = 1.0,
        dp_sigma: float = 0.05,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Performs local client training starting from the global model parameters.

        Steps:
            1. Load global parameters.
            2. Train locally for `local_epochs` using SGD and CrossEntropyLoss.
            3. Apply gradient clipping to threshold `clip_c`.
            4. Compute weight delta: delta_w = w_local - w_global.
            5. Clip weight delta L2 norm to threshold `clip_c`.
            6. Inject Differential Privacy Gaussian noise: N(0, (sigma * clip_c)^2).
            7. Return perturbed model state and training telemetry.
        """
        set_model_parameters(self.model, global_parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        # Save pre-training parameters
        pre_params = {k: v.clone().to(self.device) for k, v in global_parameters.items()}

        epoch_losses = []
        for _ in range(local_epochs):
            batch_losses = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Step 3: Gradient clipping
                if clip_c > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_c)

                optimizer.step()
                batch_losses.append(loss.item())

            epoch_losses.append(np.mean(batch_losses) if batch_losses else 0.0)

        self.last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Step 4: Compute weight update deltas
        post_params = get_model_parameters(self.model)
        deltas = {}
        total_delta_sq_norm = 0.0

        for key, post_val in post_params.items():
            if post_val.dtype.is_floating_point:
                d = post_val.to(self.device) - pre_params[key]
                deltas[key] = d
                total_delta_sq_norm += torch.sum(d ** 2).item()

        delta_norm = float(np.sqrt(total_delta_sq_norm))
        self.last_clipped_norm = delta_norm

        # Step 5: Model update L2 clipping
        clipping_factor = 1.0
        if clip_c > 0 and delta_norm > clip_c:
            clipping_factor = clip_c / (delta_norm + 1e-8)
            for k in deltas.keys():
                deltas[k] *= clipping_factor
            delta_norm = clip_c

        # Step 6: Calibrated DP Gaussian noise injection
        # Scale noise std by clipping bound C and noise multiplier sigma
        noise_std = float(dp_sigma * clip_c) if (dp_sigma > 0 and clip_c > 0) else 0.0
        self.last_noise_std = noise_std

        perturbed_state = {}
        for key, post_val in post_params.items():
            if post_val.dtype.is_floating_point:
                clipped_d = deltas[key]
                if noise_std > 0:
                    noise = torch.randn_like(clipped_d) * noise_std
                    perturbed_val = pre_params[key] + clipped_d + noise
                else:
                    perturbed_val = pre_params[key] + clipped_d
                perturbed_state[key] = perturbed_val.cpu()
            else:
                perturbed_state[key] = post_val.cpu()

        state_hash = compute_state_hash(perturbed_state)
        telemetry = {
            "client_id": self.client_id,
            "train_loss": self.last_train_loss,
            "pre_clip_norm": delta_norm / max(clipping_factor, 1e-8),
            "post_clip_norm": delta_norm,
            "noise_std": noise_std,
            "n_samples": len(self.train_dataset),
            "state_hash": state_hash,
        }

        return perturbed_state, telemetry

    def evaluate_local(self, parameters: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Evaluates the client model on the local test partition.
        """
        if parameters is not None:
            set_model_parameters(self.model, parameters)

        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
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
        metrics["client_id"] = self.client_id
        return metrics
