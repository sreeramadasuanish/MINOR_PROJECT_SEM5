"""
Standard PyTorch Convolutional Neural Network (CNN) for MNIST classification.
Consists of 2 Convolutional layers with Max-Pooling, followed by Dense layers mapping to 10 digit classes.
"""

from typing import Dict, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    Standard PyTorch CNN architecture for MNIST benchmark.
    Layer 1: Conv2d(1 -> 32, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)
    Layer 2: Conv2d(32 -> 64, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)
    Dense 1: Linear(64 * 7 * 7 -> 128) -> ReLU -> Dropout(0.25)
    Dense 2: Linear(128 -> 10)
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 1, 28, 28]
        x = F.relu(self.conv1(x))      # [B, 32, 28, 28]
        x = self.pool(x)               # [B, 32, 14, 14]

        x = F.relu(self.conv2(x))      # [B, 64, 14, 14]
        x = self.pool(x)               # [B, 64, 7, 7]
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)      # [B, 3136]
        x = F.relu(self.fc1(x))        # [B, 128]
        x = self.dropout2(x)
        logits = self.fc2(x)           # [B, 10]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes softmax class probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def get_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Returns a detached clone of the model's state dictionary on CPU.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_parameters(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Loads state dictionary into model parameters.
    """
    device = next(model.parameters()).device
    loaded_dict = {k: v.to(device) for k, v in state_dict.items()}
    model.load_state_dict(loaded_dict)
