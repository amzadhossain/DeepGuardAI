import torch
from torch import nn


class FraudNet(nn.Module):
    """Small feed-forward neural network for binary fraud classification."""

    def __init__(self, input_size: int = 30):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits. BCEWithLogitsLoss handles sigmoid internally.
        return self.network(x).squeeze(-1)

