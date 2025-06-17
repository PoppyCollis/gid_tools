import torch
import torch.nn as nn

class RewardMLP(nn.Module):
    """
    MLP reward model mirroring SEIKO's structure, but with a 256-dimensional input.
    Architecture:
      256 → 1024 → Dropout(0.2)
      1024 → 128 → Dropout(0.2)
      128 → 64  → Dropout(0.1)
      64  → 16
      16  → 1 (scalar reward)
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.layers(x).squeeze(-1)