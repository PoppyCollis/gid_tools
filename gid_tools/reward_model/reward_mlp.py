import torch
import torch.nn as nn

# class RewardMLP(nn.Module):
#     """
#     Reward MLP for the global averaged pooled extracted features:
#       256 → 128 → Dropout(0.2)
#       128 → 64  → Dropout(0.2)
#       64  → 16  → Dropout(0.1)
#       16  → 1 (scalar reward)
#     """
#     def __init__(self, input_dim: int = 256):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.Dropout(0.2),
#             nn.Linear(64, 16),
#             nn.Dropout(0.1),
#             nn.Linear(16, 1),
#         )

#     def forward(self, x):
#         return self.layers(x).squeeze(-1)

class RewardMLP(nn.Module):
    """
    Reward MLP for non pooled extracted features:
      4096 → 1024 → Dropout(0.3)
      1024 → 256 → Dropout(0.3)
      256  → 64  → Dropout(0.2)
      64   → 16  → Dropout(0.1)
      16   → 1   (scalar reward)
    """
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

