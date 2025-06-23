import torch
import torch.nn as nn

from .feature_extractor import BaseFeatureExtractor, UnetFeatureExtractor
from .reward_mlp import RewardMLP


class RewardModel(nn.Module):
    """
    Modular reward model that composes a feature extractor and an MLP reward head.
    """
    def __init__(self,
                 feature_extractor: BaseFeatureExtractor,
                 reward_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.reward_head = reward_head

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Extract a fixed‐dim embedding
        feats = self.feature_extractor.extract(x, **kwargs)  # [B, D]
        # Compute scalar reward
        return self.reward_head(feats).squeeze(-1)