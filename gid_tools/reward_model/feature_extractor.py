import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path

from gid_tools.diffusion_model.unet import UNet


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.  
    Implementations must provide an `extract` method that returns a tensor of shape [B, D].
    """
    @abstractmethod
    def extract(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class UnetFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts normalized mid‐block embeddings from a pretrained UNet.

    Parameters
    ----------
    unet_ckpt_path : str or Path, optional
        Path to a UNet checkpoint. If provided, weights will be loaded.
    ch : int
        Base channel dimension of the UNet.
    in_ch : int
        Number of input image channels.
    device : torch.device, optional
        Device on which to load the UNet.
    """
    def __init__(self, unet_ckpt_path=None, ch=128, in_ch=1, device=None):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # instantiate and load UNet
        self.unet = UNet(ch=ch, in_ch=in_ch).to(device)
        if unet_ckpt_path:
            ckpt = torch.load(str(unet_ckpt_path), map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.unet.load_state_dict(state_dict)
        self.unet.eval()
        # zero out sinusoidal projections
        for lin in (self.unet.linear1, self.unet.linear2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # placeholder
        self._mid_feats = None
        # hook the last ResNetBlock in the middle
        self.unet.middle[2].register_forward_hook(self._capture_mid_feats)

    def _capture_mid_feats(self, module, inp, output):
        # output: Tensor [B, C_mid, H_mid, W_mid]
        self._mid_feats = output

    def extract(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Run UNet forward to capture features, then global‐average pool and normalize.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor [B, in_ch, H, W].
        t : torch.Tensor
            Timesteps tensor [B].

        Returns
        -------
        torch.Tensor
            Normalized feature tensor [B, C_mid].
        """
        _ = self.unet(x.to(self.device), t.to(self.device))
        feats = self._mid_feats                              # [B, C_mid, h, w]
        pooled = feats.mean(dim=(2, 3))                      # [B, C_mid]
        normed = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)
        return normed

    
# Alternative extractor
class AutoencoderFeatureExtractor(BaseFeatureExtractor):
    """
    Uses a pretrained autoencoder to extract embeddings.
    """
    def __init__(self, encoder: nn.Module, device=None):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(device)
        self.device = device
        self.encoder.eval()

    def extract(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.encoder(x)                              # [B, D, ...]
        # If spatial, global pool
        if feats.dim() > 2:
            feats = feats.mean(dim=tuple(range(2, feats.dim())))
        return feats
