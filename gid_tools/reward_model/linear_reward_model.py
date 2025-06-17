import torch
import torch.nn as nn
from gid_tools.diffusion_model.unet import UNet
from gid_tools.reward_model.reward_mlp import RewardMLP

import sys
import subprocess
from pathlib import Path

"""
Try instead ...

Another option would just be to train a resnet on my dataset and ensure that I get good embeddings
for downstream predictions rather than these potentially strange denoising embeddings.

from gid_tools.models.cnn import tool_cnn

encoder = tool_cnn(pretrained=True)
encoder = nn.Sequential(*list(encoder.children())[:-2]) 
# now encoder(x) → [B, 512, H’, W’]

# then hook its last conv output (or just call encoder(x) directly)
feats = encoder(x)  # [B, 512, h, w]
pooled = feats.mean(dim=(2,3))  # [B, 512]
# ... then L2-norm + MLP
"""


class LinearRewardModel(nn.Module):
    def __init__(self, unet_ckpt_path=None, ch=128, in_ch=1, device=None):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1) instantiate U-Net backbone
        self.unet = UNet(ch=ch, in_ch=in_ch).to(device)
        if unet_ckpt_path:
            ckpt = torch.load(unet_ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            self.unet.load_state_dict(state_dict)
            #self.unet.load_state_dict(torch.load(unet_ckpt_path, map_location=device))
        self.unet.eval()
        
        # zero-out the 2 lin layers that project the sinusoidal embedding into the net
        for lin in (self.unet.linear1, self.unet.linear2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        
        # 2) placeholder for mid-block features
        self._mid_feats = None
        
        # hook the output of the last middle block (ResNetBlock)
        # self.unet.middle = [ResNetBlock, AttentionBlock, ResNetBlock]
        self.unet.middle[2].register_forward_hook(self._capture_mid_feats)

        # 3) define your MLP head (input_dim = C_mid = 2*ch = 256)
        self.reward_head = RewardMLP(input_dim=2*ch).to(device)
        
    def _capture_mid_feats(self, module, inp, output):
        # output: Tensor [B, C_mid, H_mid, W_mid]
        self._mid_feats = output
        
    def forward(self, x, t):
        """
        x: [B, in_ch, H, W]
        t: [B] long timesteps
        returns: [B] scalar rewards
        """
        # 1) run U-Net; hook populates self._mid_feats
        _ = self.unet(x, t)
        
        # 2) take the features we captured
        feats = self._mid_feats                  # [B, 256, 4, 4]
        
        # 3) global average pool
        pooled = feats.mean(dim=(2, 3))          # [B, 256]
        
        # 4) L2 normalize
        pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)
        
        # 5) MLP → scalar reward
        return self.reward_head(pooled)


def main():
    
    batch_size = 1
    
    # Project root and checkpoint path setup
    ROOT_DIR = Path(__file__).resolve().parents[2]
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    CKPT_FILE = "diffusion_ckpt.pth"
    CKPT_PATH = CHECKPOINT_DIR / CKPT_FILE

    # If checkpoint is missing, download using download_model_weights.py
    if not CKPT_PATH.exists():
        print(f"Checkpoint not found at {CKPT_PATH}. Downloading...")
        download_script = ROOT_DIR / "scripts" / "download_model_weights.py"
        if not download_script.exists():
            raise FileNotFoundError(f"Download script not found: {download_script}")
        subprocess.run([sys.executable, str(download_script)], check=True)
    
    reward_model = LinearRewardModel(unet_ckpt_path=CKPT_PATH)
    t = torch.zeros(batch_size, dtype=torch.long, device=x.device)
    # for now lets get an example x from the outputs file
    x = ...
    rewards = reward_model(x, t)
    
    
if __name__ == "__main__":
    
    # can I load in the non-reduced dataset model weights? Are the unets the same dimensionality? 
    # in which case I should probably use this one...
    
    
    main()
    
