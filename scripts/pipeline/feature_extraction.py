"""
Step 3. Extract features from generated samples using the U-Net middle-block hook
and compute reward predictions via the LinearRewardModel.
"""
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from gid_tools.reward_model.linear_reward_model import LinearRewardModel
from gid_tools.helpers.utils import download_checkpoint

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensures all levels are processed
# Console (stream) handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above
#formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Directory paths
    root_dir    = Path(__file__).resolve().parents[2]
    sample_dir  = Path(__file__).resolve().parent / "samples"
    tensor_path = sample_dir / "samples.pt"
    features_dir = Path(__file__).resolve().parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Ensure checkpoint is available for U-Net
    ckpt_path = download_checkpoint(root_dir)

    # Initialise model
    model = LinearRewardModel(unet_ckpt_path=ckpt_path, device=device)
    model.eval()

    # Load generated samples directly from tensor file
    if not tensor_path.exists():
        logger.error(f"Tensor file not found: {tensor_path}")
        return
    batch = torch.load(tensor_path, map_location=device)  # shape [B, C, H, W]

    # Dummy timestep tensor (zeros)
    t = torch.zeros((1,), dtype=torch.long, device=device)

    # Process each sample in the batch
    for idx, img_tensor in enumerate(batch):
        # img_tensor: [C, H, W], already normalized to [-1,1]
        x = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W] # make sure it can take batchsize

        with torch.no_grad():
            # Forward through U-Net to hook mid-block features
            _ = model.unet(x, t)
            mid_feats = model._mid_feats             # [1, C_mid, H_mid, W_mid]

            # Global average pool and L2-normalize
            pooled      = mid_feats.mean(dim=(2, 3)) # [1, C_mid]
            pooled_norm = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-6)

            # Compute reward prediction
            reward = model.reward_head(pooled_norm)

        # global average pooling
        features_t = pooled_norm.squeeze(0).cpu()  
        reward_t   = reward.cpu()     
        out_path = features_dir / f"sample_{idx}_features.pt"
        torch.save({'features': features_t, 'reward': reward_t,}, out_path)
        
    logger.info("Saved extracted features.")

if __name__ == "__main__":
    main()