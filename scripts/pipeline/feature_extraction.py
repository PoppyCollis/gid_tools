# File: scripts/pipeline/feature_extraction.py

"""
Step 3. Extract features from generated samples using the U-Net middle-block hook
and compute reward predictions via the modular RewardModel (extractor + MLP head).
"""

from pathlib import Path
import torch
import logging

from gid_tools.reward_model.feature_extractor import UnetFeatureExtractor
from gid_tools.reward_model.reward_mlp import RewardMLP
from gid_tools.reward_model.reward_model import RewardModel
from gid_tools.helpers.utils import download_checkpoint

# === logger setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    root_dir     = Path(__file__).resolve().parents[2]
    sample_dir   = Path(__file__).resolve().parent / "samples"
    tensor_path  = sample_dir / "samples.pt"
    features_dir = Path(__file__).resolve().parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Download / load UNet checkpoint
    ckpt_path = download_checkpoint(root_dir)

    # --- Modular model setup ---
    # 1) feature extractor
    extractor = UnetFeatureExtractor(
        unet_ckpt_path=ckpt_path,
        ch=128,            # same base‐channel as your UNet
        in_ch=1,           # input channels
        device=device
    )
    # 2) reward head (MLP)
    #    input_dim must match extractor.extract’s output dim (here mid‐block channels, e.g. 2*128)
    head = RewardMLP(input_dim=2*128).to(device)
    # 3) combine
    model = RewardModel(feature_extractor=extractor, reward_head=head).to(device)
    model.eval()

    # Load samples
    if not tensor_path.exists():
        logger.error(f"Tensor file not found: {tensor_path}")
        return
    batch = torch.load(tensor_path, map_location=device)  # [B, C, H, W]

    # Dummy timestep (for diffusion UNet)
    t = torch.zeros((1,), dtype=torch.long, device=device)

    # Process each image
    for idx, img in enumerate(batch):
        x = img.unsqueeze(0).to(device)  # [1, C, H, W]

        with torch.no_grad():
            # 1) extract features [1, D]
            feats = extractor.extract(x, t=t)

            # 2) compute reward
            reward = head(feats).squeeze(-1)

        # Move to CPU and save
        out = {
            "features": feats.squeeze(0).cpu(),
            "reward":   reward.cpu(),
        }
        out_path = features_dir / f"sample_{idx}_features.pt"
        torch.save(out, out_path)

    logger.info("All features extracted and saved.")


if __name__ == "__main__":
    main()
