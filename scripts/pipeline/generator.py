"""
Step 1. Load in the diffusion model, generate images and save them to output folder
"""

import os
import sys
import subprocess
from pathlib import Path
import torch

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import save_samples, download_checkpoint


def main():
    
    B = 10 # batch size 

    # Project root and checkpoint path setup
    ROOT_DIR = Path(__file__).resolve().parents[2]
    # Ensure checkpoint is available (downloads if missing)
    CKPT_PATH = download_checkpoint(ROOT_DIR)
    # Script’s own directory (for samples output)
    CUR_DIR = Path(__file__).resolve().parent
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Customize UNet init params as necessary
    model = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)

    # Load pretrained weights
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded pretrained weights into UNet.")

    # Sampling
    samples = diffusion.sampling(
        n_samples=B,
        image_channels=1,
        img_size=(32, 32),
        use_tqdm=True
    )

    OUTPUT_DIR = CUR_DIR / "samples"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_samples(samples, OUTPUT_DIR, prefix="sample")

if __name__ == "__main__":
    main()