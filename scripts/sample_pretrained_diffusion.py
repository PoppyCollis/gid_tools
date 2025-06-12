import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel

# Project root and checkpoint path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
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

# Import and instantiate your diffusion model
# Adjust import to match your actual package name if different
from gid_tools.diffusion_model.unet import UNet  # replace with your actual model class

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Customize UNet init params as necessary
model = UNet(ch=128, in_ch=1).to(device)
diffusion = DiffusionModel(T=1000, model=model, device=device)

# Load pretrained weights
ckpt = torch.load(str(CKPT_PATH), map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt
model.load_state_dict(state_dict)
model.eval()
print("Loaded pretrained weights into UNet.")

# Sampling
# 7) Sampling
samples = diffusion.sampling(
    n_samples=1,
    image_channels=1,
    img_size=(32, 32),
    use_tqdm=True
)

OUTPUT_DIR = ROOT_DIR / "outputs" / "pretrained_diffusion"
OUTPUT_DIR.mkdir(exist_ok=True)

for idx, img_tensor in enumerate(samples):
    # scale from [-1,1] to [0,255]
    arr = img_tensor.squeeze().cpu().numpy()
    arr = ((arr + 1) * 127.5).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)
    out_path = OUTPUT_DIR / f"sample_{idx}.png"
    img.save(out_path)
    print(f"Saved sample {idx} → {out_path}")