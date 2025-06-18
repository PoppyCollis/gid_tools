"""
Step 1. Load in the diffusion model, generate images and save them to output folder
"""

from pathlib import Path
import torch

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import save_samples, download_checkpoint

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
    
    B = 50 # batch size 

    # Project root and checkpoint path setup
    ROOT_DIR = Path(__file__).resolve().parents[2]
    # Ensure checkpoint is available (downloads if missing)
    CKPT_PATH = download_checkpoint(ROOT_DIR)
    # Script’s own directory (for samples output)
    CUR_DIR = Path(__file__).resolve().parent
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Customize UNet init params as necessary
    model = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)

    # Load pretrained weights
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded pretrained weights into UNet.")

    # Sampling
    samples = diffusion.sampling(
        n_samples=B,
        image_channels=1,
        img_size=(32, 32),
        use_tqdm=True
    )

    OUTPUT_DIR = CUR_DIR / "samples"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # if I want to save samples as png files
    save_samples(samples, OUTPUT_DIR, prefix="sample")
    logger.info("Saved generated images as PNG files.")

    
    tensor_path = OUTPUT_DIR / "samples.pt"
    torch.save(samples.cpu(), tensor_path)
    logger.info(f"Saved raw tensor batch to {tensor_path}")


if __name__ == "__main__":
    main()