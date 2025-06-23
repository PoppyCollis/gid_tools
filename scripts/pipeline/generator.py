import torch
import argparse
import configparser
from pathlib import Path

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import save_samples, download_checkpoint

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    # Default config path: same directory as this script
    default_cfg = Path(__file__).resolve().parent / 'config.ini'

    p = argparse.ArgumentParser(
        description="Generate samples using the diffusion model."
    )
    p.add_argument(
        '--config',
        type=Path,
        default=default_cfg,
        help=f"Path to config.ini (default: {default_cfg})"
    )
    args = p.parse_args()

    # Read configuration
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    batch_size = cfg.getint('generator', 'batch_size')

    # Project root and checkpoint path setup
    ROOT_DIR = Path(__file__).resolve().parents[2]
    CKPT_PATH = download_checkpoint(ROOT_DIR)

    # Script directory for output
    CUR_DIR = Path(__file__).resolve().parent

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize UNet and diffusion
    model = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)

    # Load pretrained weights
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded pretrained weights into UNet.")

    # Generate samples
    samples = diffusion.sampling(
        n_samples=batch_size,
        image_channels=1,
        img_size=(32, 32),
        use_tqdm=True
    )

    # Save images as PNG
    OUTPUT_DIR = CUR_DIR / "samples"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_samples(samples, OUTPUT_DIR, prefix="sample")
    logger.info("Saved generated images as PNG files.")

    # Save raw tensor batch
    tensor_path = OUTPUT_DIR / "samples.pt"
    torch.save(samples.cpu(), tensor_path)
    logger.info(f"Saved raw tensor batch to {tensor_path}")


if __name__ == "__main__":
    main()
