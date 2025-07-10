#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import load_config


def estimate_kl_at_T(diffusion: DiffusionModel,
                     data_loader: DataLoader,
                     device: torch.device):
    # Get the final alpha_bar_T scalar
    alpha_bar_T = diffusion.alpha_bar[-1]  # tensor scalar
    mu_coef     = torch.sqrt(alpha_bar_T)
    sigma_coef  = torch.sqrt(1 - alpha_bar_T)

    sum_xT  = None
    sum_xT2 = None
    total_n = 0

    with torch.no_grad():
        for x0_batch in data_loader:
            # If your DataLoader yields (x0, label), do: x0, _ = x0_batch
            x0 = x0_batch.to(device)  
            B  = x0.shape[0]
            eps = torch.randn_like(x0, device=device)
            xT  = mu_coef * x0 + sigma_coef * eps

            if sum_xT is None:
                sum_xT  = xT.sum(dim=0)
                sum_xT2 = (xT**2).sum(dim=0)
            else:
                sum_xT  += xT.sum(dim=0)
                sum_xT2 += (xT**2).sum(dim=0)
            total_n += B

    # per-pixel empirical mean & var
    mean_T = sum_xT  / total_n          # [C,H,W]
    var_T  = sum_xT2 / total_n - mean_T**2

    # flatten
    mu_vec  = mean_T .reshape(-1)
    var_vec = var_T   .reshape(-1)

    # KL[N(mu,var) ‖ N(0,1)] diag case
    kl = 0.5 * (var_vec + mu_vec**2 - 1 - torch.log(var_vec))
    return kl.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=Path,
                        required=True,
                        help="path to config_ground_truth.ini")
    parser.add_argument("--batch-size",
                        type=int,
                        default=64)
    args = parser.parse_args()

    # --- 1) Load config to find where the test tensor lives ---
    cfg = load_config(args.config)
    samples_dir    = Path(__file__).parent / cfg["samples"]["directory"]
    test_file_name = cfg["samples"]["test_tensor_file"]
    test_path      = samples_dir / test_file_name

    if not test_path.exists():
        raise FileNotFoundError(f"Test tensor not found at {test_path}; "
                                f"run load_ground_truth_samples.py first")

    # --- 2) Load raw test tensor and wrap in DataLoader ---
    #    Expect test_data shape [N, C, H, W]
    test_data = torch.load(test_path)
    test_ds   = TensorDataset(test_data)
    loader    = DataLoader(test_ds,
                           batch_size=args.batch_size,
                           shuffle=False)

    # --- 3) Instantiate diffusion model & load checkpoint ---
    #    (copy‐pasted from your sample_pretrained_diffusion.py)
    ROOT_DIR      = Path(__file__).resolve().parents[2]
    ckpt_path     = ROOT_DIR / "checkpoints" / "diffusion_ckpt.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # --- 4) Compute and print KL ---
    kl_value = estimate_kl_at_T(diffusion, loader, device)
    print(f"Estimated KL[q(x_T) ‖ N(0,I)]  =  {kl_value:.4e}")


if __name__ == "__main__":
    main()
