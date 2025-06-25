#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import logging

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import load_config, download_checkpoint, save_samples
from gid_tools.helpers.plots import plot_tuning_stats
from gid_tools.envs.feedback import ToolRewardEnv
from gid_tools.reward_model.feature_extractor import UnetFeatureExtractor
from gid_tools.reward_model.reward_mlp import RewardMLP

# === logger setup ===
logger = logging.getLogger("greedy_finetune")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config.ini")
    args = parser.parse_args()

    # 1) load cfg
    cfg = load_config(args.config)
    gen_cfg   = cfg["generator"]
    eval_cfg  = cfg["evaluate"]
    ft_cfg    = cfg["fine_tune"]
    device    = torch.device(gen_cfg.get("device", "cuda"))
    
    
    # 2) download + load diffusion checkpoint
    project_root = args.config.resolve().parents[2]
    ckpt_path    = download_checkpoint(project_root)
    logger.info(f"Using diffusion ckpt at {ckpt_path}")

    # 3) build models
    model   = UNet(ch=128, in_ch=1).to(device)
    diffusion = DiffusionModel(T=1000, model=model, device=device)
    ckpt     = torch.load(ckpt_path, map_location=device)
    state    = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    # 4) hook up the reward-env
    env = ToolRewardEnv(default_method=eval_cfg.get("method","pixel_area"))

    # 5) set up feature extractor & reward-head
    extractor  = UnetFeatureExtractor(unet_ckpt_path=ckpt_path,
                                      ch=128, in_ch=1, device=device)
    reward_head= RewardMLP(input_dim=2*128).to(device)

    # 6) optimizers
    opt_diff   = Adam(model.parameters(), lr=ft_cfg.getfloat("lr_diff",1e-5))
    opt_rew    = Adam(reward_head.parameters(), lr=ft_cfg.getfloat("lr_rew",1e-3))
    criterion  = MSELoss()

    B       = ft_cfg.getint("batch_size", 16)
    K       = ft_cfg.getint("num_iters",   5)
    epochs  = ft_cfg.getint("reward_epochs", 5)

    all_feats = []
    all_yt = []
    
    avg_pred_r = []
    std_pred_r = []
    avg_true_r = []
    std_true_r = []
    
    # check parameters are changing
    par0 = next(model.parameters())
    
    for it in range(1, K+1):
        
        # check parameters are changing
        before_norm = par0.data.clone()
        
        OUTPUT_DIR = Path(ft_cfg.get("output_dir", "greedy_outputs")) / f"iter_{it:03d}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        
        # ---- A) sample a batch end-to-end ----
        # this returns a (B,1,H,W) tensor with grad
        x0 = diffusion.sampling(n_samples=B,
                                 image_channels=1,
                                 img_size=(32,32),
                                 use_tqdm=True,
                                 require_grad=True) \
                       .to(device)
        
        save_samples(
            samples=x0, 
            output_dir=OUTPUT_DIR,
            prefix="sample",
            scale_to_uint8=True
        )

        # ---- B) ground-truth reward feedback ----
        # move to cpu, loop in Python, get list of floats
        yt = []
        for img in x0.detach().cpu():
            r = env.compute(img, method=eval_cfg.get("method"),
                            threshold=eval_cfg.getfloat("threshold",0.0))
            yt.append(r)
        yt = torch.tensor(yt, dtype=torch.float32, device=device)  # [B]

        # ---- C) extract features & train reward head ----
        feats = []
        with torch.no_grad():
            t0 = torch.zeros((B,),dtype=torch.long,device=device)
            feats = extractor.extract(x0, t=t0)  # [B, 256]
            pred_r = reward_head(feats) # [B]
        
        avg_true_r.append( yt.mean().item())
        std_true_r.append( yt.std().item())
        avg_pred_r.append( pred_r.mean().item())
        std_pred_r.append( pred_r.std().item())
            
        all_feats.append(feats)
        all_yt.append(yt)
        
        feats_cat = torch.cat(all_feats, dim=0)  # [N, 256]
        yt_cat    = torch.cat(all_yt,    dim=0)  # [N]
        ds = TensorDataset(feats_cat, yt_cat)
        
        # build tiny dataset
        dl = DataLoader(ds, batch_size=ft_cfg.getint("reward_batch_size", B),
                        shuffle=True, num_workers=0)
        reward_head.train()
        for ep in range(epochs):
            for fx, fy in dl:
                pred = reward_head(fx)
                loss = criterion(pred, fy)
                opt_rew.zero_grad()
                loss.backward()
                opt_rew.step()

        # ---- D) direct reward backprop through diffusion ----
        # now treat reward_head∘extractor as a differentiable surrogate
        reward_head.eval()
        t0 = torch.zeros((B,),dtype=torch.long,device=device)
        fx = extractor.extract(x0, t=t0)      # [B,256]
        pred_r = reward_head(fx)             # [B]
        diff_loss = - pred_r.mean()          # maximize reward

        opt_diff.zero_grad()
        diff_loss.backward()
        opt_diff.step()
        
        # check parameters are changing
        after_norm = par0.data
        
         # compute how much that parameter moved
        delta = torch.norm(after_norm - before_norm).item()
        logger.info(f"iter {it:3d} ‖Δparam0‖₂ = {delta:.4e}  diff_loss={diff_loss.item():.4f}")


        logger.info(f"[iter {it}/{K}] reward_head MSE={criterion(reward_head(fx),yt).item():.3f}"
                    + f"   diff_loss = {diff_loss.item():.3f}")
        
        torch.cuda.empty_cache()


    logger.info("Finished greedy fine-tuning.")
    
    plot_tuning_stats(K, avg_true_r, std_true_r, avg_pred_r, std_pred_r)

if __name__ == "__main__":
    main()
