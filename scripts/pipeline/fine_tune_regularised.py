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

import matplotlib.pyplot as plt

# === logger setup ===
logger = logging.getLogger("greedy_finetune")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", 
                        type=Path,
                        default=Path("scripts/pipeline/config.ini"),
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

    # 3) build tunable and prior models
    model   = UNet(ch=128, in_ch=1).to(device) # tunable model
    model_orig = UNet(ch=128, in_ch=1).to(device) # orginal prior diffusion model
    
    # load pretrained diffusion checkpoints
    ckpt     = torch.load(ckpt_path, map_location=device)
    state    = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model_orig.load_state_dict(state)
    
    model.train()
    for p in model.parameters(): # is this redundant given model.train()?
        p.requires_grad = True
    model_orig.eval() 
    
    diffusion = DiffusionModel(T=1000, model=model, device=device, model_orig=model_orig)    

    # 4) hook up the reward-env
    env = ToolRewardEnv(default_method=eval_cfg.get("method","cnn_prob"))

    # 5) set up feature extractor & reward-head
    extractor = UnetFeatureExtractor(unet_ckpt_path=ckpt_path,
                                      ch=128, in_ch=1, device=device)
    
    input_dim = 256*4*4 # greyscale x unet feature dims
    reward_head= RewardMLP(input_dim=input_dim).to(device)

    # 6) optimizers
    opt_diff = Adam(model.parameters(), lr=ft_cfg.getfloat("lr_diff",1e-5))
    opt_rew  = Adam(reward_head.parameters(), lr=ft_cfg.getfloat("lr_rew",1e-3))
    criterion = MSELoss()

    B = ft_cfg.getint("batch_size", 16)
    K = ft_cfg.getint("num_iters",   5)
    epochs = ft_cfg.getint("reward_epochs", 5)

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
        
        OUTPUT_DIR = Path(ft_cfg.get("output_dir", "regularised_outputs")) / f"iter_{it:03d}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        
        # ---- A) sample a batch end-to-end ----
        # this returns a (B,1,H,W) tensor with grad

        # here I want to accumulate the KL-terms during unrolled sampling
        # run for diffusion and diffusion_orig
        
        with torch.set_grad_enabled(True):
            x0, kl_z_total, kl_Z_total = diffusion.unrolled_sampling_with_kls(
                n_samples=B,
                use_tqdm=True, 
                return_all_latents=False,
                trunc_backprop_steps=50
            ) # returns (x, kl_z_total, kl_Z_total)
           
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
        
        # add current features and rewards to the full dataset
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
        
        # reward term in loss
        pred_r = reward_head(fx)             # [B]
        
        # add in KL terms to loss
        gamma_z = ft_cfg.getfloat("kl_gamma_z_prev", 1.0) # regularisation strength for KL_z (previous timestep)
        gamma_Z  = ft_cfg.getfloat("kl_gamma_z_pre", 1.0) # regularisation strength for KL_Z (pretrained)

        kl_reg = gamma_z * kl_z_total.mean() + gamma_Z * kl_Z_total.mean()
        diff_loss = - pred_r.mean() + kl_reg
        

        opt_diff.zero_grad()
        diff_loss.backward()
        opt_diff.step()
        
        # sanity check parameters are changing
        after_norm = par0.data
        
         # compute how much that parameter moved - sanity check
        delta = torch.norm(after_norm - before_norm).item()
        logger.info(f"iter {it:3d} ‖Δparam0‖₂ = {delta:.4e}  diff_loss={diff_loss.item():.3f}")


        logger.info(f"[iter {it}/{K}] reward_head MSE={criterion(reward_head(fx),yt).item():.3f}")
        
        logger.info(f"[iter {it}/{K}], avg true = {yt.mean().item()}, avg_pred = {pred_r.mean().item()}")
        
        torch.cuda.empty_cache()
        
    print(avg_pred_r, avg_true_r)


    logger.info("Finished greedy fine-tuning.")
    
    plot_tuning_stats(K, avg_true_r, std_true_r, avg_pred_r, std_pred_r)

if __name__ == "__main__":
    main()
