#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from torch.optim import Adam
import logging

from gid_tools.diffusion_model.unet import UNet
from gid_tools.diffusion_model.diffusion import DiffusionModel
from gid_tools.helpers.utils import load_config, download_checkpoint, save_samples
from gid_tools.helpers.plots import plot_tuning_stats, plot_reward_tuning, plot_mean_p_target
from gid_tools.envs.feedback import ToolRewardEnv
from gid_tools.envs.training_functions.classifier.cnn import ToolCNN

import matplotlib.pyplot as plt
import numpy as np


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
                        default=Path("scripts/pipeline/config_no_reward_head.ini"),
                        help="Path to config.ini")
    
    args = parser.parse_args()

    # 1) load cfg
    cfg = load_config(args.config)
    gen_cfg   = cfg["generator"]
    eval_cfg  = cfg["evaluate"]
    ft_cfg    = cfg["fine_tune"]
    device    = torch.device(gen_cfg.get("device", "cuda"))
    
    target_class = ft_cfg.getint("target_class", 4)

    print(f"target class = {target_class}")
    
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
    model_orig.eval() # freeze prior model
    
    diffusion = DiffusionModel(T=1000, model=model, device=device, model_orig=model_orig)  

    # 4) hook up the differentiable CNN as environment
    cnn = ToolCNN(num_classes=5).to(device)

    # ─── load your pretrained classifier weights ───
    model_path = project_root / "gid_tools" /"envs" / "training_functions" / "classifier" / "checkpoints" /"model.pth"
    state = torch.load(model_path, map_location=device)
    cnn.load_state_dict(state)
    logger.info(f"Using CNN ckpt at {model_path}")

    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False
        
    


    # 6) optimizers
    opt_diff = Adam(model.parameters(), lr=ft_cfg.getfloat("lr_diff",1e-3))

    B = ft_cfg.getint("batch_size", 16)
    K = ft_cfg.getint("num_iters",   5)

    avg_true_r = []
    mean_p_targets = []
    
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
                trunc_backprop_steps=2
            ) # returns (x, kl_z_total, kl_Z_total)
           
        save_samples(
            samples=x0, 
            output_dir=OUTPUT_DIR,
            prefix="sample",
            scale_to_uint8=True
        )
                
        # ---- D) direct reward backprop through diffusion ----
         # 1) Create a [B] tensor where every entry == target_class
        labels = torch.full(
            (B,), 
            fill_value=target_class, 
            dtype=torch.long, 
            device=device
        )
                        
        # 2) Forward through frozen CNN and get CE loss
        logits   = cnn(x0)                                    # [B,5]
        predicted_labels = logits.argmax(dim=1)  # [B]
        logger.info(f"Predicted labels: {predicted_labels.tolist()}")

        ce_loss  = F.cross_entropy(logits, labels, reduction="mean")
        
        # # Margin loss (logit difference)
        # margin = 1.0
        # target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # z_y
        # # Soft margin: log-sum-exp over incorrect classes
        # mask = torch.ones_like(logits).bool()
        # mask[torch.arange(B), labels] = False
        # logsumexp_other = torch.logsumexp(logits[mask].view(B, -1), dim=1)
        # margin_loss = F.relu(margin - (target_logits - logsumexp_other)).mean()
        
        # check classifier probabilities
        probs = logits.softmax(-1)              # [B,5]
        mean_p_target = probs[:,target_class].mean().detach().cpu().numpy() 
        logger.info(f"mean p_target={mean_p_target:.3f},")
        mean_p_targets.append(mean_p_target)

        # per_sample = F.cross_entropy(logits, labels, reduction="none")  # [B]
        # logger.info(f"CE per-sample: mean={per_sample.mean():.3f}, "
        #             f"std={per_sample.std():.3f}")

        avg_true_r.append(ce_loss.item())
        # avg_true_r.append(margin_loss.item())
        
        # 3) KL regulariser
        gamma_z = ft_cfg.getfloat("kl_gamma_z_prev", 1.0) # regularisation strength for KL_z (previous timestep)
        gamma_Z  = ft_cfg.getfloat("kl_gamma_z_pre", 1.0) # regularisation strength for KL_Z (pretrained)

        kl_reg = gamma_z * kl_z_total.mean() + gamma_Z * kl_Z_total.mean()
        #kl_reg = gamma_z * kl_z_total + gamma_Z * kl_Z_total
        
         # 4) Total loss: cross-entropy + kl
        diff_loss = ce_loss + kl_reg
        # diff_loss = margin_loss + kl_reg

        # 5) Backprop into diffusion model only
        opt_diff.zero_grad()
        diff_loss.backward()
        
        clip_grad_norm_(model.parameters(), max_norm=0.001) # clip gradients

        opt_diff.step()
        
        # Log gradient norms for both CE and KL terms to ensure the CE gradient isn’t vanishing
        total_ce_grad = sum(p.grad.norm() for p in model.parameters())
        logger.info(f"CE-grad norm={total_ce_grad:.3e}")

        #logger.info(f"[iter {it}/{K}] Margin_loss = {margin_loss:.3f}, regKL= {gamma_z *  kl_z_total.mean()}, {gamma_Z * kl_Z_total.mean()}  total={diff_loss:.3f}")
        logger.info(f"[iter {it}/{K}] CE_loss={ce_loss:.3f},  regKL= {gamma_z * kl_z_total.mean()}, {gamma_Z * kl_Z_total.mean()}  total={diff_loss:.3f}")

        torch.cuda.empty_cache()
        
        # sanity check parameters are changing
        after_norm = par0.data
        
         # compute how much that parameter moved - sanity check
        delta = torch.norm(after_norm - before_norm).item()
        logger.info(f"iter {it:3d} ‖Δparam0‖₂ = {delta:.4e}")

        
        
        torch.cuda.empty_cache()
    


    logger.info("Finished greedy fine-tuning.")
    
    plot_reward_tuning(avg_true_r, K)
    plot_mean_p_target(K, mean_p_targets, target_class)

    

if __name__ == "__main__":
    main()
