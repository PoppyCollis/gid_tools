#!/usr/bin/env python
import argparse
from pathlib import Path
from copy import deepcopy
import pandas as pd



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

<<<<<<< HEAD
import os
import matplotlib.pyplot as plt

def plot_reward_tuning(df, out_dir=None, fname="reward_tuning.png"):
    """
    Plot both post-update p_target and CE loss vs learning rate,
    for each (gamma_prev, gamma_pre) combination, and optionally save as PNG.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'lr', 'gamma_prev', 'gamma_pre', 'post_p', 'ce_loss'.
    out_dir : str or pathlib.Path, optional
        Directory to which to save the PNG. If None, no file is written.
    fname : str, default "reward_tuning.png"
        Filename for the saved figure.
    """
    # Group by the two KL hyperparameters
    groups = df.groupby(['gamma_prev', 'gamma_pre'])
    
    # Create plot
    fig, ax1 = plt.subplots()
    # Left y-axis: post-update probability
    for (g_prev, g_pre), grp in groups:
        grp = grp.sort_values('lr')
        ax1.plot(
            grp['lr'], grp['post_p'],
            marker='o',
            label=f'p: γ_prev={g_prev}, γ_pre={g_pre}'
        )
=======
import matplotlib.pyplot as plt
import numpy as np

def plot_reward_tuning(df):
    """
    Plot both post-update p_target and CE loss vs learning rate,
    for each (gamma_prev, gamma_pre) combination.
    """
    groups = df.groupby(['gamma_prev', 'gamma_pre'])
    fig, ax1 = plt.subplots()

    # Plot p_target on the left y-axis
    for (g_prev, g_pre), grp in groups:
        grp = grp.sort_values('lr')
        ax1.plot(grp['lr'], grp['post_p'], marker='o',
                 label=f'p: γ_prev={g_prev}, γ_pre={g_pre}')
>>>>>>> 10b5743 (sweep over learning rate to see effect on prob target label)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning rate')
    ax1.set_ylabel('Post-update p_target')
    ax1.legend(loc='upper left')
<<<<<<< HEAD
    
    # Right y-axis: CE loss
    ax2 = ax1.twinx()
    for (g_prev, g_pre), grp in groups:
        grp = grp.sort_values('lr')
        ax2.plot(
            grp['lr'], grp['ce_loss'],
            marker='x', linestyle='--',
            label=f'loss: γ_prev={g_prev}, γ_pre={g_pre}'
        )
    ax2.set_ylabel('Post-update CE loss')
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    plt.show()
    
    # Save to file if requested
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, fname)
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")
=======

    # Plot CE loss on the right y-axis
    ax2 = ax1.twinx()
    for (g_prev, g_pre), grp in groups:
        grp = grp.sort_values('lr')
        ax2.plot(grp['lr'], grp['ce_loss'], marker='x', linestyle='--',
                 label=f'loss: γ_prev={g_prev}, γ_pre={g_pre}')
    ax2.set_ylabel('Post-update CE loss')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()
>>>>>>> 10b5743 (sweep over learning rate to see effect on prob target label)


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
    
    it = 0
    
    OUTPUT_DIR = Path(ft_cfg.get("output_dir", "regularised_outputs")) / f"iter_{it:03d}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # TUNING
    
    gamma_prev = ft_cfg.getfloat("kl_gamma_z_prev", 1.0) # regularisation strength for KL_z (previous timestep)
    gamma_pre  = ft_cfg.getfloat("kl_gamma_z_pre", 1.0) # regularisation strength for KL_Z (pretrained)
    
    # 1. Get first set of samples
    with torch.set_grad_enabled(True):
            x0_init, kl_z_init, kl_Z_init = diffusion.unrolled_sampling_with_kls(
                n_samples=B,
                use_tqdm=True, 
                return_all_latents=False,
                trunc_backprop_steps=2
            ) # returns (x, kl_z_total, kl_Z_total)
        
        
    # save generated tools       
    save_samples(
        samples=x0_init, 
        output_dir=OUTPUT_DIR,
        prefix="sample",
        scale_to_uint8=True
    )
    
    labels = torch.full(
        (B,), 
        fill_value=target_class, 
        dtype=torch.long, 
        device=device
    )
    
   # Compute and store the initial CE loss if you want to log it later
    logits  = cnn(x0_init)                                # frozen classifier
    ce_loss = F.cross_entropy(logits, labels)
    pre_p   = logits.softmax(-1)[:, target_class].mean().item()
    kl_reg  = gamma_prev * kl_z_init.mean() + gamma_pre * kl_Z_init.mean()
    diff_loss = ce_loss + kl_reg                      # this has grad_fn
    
    print("pre_tuning target probability", pre_p)
    # Compute and store gradients for each model param
    grads = torch.autograd.grad(
        outputs=diff_loss, 
        inputs=list(model.parameters()), 
        create_graph=False,    # we don’t need higher‐order grads
        retain_graph=False     # free the graph afterwards
    )
    # grads is a tuple of Tensors matching model.parameters()

    # snapshot the untouched model weights
    initial_state = {n: p.data.clone() 
                    for n, p in model.named_parameters()}

    lr_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    
    # 2. Grid-search loop — everything else is recomputed each trial ---
    results = []
    for lr in lr_list:
            ## a) reload original weights
        for (n, p) in model.named_parameters():
            p.data.copy_(initial_state[n])
            
        opt = Adam(model.parameters(), lr=lr)
        
        # b) apply one manual gradient step
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.data.sub_(lr * g)
                
         # c) sample from the updated model
        with torch.set_grad_enabled(True):
            x0, kl_z_total, kl_Z_total = diffusion.unrolled_sampling_with_kls(
                n_samples=B,
                use_tqdm=True, 
                return_all_latents=False,
                trunc_backprop_steps=2
            )
            
        logits1  = cnn(x0)                                # frozen classifier
        ce_loss1 = F.cross_entropy(logits, labels).item()
        post_p   = logits1.softmax(-1)[:, target_class].mean().item()
<<<<<<< HEAD
        print(f"post_p: {post_p}, lr: {lr}")
=======
        
>>>>>>> 10b5743 (sweep over learning rate to see effect on prob target label)
        
        results.append({
            "lr": lr,
            "gamma_prev": gamma_prev,
            "gamma_pre":  gamma_pre,
            "ce_loss":    ce_loss1,
            "post_p":     post_p
        })
                
    df = pd.DataFrame(results)
<<<<<<< HEAD
    out_dir = Path("hyperparam_plots")
    plot_reward_tuning(df, out_dir=out_dir, fname="sweep1.png")
    
=======
    plot_reward_tuning(df)   # or any custom plotting you like
>>>>>>> 10b5743 (sweep over learning rate to see effect on prob target label)
    
if __name__ == "__main__":
    main()