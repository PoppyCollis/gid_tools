import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm

class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device: str):
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training_step(self, x0, optimizer):
        """
        Single training step on a batch x0 coming from your DataLoader.
        x0: [B,1,32,32] tensor in [-1,1]
        """
        x0 = x0.to(self.device)
        B = x0.shape[0]
        # sample timesteps and noise
        t   = torch.randint(1, self.T+1, (B,), device=self.device)
        eps = torch.randn_like(x0)

        # compute noisy input
        alpha_bar_t = self.alpha_bar[t-1].view(B, 1, 1, 1)
        noisy_x0    = torch.sqrt(alpha_bar_t) * x0 \
                    + torch.sqrt(1 - alpha_bar_t) * eps

        # predict noise
        eps_pred = self.function_approximator(noisy_x0, t-1)
        loss     = nn.functional.mse_loss(eps, eps_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def p_sample(self, x, t, noise_pred):
        """
        One reverse sampling step in DDPM.
        Args:
            x: current latent at timestep t
            t: tensor of timesteps (shape [B])
            noise_pred: predicted noise ε_θ(x_t, t)
        Returns:
            x_{t-1}: next latent
        """
        beta_t       = self.beta[t - 1].view(-1, 1, 1, 1)
        alpha_t      = self.alpha[t - 1].view(-1, 1, 1, 1)
        alpha_bar_t  = self.alpha_bar[t - 1].view(-1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        sigma = torch.sqrt(beta_t)

        noise = torch.randn_like(x) if (t[0] > 1) else torch.zeros_like(x)
        x_prev = mean + sigma * noise
        return x_prev


    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                use_tqdm=True):
        print("Sampling from diffusion model...")
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                        device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x

        for time_step in progress_bar(range(self.T, 0, -1)):
            t = torch.full((n_samples,), time_step, dtype=torch.long, device=self.device)
            noise_pred = self.function_approximator(x, t - 1)
            x = self.p_sample(x, t, noise_pred)

        return x

    
    def unrolled_sampling(self, n_samples, image_channels=1, img_size=(32, 32),
                      use_tqdm=False, return_all_latents=False,
                      trunc_backprop_steps=50): 

        """
        Differentiable sampling: unrolls the full denoising trajectory with autograd enabled.
        Returns the final x₀ with a computation graph linking back through UNet calls.
        """
        device = self.device
        model = self.function_approximator
        T = self.T

        x = torch.randn((n_samples, image_channels, *img_size), device=device)

        timesteps = list(range(T))[::-1]
        if use_tqdm:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Unrolled Sampling")

        latents = [x]

        for i, t in enumerate(timesteps):  # t = T, T-1, ..., 1
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # detach computation graph if beyond the truncation window
            if i < self.T - trunc_backprop_steps:
                x = x.detach()
                
            # UNet forward with grad
            noise_pred = checkpoint.checkpoint(lambda x_,
                                               t_: model(x_, t_),
                                               x, t_tensor,
                                               use_reentrant=False)  # shape [B, C, H, W]

            # scheduler step
            x = self.p_sample(x, t_tensor, noise_pred=noise_pred)

            if return_all_latents:
                latents.append(x)

        return (x, latents) if return_all_latents else x

        
    
