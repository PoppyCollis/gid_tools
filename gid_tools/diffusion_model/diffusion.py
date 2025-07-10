import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
import warnings

class DiffusionModel:

    def __init__(self, T: int, model: nn.Module, device: str, model_orig: nn.Module = None):
        self.T = T
        self.function_approximator = model.to(device)
        self.function_approximator_orig = model_orig
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
    
    def unrolled_sampling_with_kls(self, n_samples, image_channels=1, img_size=(32, 32),
                        use_tqdm=False, return_all_latents=False,
                        trunc_backprop_steps=50): 
        """
        Differentiable sampling: unrolls the full denoising trajectory with autograd enabled.
        Returns the final x₀ with a computation graph linking back through UNet calls.
        Also computes KL divergence terms:
            - KL to original pretrained model (z_t)
            - KL to previous timestep model (Z_t)
        """
        device = self.device
        model = self.function_approximator
        model_orig = self.function_approximator_orig
        model_orig.eval()

        T = self.T
        x = torch.randn((n_samples, image_channels, *img_size), device=device)
        timesteps = list(range(T))[::-1]

        if use_tqdm:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Unrolled Sampling")

        latents = [x]
        kl_z_accum = []
        kl_Z_accum = []

        noise_pred_prev = None  # for KL with previous timestep

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

            if i < self.T - trunc_backprop_steps:
                x = x.detach()

            # Predict noise from current model
            noise_pred = checkpoint.checkpoint(lambda x_, t_: model(x_, t_),
                                            x, t_tensor, use_reentrant=False)

            # Predict noise from pretrained model (f^0)
            if self.function_approximator_orig is not None:
                with torch.no_grad():
                    noise_pred_orig = self.function_approximator_orig(x, t_tensor)
                kl_z_t = ((noise_pred - noise_pred_orig) ** 2).mean(dim=(1, 2, 3)) / (2 * self.beta[t])
                kl_z_accum.append(kl_z_t)
            else:
                # Append zeros with same batch shape as noise_pred
                zero_kl = torch.zeros(noise_pred.shape[0], device=device, dtype=noise_pred.dtype)
                kl_z_accum.append(zero_kl)
                warnings.warn("No original model provided; skipping z_T computation.")
                    
            # KL with previous: Z_t
            if noise_pred_prev is not None:
                kl_Z_t = ((noise_pred - noise_pred_prev) ** 2).mean(dim=(1, 2, 3)) / (2 * self.beta[t])
                kl_Z_accum.append(kl_Z_t)
            else:
                warnings.warn("No prev model iteration; skipping z_T computation.")


            noise_pred_prev = noise_pred.detach()  # prevent backprop through old drift

            # Step forward
            x = self.p_sample(x, t_tensor, noise_pred=noise_pred)

            if return_all_latents:
                latents.append(x)

        # Stack and sum over time: [T, B] → [B]
        kl_z_total = torch.stack(kl_z_accum).sum(dim=0)
        kl_Z_total = torch.stack(kl_Z_accum).sum(dim=0) if kl_Z_accum else torch.zeros_like(kl_z_total)

        outputs = (x, kl_z_total, kl_Z_total)
        if return_all_latents:
            outputs = outputs + (latents,)
        return outputs

        

        
        
        
        

            
        
