import torch
import torch.nn as nn
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

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32),
                use_tqdm=True):
        """
        Algorithm 2 in Denoising Diffusion Probabilistic Models
        """
        print("Sampling from diffusion model...")
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]),
                    device=self.device)
        progress_bar = tqdm if use_tqdm else lambda x: x

        # iterate with a plain Python int `time_step`
        for time_step in progress_bar(range(self.T, 0, -1)):
            # build a tensor t from that int
            t = torch.full(
                (n_samples,), time_step,
                dtype=torch.long, device=self.device
            )

            beta_t = self.beta[t - 1].view(-1, 1, 1, 1)
            alpha_t = self.alpha[t - 1].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t - 1].view(-1, 1, 1, 1)

            mean = (1 / torch.sqrt(alpha_t)*(x-((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t))* self.function_approximator(x, t - 1)))
            sigma = torch.sqrt(beta_t)

            # decide on noise using the Python int `time_step`
            if time_step > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = mean + sigma * noise

        return x
