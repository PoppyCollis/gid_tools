import torch
import torch.nn as nn
from gid_tools.diffusion_model.unet import UNet

class LinearRewardModel(nn.Module):
    def __init__(self, unet_ckpt_path=None, ch=128, in_ch=1):
        super().__init__()
        # 1) instantiate your U-Net backbone
        self.unet = UNet(ch=ch, in_ch=in_ch)
        if unet_ckpt_path:
            self.unet.load_state_dict(torch.load(unet_ckpt_path))
        self.unet.eval()  # or fine-tune, as you prefer
        
        # placeholder for the middle‐block features
        self._mid_feats = None
        
        # 2) hook the *output* of the last middle block (ResNetBlock)
        #    self.unet.middle is [ResNetBlock, AttentionBlock, ResNetBlock]
        self.unet.middle[2].register_forward_hook(self._capture_mid_feats)
        
        # 3) define your linear head: flatten spatial dims then a scalar reward
        #    assume feature‐map size is [B, C_mid, H_mid, W_mid]
        #    you can read off C_mid = 2*ch, but H_mid/W_mid depend on input size
        self.reward_head = nn.Linear(2*ch * H_mid * W_mid, 1)

    def _capture_mid_feats(self, module, inp, output):
        # output is a Tensor [B, C_mid, H_mid, W_mid]
        self._mid_feats = output

    def forward(self, x, t):
        # run U-Net; _mid_feats is populated by the hook
        _ = self.unet(x, t)
        feats = self._mid_feats  # [B, C_mid, H_mid, W_mid]
        # flatten and score
        return self.reward_head(feats.flatten(start_dim=1))
