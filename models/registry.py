# ==============================================================================
# Indian Folk Art Restoration AI Pipeline: Model Registry
# ------------------------------------------------------------------------------
# Unified Factory Pattern for loading Super-Resolution and Inpainting models
# via reproducible YAML/JSON configurations for research ablation and benchmarks.
# ==============================================================================

import yaml
from pathlib import Path
import torch
import torch.nn as nn

from models.edsr import EDSR
from models.lama import Generator as LaMaGenerator
from models.baseline_unet import VanillaUNetInpainter


class BicubicUpsampler(nn.Module):
    """Classical Bicubic baseline wrapper matching PyTorch Module interface."""
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )


class ChannelAttentionSR(nn.Module):
    """
    Lightweight Channel Attention Super-Resolution (RCAN-style residual block).
    Serves as an Attention-based SR alternative to standard EDSR.
    """
    def __init__(self, in_ch: int = 3, num_feat: int = 64, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, padding=1)
        
        # Channel Attention Block
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        
        # Squeeze-and-Excitation channel gating
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 4, num_feat, 1),
            nn.Sigmoid()
        )
        
        # Upsampling block
        self.up = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(num_feat, 3, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv_first(x)
        res = self.conv2(self.relu(self.conv1(f)))
        res = res * self.ca(res)
        out = f + res
        return self.up(out)


class ModelRegistry:
    """
    Unified Factory Pattern for loading Super-Resolution and Inpainting models
    via YAML configurations for reproducible research experiments.
    """
    def __init__(self, config_path: str = None, config_dict: dict = None):
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

    def build_sr_model(self, device: torch.device):
        """Constructs and returns the configured Super-Resolution model."""
        sr_cfg = self.config.get("super_resolution", {})
        name = sr_cfg.get("name", "edsr").lower()
        scale = sr_cfg.get("scale", 2)

        if name == "edsr":
            model = EDSR(scale=scale, num_blocks=sr_cfg.get("num_blocks", 16))
        elif name == "bicubic":
            model = BicubicUpsampler(scale=scale)
        elif name in ["rcan", "channel_attention", "swinir_proxy"]:
            model = ChannelAttentionSR(scale=scale)
        else:
            raise ValueError(f"Unknown SR model in registry: {name}. Supported: ['edsr', 'bicubic', 'rcan']")

        return model.to(device)

    def build_inpaint_model(self, device: torch.device):
        """Constructs and returns the configured Inpainting model."""
        inp_cfg = self.config.get("inpainting", {})
        name = inp_cfg.get("name", "lama").lower()

        if name == "lama":
            # Fast Fourier Convolution + Attention Gate U-Net
            model = LaMaGenerator(
                base_ch=inp_cfg.get("base_ch", 64),
                n_ffc=inp_cfg.get("n_ffc", 4),
                use_attention=inp_cfg.get("use_attention", True),
            )
        elif name in ["lama_no_attn", "lama_no_attention"]:
            # Ablation variant: Fourier Convolutions without Attention Gates
            model = LaMaGenerator(
                base_ch=inp_cfg.get("base_ch", 64),
                n_ffc=inp_cfg.get("n_ffc", 4),
                use_attention=False,
            )
        elif name in ["unet", "vanilla_unet", "spatial_unet"]:
            # Ablation baseline: Pure spatial convolutions without spectral transforms
            model = VanillaUNetInpainter(base_ch=inp_cfg.get("base_ch", 64))
        elif name in ["stable_diffusion", "diffusion"]:
            try:
                from models.diffusion import StableDiffusionInpainter
                model = StableDiffusionInpainter(device=device)
            except Exception as e:
                print(f"[Warning] Stable Diffusion pipeline unavailable ({e}). Falling back to LaMa Generator.")
                model = LaMaGenerator(use_attention=True)
        else:
            raise ValueError(f"Unknown Inpainting model in registry: {name}. Supported: ['lama', 'lama_no_attn', 'unet', 'diffusion']")

        return model.to(device)
