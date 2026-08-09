# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
models/edsr.py
--------------
Enhanced Deep Super-Resolution (EDSR) network.

Architecture (from Lim et al. 2017, "Enhanced Deep Residual Networks
for Single Image Super-Resolution"):
  - 16 residual blocks, each with 2 × Conv3x3 + ReLU (no BatchNorm)
  - 64 feature channels throughout
  - Pixel-shuffle upsampling (PixelShuffle × scale_factor)
  - Scale factors supported: 2×, 4×

Input  : low-resolution degraded image  (B, 3, H/scale, W/scale) in [0,1]
Output : high-resolution restored image (B, 3, H, W)             in [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """
    EDSR residual block — two Conv3×3 layers with ReLU in between,
    NO batch normalisation, with a residual scaling factor for stability.
    """

    def __init__(self, num_channels: int = 64, res_scale: float = 0.1):
        """
        Args:
        num_channels : number of feature channels.
        res_scale    : multiply residual branch by this scalar (helps
                       convergence when stacking many blocks).
        """
        super().__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.block(x)


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling (PixelShuffle).
    Converts (B, C, H, W) → (B, C, H*scale, W*scale).
    Supports scale = 2 or 4 (two consecutive 2× steps for 4×).
    """

    def __init__(self, num_channels: int, scale: int):
        super().__init__()
        assert scale in (2, 4), "scale must be 2 or 4"

        layers = []
        # For 4× we stack two 2× pixel-shuffle blocks
        for _ in range(scale // 2):
            layers += [
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # channels ÷ 4, spatial × 2
                nn.ReLU(inplace=True),
            ]
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# Full EDSR model
# ---------------------------------------------------------------------------


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution network.

    Parameters
    ----------
    scale        : upsampling factor (2 or 4).
    num_channels : feature map width (default 64, paper uses 256 for large).
    num_blocks   : number of residual blocks (default 16).
    res_scale    : residual scaling in each ResBlock.
    """

    def __init__(
        self,
        scale: int = 2,
        num_channels: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale

        # --- Head: initial feature extraction ----------------------------
        self.head = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)

        # --- Body: residual tower ----------------------------------------
        self.body = nn.Sequential(
            *[ResBlock(num_channels, res_scale) for _ in range(num_blocks)],
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

        # --- Tail: upsample + reconstruction -----------------------------
        self.tail = nn.Sequential(
            UpsampleBlock(num_channels, scale),
            nn.Conv2d(num_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x : (B, 3, H, W) low-resolution image in [0, 1].

        Returns:
        (B, 3, H*scale, W*scale) super-resolved image clamped to [0, 1].
        """
        # Head
        feat = self.head(x)
        # Body with long skip connection
        feat = feat + self.body(feat)
        # Tail
        out = self.tail(feat)
        return out.clamp(0.0, 1.0)

    def load_checkpoint(self, path: str, device: torch.device = None) -> None:
        """Convenience wrapper to load a saved state dict."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        # Support both raw state-dict saves and wrapped checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.load_state_dict(state)
        print(f"EDSR weights loaded from {path}")


# ---------------------------------------------------------------------------
# Dataset helper (paired low/high resolution)
# ---------------------------------------------------------------------------

from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class FolkArtSRDataset(Dataset):
    """
    Dataset that returns (low_res, high_res) pairs for EDSR training.
    'high_res' is the clean 256×256 image.
    'low_res'  is the damaged image downsampled to 256/scale × 256/scale.

    Folder layout expected:
      root/clean/<split>/<image>.png
      root/damaged/<split>/<image>.png

    Parameters
    ----------
    root   : dataset root folder (output of create_damaged_dataset).
    split  : 'train', 'val', or 'test'.
    scale  : downsampling / upsampling factor for EDSR.
    """

    def __init__(self, root: str, split: str = "train", scale: int = 2):
        super().__init__()
        self.scale = scale
        root = Path(root)
        self.clean_dir = root / "clean" / split
        self.damaged_dir = root / "damaged" / split

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean dir not found: {self.clean_dir}")

        self.filenames = sorted([p.name for p in self.clean_dir.glob("*.png")])
        if not self.filenames:
            raise FileNotFoundError(f"No PNG files in {self.clean_dir}")

        self.hr_size = 256
        self.lr_size = self.hr_size // scale

        # High-resolution transform: normalise to [0,1]
        self.hr_tf = T.Compose(
            [
                T.Resize(
                    (self.hr_size, self.hr_size),
                    interpolation=T.InterpolationMode.LANCZOS,
                ),
                T.ToTensor(),
            ]
        )
        # Low-resolution: take the damaged image & resize to lr_size
        self.lr_tf = T.Compose(
            [
                T.Resize(
                    (self.lr_size, self.lr_size),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]

        hr_img = Image.open(self.clean_dir / fname).convert("RGB")
        lr_img = Image.open(self.damaged_dir / fname).convert("RGB")

        hr = self.hr_tf(hr_img)  # (3, 256, 256)
        lr = self.lr_tf(lr_img)  # (3, 128, 128) for scale=2

        return {"lr": lr, "hr": hr, "filename": fname}
