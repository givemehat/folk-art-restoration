# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
models/lama.py
--------------
Simplified LaMa-style (Large Mask) inpainting GAN.

Generator
---------
  U-Net with:
    - Dilated convolutions in the encoder for a large receptive field
    - Fast Fourier Convolution (FFC) blocks in the bottleneck
    - Skip connections encoder → decoder
  Input  : (B, 4, H, W)  — damaged RGB (3ch) concatenated with binary mask (1ch)
  Output : (B, 3, H, W)  — restored RGB image in [0, 1]

Discriminator
-------------
  PatchGAN (70×70 receptive field)
  Input  : (B, 4, H, W)  — image (3ch) + mask (1ch)

Reference: Suvorov et al. "Resolution-robust Large Mask Inpainting with
Fourier Convolutions." WACV 2022.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================================
# Utility layers
# ==========================================================================


def _norm(num_ch: int):
    """Instance normalisation used throughout both G and D."""
    return nn.InstanceNorm2d(num_ch, affine=True)


def _act():
    """Default activation: LeakyReLU(0.2) for stability."""
    return nn.LeakyReLU(0.2, inplace=True)


class ConvBnAct(nn.Module):
    """Conv2d → Norm → Act convenience block."""

    def __init__(
        self, in_ch, out_ch, kernel=3, stride=1, dilation=1, pad_mode="reflect"
    ):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel,
                stride=stride,
                padding=pad,
                dilation=dilation,
                padding_mode=pad_mode,
            ),
            _norm(out_ch),
            _act(),
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================================
# Fast Fourier Convolution (FFC)
# ==========================================================================


class SpectralTransform(nn.Module):
    """
    Applies a 1×1 convolution in the frequency domain (via rFFT2d).
    Captures global context without increasing spatial kernel size.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch * 2, out_ch * 2, kernel_size=1)
        self.bn = _norm(out_ch * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Real-valued FFT along spatial dims
        fft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1) complex
        # Decompose into real and imaginary parts and stack along channel dim
        fft_r = torch.view_as_real(fft)  # (B, C, H, W//2+1, 2)
        fft_cat = fft_r.permute(0, 1, 4, 2, 3).reshape(  # (B, 2C, H, W//2+1)
            B, C * 2, H, fft.shape[-1]
        )
        fft_cat = self.act(self.bn(self.conv(fft_cat)))
        # Reshape back
        fft_out = fft_cat.reshape(B, C, 2, H, fft.shape[-1]).permute(0, 1, 3, 4, 2)
        fft_out = torch.view_as_complex(fft_out.contiguous())
        # Inverse FFT → spatial domain
        out = torch.fft.irfft2(fft_out, s=(H, W), norm="ortho")
        return out


class FFCResBlock(nn.Module):
    """
    FFC residual block: split channels into 'local' (spatial conv) and
    'global' (spectral) branches, then sum them.
    """

    def __init__(self, num_ch: int, ratio_global: float = 0.5):
        super().__init__()
        self.g = max(1, int(num_ch * ratio_global))
        self.l = num_ch - self.g

        # Local branch (standard 3×3 conv)
        if self.l > 0:
            self.local_conv = nn.Sequential(
                nn.Conv2d(self.l, self.l, 3, padding=1, padding_mode="reflect"),
                _norm(self.l),
                _act(),
                nn.Conv2d(self.l, self.l, 3, padding=1, padding_mode="reflect"),
                _norm(self.l),
            )

        # Global branch (spectral transform)
        self.global_st = SpectralTransform(self.g, self.g)

        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = x[:, : self.l]  # local channels
        xg = x[:, self.l :]  # global channels

        # Local
        if self.l > 0:
            xl = xl + self.local_conv(xl)

        # Global
        xg = xg + self.global_st(xg)

        out = torch.cat([xl, xg], dim=1)
        return self.act(out)


# ==========================================================================
# Attention Gate
# ==========================================================================


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    Filters the skip connection features (from encoder) using the gating signal
    (from the coarser decoder layer).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
        F_g   : number of channels in gating signal (coarser decoder layer)
        F_l   : number of channels in skip connection (encoder layer)
        F_int : number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_in = self.W_g(g)
        x_in = self.W_x(x)
        combined = self.relu(g_in + x_in)
        attn = self.psi(combined)
        return x * attn


# ==========================================================================
# Generator (U-Net with FFC bottleneck)
# ==========================================================================


class Generator(nn.Module):
    """
    U-Net generator with:
      Encoder: dilated convolutions for large receptive field
      Bottleneck: FFC residual blocks
      Decoder: bilinear up + conv, skip connections from encoder

    Input  : (B, 4, H, W)  RGB damaged + mask
    Output : (B, 3, H, W)  restored image in [0, 1]
    """

    def __init__(self, base_ch: int = 64, n_ffc: int = 4, use_attention: bool = True):
        """
        Args:
        base_ch       : base number of channels (doubles with each encoder level).
        n_ffc         : number of FFC residual blocks in the bottleneck.
        use_attention : if True, use Attention Gates on skip connections.
        """
        super().__init__()
        self.use_attention = use_attention

        # ------------------------------------------------------------------ Encoder
        # enc1 : 4  → base_ch        dilation 1
        # enc2 : base_ch → base_ch*2 dilation 2 (stride 2 for downsampling)
        # enc3 : base_ch*2 → base_ch*4 dilation 4 (stride 2)
        # enc4 : base_ch*4 → base_ch*8 dilation 2 (stride 2)
        self.enc1 = ConvBnAct(4, base_ch, dilation=1)
        self.enc2 = ConvBnAct(base_ch, base_ch * 2, dilation=2, stride=2)
        self.enc3 = ConvBnAct(base_ch * 2, base_ch * 4, dilation=4, stride=2)
        self.enc4 = ConvBnAct(base_ch * 4, base_ch * 8, dilation=2, stride=2)

        # ------------------------------------------------------------------ Bottleneck (FFC)
        self.bottleneck = nn.Sequential(
            *[FFCResBlock(base_ch * 8) for _ in range(n_ffc)]
        )

        # ------------------------------------------------------------------ Decoder
        # Each decoder block: bilinear 2× upsample → conv
        # Skip connections from encoder (channel concat) → adjust in_ch
        self.dec4 = ConvBnAct(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec3 = ConvBnAct(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec2 = ConvBnAct(base_ch * 2 + base_ch, base_ch)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh(),  # output in [-1, 1]; we rescale to [0,1] in forward()
        )

        # ------------------------------------------------------------------ Attention Gates
        if self.use_attention:
            self.attn4 = AttentionGate(
                F_g=base_ch * 8, F_l=base_ch * 4, F_int=base_ch * 4
            )
            self.attn3 = AttentionGate(
                F_g=base_ch * 4, F_l=base_ch * 2, F_int=base_ch * 2
            )
            self.attn2 = AttentionGate(F_g=base_ch * 2, F_l=base_ch, F_int=base_ch)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample x to match target's spatial size."""
        return F.interpolate(
            x, size=target.shape[2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x : (B, 4, H, W) — degraded image (3 ch) + binary mask (1 ch) in [0, 1]

        Returns:
        (B, 3, H, W) restored image in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 64,   H,   W)
        e2 = self.enc2(e1)  # (B, 128,  H/2, W/2)
        e3 = self.enc3(e2)  # (B, 256,  H/4, W/4)
        e4 = self.enc4(e3)  # (B, 512,  H/8, W/8)

        # Bottleneck
        b = self.bottleneck(e4)  # (B, 512,  H/8, W/8)

        # Decoder with skip connections (optionally gated with Attention)
        b_up = self._up(b, e3)
        if self.use_attention:
            e3_gate = self.attn4(b_up, e3)
        else:
            e3_gate = e3
        d4 = self.dec4(torch.cat([b_up, e3_gate], dim=1))

        d4_up = self._up(d4, e2)
        if self.use_attention:
            e2_gate = self.attn3(d4_up, e2)
        else:
            e2_gate = e2
        d3 = self.dec3(torch.cat([d4_up, e2_gate], dim=1))

        d3_up = self._up(d3, e1)
        if self.use_attention:
            e1_gate = self.attn2(d3_up, e1)
        else:
            e1_gate = e1
        d2 = self.dec2(torch.cat([d3_up, e1_gate], dim=1))

        out = self.dec1(d2)  # (B, 3, H, W)  in [-1, 1]

        # Re-scale from [-1, 1] to [0, 1]
        return (out + 1.0) / 2.0


# ==========================================================================
# Discriminator (PatchGAN, ~70×70 receptive field)
# ==========================================================================


class PatchDiscriminator(nn.Module):
    """
    Standard PatchGAN discriminator with 4 Conv layers.
    Receptive field ≈ 70×70.
    Input: (B, 4, H, W) — image (3ch) + mask (1ch)
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        # No normalisation in the first layer (common practice in PatchGAN)
        self.net = nn.Sequential(
            # Layer 1: 4 → 64,   stride 2  (no norm)
            nn.Conv2d(4, base_ch, 4, stride=2, padding=1),
            _act(),
            # Layer 2: 64 → 128,  stride 2
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            _norm(base_ch * 2),
            _act(),
            # Layer 3: 128 → 256,  stride 2
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            _norm(base_ch * 4),
            _act(),
            # Layer 4: 256 → 512,  stride 1
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1),
            _norm(base_ch * 8),
            _act(),
            # Output: 512 → 1,  stride 1 (patch logits)
            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
        img  : (B, 3, H, W) image in [0, 1]
        mask : (B, 1, H, W) binary mask

        Returns:
        (B, 1, Ph, Pw) patch logits (no sigmoid — used with hinge loss)
        """
        x = torch.cat([img, mask], dim=1)  # (B, 4, H, W)
        return self.net(x)


# ==========================================================================
# Inpainting Dataset helper
# ==========================================================================

from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np


class FolkArtInpaintDataset(Dataset):
    """
    Dataset for LaMa inpainting training.
    Returns dicts with keys: 'clean', 'damaged', 'mask'.

    Folder layout (output of create_damaged_dataset()):
      root/clean/<split>/<name>.png
      root/damaged/<split>/<name>.png
      root/masks/<split>/<name>.png

    Parameters
    ----------
    root   : dataset root
    split  : 'train', 'val', or 'test'
    size   : spatial size to resize all images to (default 256)
    augment: apply random h-flip augmentation (train only)
    """

    def __init__(
        self, root: str, split: str = "train", size: int = 256, augment: bool = False
    ):
        super().__init__()
        root = Path(root)
        self.clean_dir = root / "clean" / split
        self.damaged_dir = root / "damaged" / split
        self.mask_dir = root / "masks" / split
        self.size = size
        self.augment = augment

        if not self.clean_dir.exists():
            raise FileNotFoundError(self.clean_dir)

        self.filenames = sorted(p.name for p in self.clean_dir.glob("*.png"))
        if not self.filenames:
            raise FileNotFoundError(f"No .png files in {self.clean_dir}")

        self.img_tf = T.Compose(
            [
                T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),
            ]
        )
        self.mask_tf = T.Compose(
            [
                T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]

        clean = Image.open(self.clean_dir / fname).convert("RGB")
        damaged = Image.open(self.damaged_dir / fname).convert("RGB")

        # Mask may be stored as grayscale
        mask_path = self.mask_dir / fname
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # Fallback: all-ones mask (whole image)
            mask = Image.new("L", clean.size, 255)

        # Random horizontal flip augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
            damaged = damaged.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random 90/180/270-degree rotation augmentation (safe for pixel grid alignment)
        if self.augment and torch.rand(1).item() > 0.5:
            rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            clean = clean.transpose(rot)
            damaged = damaged.transpose(rot)
            mask = mask.transpose(rot)

        c = self.img_tf(clean)  # (3, H, W) [0,1]
        d = self.img_tf(damaged)  # (3, H, W)
        m = self.mask_tf(mask)  # (1, H, W) {0,1}
        m = (m > 0.5).float()  # binarise

        return {"clean": c, "damaged": d, "mask": m, "filename": fname}
