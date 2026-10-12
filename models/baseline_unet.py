# ==============================================================================
# Indian Folk Art Restoration AI Pipeline: Baseline Architecture
# ------------------------------------------------------------------------------
# Vanilla Spatial U-Net Inpainting Baseline (without FFC spectral transform)
# Provides rigorous empirical ablation comparing spatial vs spectral convolutions.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Conv2d -> InstanceNorm2d -> LeakyReLU block."""
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        pad = dilation * 2 // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=pad, dilation=dilation, padding_mode="reflect"),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VanillaUNetInpainter(nn.Module):
    """
    Standard Spatial U-Net Inpainting Network (no Fourier / Spectral blocks).
    Serves as the primary spatial baseline to rigorously isolate and justify the
    contribution of Fast Fourier Convolutions (FFC) and Attention Gates in LaMa.
    
    Input:  (B, 4, H, W) [Damaged RGB (3) + Binary Mask (1)]
    Output: (B, 3, H, W) [Restored RGB in [0, 1]]
    """
    def __init__(self, base_ch: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(4, base_ch, stride=1)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, stride=2)

        # Standard Spatial Bottleneck (Residual blocks without spectral transforms)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8),
        )

        # Decoder
        self.dec4 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec3 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch, base_ch)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.dec1(d2)
        return (out + 1.0) / 2.0
