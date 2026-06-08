# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
restore.py
----------
Single-image inference script for the folk-art restoration pipeline.

Usage:
  # Restore only:
  python restore.py --input ./damaged.png --output ./restored.png

  # Restore and score against ground-truth:
  python restore.py --input ./damaged.png \
                    --output ./restored.png \
                    --ground_truth ./original.png

  # Override default checkpoint paths:
  python restore.py --input ./damaged.png \
                    --edsr_ckpt ./checkpoints/edsr/edsr_best.pth \
                    --lama_ckpt ./checkpoints/lama/lama_best.pth \
                    --output ./restored.png

Optional flags:
  --scale INT      EDSR scale factor (default 2)
  --no_edsr        Skip EDSR; run LaMa inpainting only
  --mask PATH      Supply your own binary mask PNG (255 = damaged region)
                   If omitted, the script auto-generates a mask by detecting
                   bright white/grey damaged regions in the input image.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from models.edsr import EDSR
from models.lama import Generator
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load an image from disk and convert to (3, size, size) float [0,1] tensor."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return TF.to_tensor(img)            # (3, H, W)  [0, 1]


def load_mask(path: str, size: int = 256) -> torch.Tensor:
    """Load a grayscale mask PNG → (1, H, W) binary float tensor."""
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    t = TF.to_tensor(m)                 # (1, H, W)  [0, 1]
    return (t > 0.5).float()


def auto_mask(damaged_tensor: torch.Tensor, threshold: float = 0.90) -> torch.Tensor:
    """
    Heuristic mask generation: flag pixels where all RGB channels are above
    *threshold* (bright white/grey noise from tear/scratch damage).
    Also flags very low-saturation regions typical of fade damage.

    Parameters
    ----------
    damaged_tensor : (3, H, W) float [0,1]
    threshold      : brightness threshold for near-white detection.

    Returns
    -------
    (1, H, W) binary float tensor — 1 = likely damaged.
    """
    # Near-white mask
    bright = (damaged_tensor.min(dim=0, keepdim=True).values > threshold).float()

    # Near-grey (low saturation): max - min channel < 0.05 AND brightness > 0.85
    brightness = damaged_tensor.mean(dim=0, keepdim=True)
    saturation = damaged_tensor.max(dim=0, keepdim=True).values - \
                 damaged_tensor.min(dim=0, keepdim=True).values
    grey_damaged = ((saturation < 0.05) & (brightness > 0.85)).float()

    mask = ((bright + grey_damaged) > 0).float()

    # Dilate slightly with a 5×5 max-pool to cover damage borders
    import torch.nn.functional as F
    mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    return mask


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (3, H, W) float [0,1] tensor as PNG."""
    img = TF.to_pil_image(tensor.clamp(0, 1).cpu())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"Saved restored image → {path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Folk Art Restoration — single-image inference")
    p.add_argument("--input",        type=str, required=True,  help="Path to damaged input image")
    p.add_argument("--output",       type=str, required=True,  help="Path for restored output PNG")
    p.add_argument("--ground_truth", type=str, default=None,   help="Optional: clean reference for metrics")
    p.add_argument("--mask",         type=str, default=None,   help="Optional: binary mask PNG (255=damaged)")
    p.add_argument("--edsr_ckpt",    type=str, default="./checkpoints/edsr/edsr_best.pth")
    p.add_argument("--lama_ckpt",    type=str, default="./checkpoints/lama/lama_best.pth")
    p.add_argument("--scale",        type=int, default=2, choices=[2, 4])
    p.add_argument("--no_edsr",      action="store_true", help="Skip EDSR; use LaMa only")
    p.add_argument("--size",         type=int, default=256, help="Spatial resolution to work at")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ Load image & mask
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading input image: {args.input}")
    damaged = load_image(args.input, size=args.size)   # (3, H, W)

    if args.mask:
        print(f"Loading supplied mask: {args.mask}")
        mask = load_mask(args.mask, size=args.size)
    else:
        print("No mask supplied — auto-generating damage mask …")
        mask = auto_mask(damaged)
        n_damaged = mask.sum().item()
        pct = 100 * n_damaged / (args.size * args.size)
        print(f"  Auto-mask: {n_damaged:.0f} pixels flagged ({pct:.1f}% of image)")

    # ------------------------------------------------------------------ Load LaMa
    print("\nLoading LaMa generator …")
    gen = Generator().to(device).eval()
    lama_ckpt_path = Path(args.lama_ckpt)
    if not lama_ckpt_path.exists():
        print(f"ERROR: LaMa checkpoint not found: {lama_ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(lama_ckpt_path, map_location=device)
    gen.load_state_dict(ckpt.get("gen_state_dict", ckpt))

    # ------------------------------------------------------------------ Optional: EDSR
    if not args.no_edsr:
        print("Loading EDSR …")
        edsr_ckpt_path = Path(args.edsr_ckpt)
        if not edsr_ckpt_path.exists():
            print(f"WARNING: EDSR checkpoint not found ({edsr_ckpt_path}). Falling back to LaMa-only.")
            args.no_edsr = True
        else:
            edsr = EDSR(scale=args.scale).to(device).eval()
            ckpt_edsr = torch.load(edsr_ckpt_path, map_location=device)
            edsr.load_state_dict(ckpt_edsr.get("model_state_dict", ckpt_edsr))

    # ------------------------------------------------------------------ Inference
    print("\nRunning restoration pipeline …")
    with torch.no_grad():
        if not args.no_edsr:
            # Stage 1: Downscale → EDSR upsample
            resize_tf = T.Resize(
                (args.size // args.scale, args.size // args.scale),
                interpolation=T.InterpolationMode.BICUBIC,
            )
            lr = resize_tf(damaged).unsqueeze(0).to(device)  # (1, 3, H/s, W/s)
            sr = edsr(lr).squeeze(0).cpu()                   # (3, H, W)
            stage1_label = f"EDSR (×{args.scale})"
        else:
            sr = damaged
            stage1_label = "No EDSR"

        # Stage 2: LaMa inpainting
        lama_inp = torch.cat([sr, mask], dim=0).unsqueeze(0).to(device)  # (1,4,H,W)
        restored = gen(lama_inp).squeeze(0).cpu().clamp(0, 1)           # (3, H, W)

    print(f"Pipeline: {stage1_label} → LaMa Inpainting")

    # ------------------------------------------------------------------ Save output
    save_image(restored, args.output)

    # ------------------------------------------------------------------ Metrics (optional)
    if args.ground_truth:
        if not Path(args.ground_truth).exists():
            print(f"WARNING: Ground-truth file not found: {args.ground_truth}")
        else:
            print(f"\nComputing metrics vs ground truth: {args.ground_truth}")
            gt = load_image(args.ground_truth, size=args.size)

            psnr = compute_psnr(gt, restored)
            ssim = compute_ssim(gt, restored)
            try:
                lpips_val = compute_lpips(gt, restored, device=device)
                lpips_str = f"{lpips_val:.4f}"
            except ImportError:
                lpips_str = "N/A (install lpips)"

            print(f"\n  {'Metric':<10} | {'Score':>10}")
            print(f"  {'-'*24}")
            print(f"  {'PSNR ↑':<10} | {psnr:>9.2f} dB")
            print(f"  {'SSIM ↑':<10} | {ssim:>10.4f}")
            print(f"  {'LPIPS ↓':<10} | {lpips_str:>10}")


if __name__ == "__main__":
    main()
