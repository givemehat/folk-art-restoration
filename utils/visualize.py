# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
utils/visualize.py
------------------
Plotting helpers for the folk-art restoration pipeline:
  - plot_results()         : 3-panel before / after comparison figure
  - plot_training_curves() : train & val loss curves from CSV log
"""

import os
import csv
from pathlib import Path

import numpy as np
import sys
# Use non-interactive Agg backend only if we are running headlessly (not in a notebook/IPython environment)
if "ipykernel" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from PIL import Image

from utils.metrics import compute_psnr, compute_ssim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_display(img) -> np.ndarray:
    """
    Convert any image representation to an HxWx3 uint8 numpy array
    suitable for imshow().
    Accepts: torch.Tensor (C,H,W or 1,C,H,W), PIL.Image, or np.ndarray.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        img = img.permute(1, 2, 0).numpy()

    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))

    img = np.array(img, dtype=np.float32)
    if img.max() <= 1.0 + 1e-6:
        img = (img * 255.0).clip(0, 255)

    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# plot_results
# ---------------------------------------------------------------------------

def plot_results(
    original,
    damaged,
    restored,
    save_path: str,
    fig_title: str = "Folk Art Restoration",
) -> None:
    """
    Create a 3-panel side-by-side comparison figure and save it as a
    high-resolution PNG (300 DPI) suitable for paper inclusion.

    Panel 1 – Original (clean ground-truth)
    Panel 2 – Damaged  (model input)
    Panel 3 – Restored (model output) with PSNR & SSIM overlay

    Parameters
    ----------
    original  : ground-truth image (tensor, PIL, or numpy).
    damaged   : degraded model input.
    restored  : model output.
    save_path : full path to save the output PNG.
    fig_title : optional super-title for the figure.
    """
    orig_np = _to_display(original)
    dmg_np  = _to_display(damaged)
    rst_np  = _to_display(restored)

    # Compute quality scores (restored vs original)
    psnr = compute_psnr(orig_np, rst_np)
    ssim = compute_ssim(orig_np, rst_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    panels = [
        (orig_np, "Original (Clean)"),
        (dmg_np,  "Damaged (Input)"),
        (rst_np,  f"Restored (Output)\nPSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure → {save_path}")


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    csv_path: str,
    save_path: str = None,
    title: str = "Training Curves",
) -> None:
    """
    Read a CSV log file and plot train & validation loss over epochs.

    Expected CSV columns (case-insensitive):
      epoch, train_loss, val_loss  (plus optional: val_psnr, val_ssim)

    Parameters
    ----------
    csv_path  : path to the CSV log produced during training.
    save_path : where to save the figure PNG.  If None, saves alongside the CSV.
    title     : figure title.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    # --- Parse CSV --------------------------------------------------------
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Normalise column names to lowercase
        for row in reader:
            rows.append({k.lower().strip(): v for k, v in row.items()})

    if not rows:
        raise ValueError("CSV file is empty.")

    def _col(key, default=None):
        """Extract a numeric column if it exists."""
        if key in rows[0]:
            return [float(r[key]) for r in rows if r.get(key, "").strip() != ""]
        return default

    epochs = _col("epoch") or list(range(1, len(rows) + 1))
    train_loss = _col("train_loss")
    val_loss   = _col("val_loss")
    val_psnr   = _col("val_psnr")
    val_ssim   = _col("val_ssim")

    # Decide subplot layout
    has_extra = bool(val_psnr or val_ssim)
    n_plots = 1 + int(bool(val_psnr)) + int(bool(val_ssim))
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Loss subplot -----------------------------------------------------
    ax = axes[0]
    if train_loss:
        ax.plot(epochs, train_loss, label="Train Loss", color="steelblue", linewidth=1.8)
    if val_loss:
        ax.plot(epochs, val_loss,   label="Val Loss",   color="tomato",    linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- PSNR subplot (optional) ------------------------------------------
    plot_idx = 1
    if val_psnr and plot_idx < len(axes):
        ax2 = axes[plot_idx]
        ax2.plot(epochs[:len(val_psnr)], val_psnr, color="mediumseagreen", linewidth=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title("Validation PSNR")
        ax2.grid(alpha=0.3)
        plot_idx += 1

    # --- SSIM subplot (optional) ------------------------------------------
    if val_ssim and plot_idx < len(axes):
        ax3 = axes[plot_idx]
        ax3.plot(epochs[:len(val_ssim)], val_ssim, color="darkorange", linewidth=1.8)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("SSIM")
        ax3.set_title("Validation SSIM")
        ax3.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = csv_path.with_suffix(".png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves → {save_path}")


# ---------------------------------------------------------------------------
# Quick grid of N restoration examples
# ---------------------------------------------------------------------------

def plot_grid(
    originals: list,
    damaged_list: list,
    restored_list: list,
    save_path: str,
    n: int = 5,
) -> None:
    """
    Plot *n* rows of (original | damaged | restored) comparisons in one figure.
    Useful for a quick qualitative overview at the end of a Colab session.

    Parameters
    ----------
    originals      : list of clean images
    damaged_list   : list of degraded images (same order)
    restored_list  : list of restored images (same order)
    save_path      : path to save PNG
    n              : number of rows (samples) to show
    """
    n = min(n, len(originals))
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Original", "Damaged", "Restored"]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        orig = _to_display(originals[i])
        dmg  = _to_display(damaged_list[i])
        rst  = _to_display(restored_list[i])
        psnr = compute_psnr(orig, rst)
        ssim = compute_ssim(orig, rst)

        for j, img in enumerate([orig, dmg, rst]):
            axes[i][j].imshow(img)
            axes[i][j].axis("off")

        axes[i][2].set_xlabel(f"PSNR {psnr:.1f} dB | SSIM {ssim:.3f}", fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid → {save_path}")
