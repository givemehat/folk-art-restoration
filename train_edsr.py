# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
train_edsr.py
-------------
Training script for the EDSR super-resolution model.

Usage (Colab / terminal):
  python train_edsr.py --data_root ./data \
                       --ckpt_dir  ./checkpoints/edsr \
                       --scale 2 --epochs 50 --batch_size 16

Trains EDSR to upsample damaged low-res images back to 256×256.
Saves best checkpoint (lowest val loss) and logs training stats to CSV.
"""

import os
import csv
import math
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.edsr import EDSR, FolkArtSRDataset
from models.losses import ReconstructionLoss
from utils.metrics import compute_psnr, compute_ssim

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description="Train EDSR for folk-art super-resolution")
    p.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Dataset root (output of create_damaged_dataset)",
    )
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/edsr",
        help="Where to save checkpoints",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[2, 4],
        help="Super-resolution scale factor",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_channels", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=16)
    p.add_argument(
        "--patience", type=int, default=7, help="Early stopping patience (epochs)"
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    """One full training epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(lr)
            loss, _ = criterion(pred, hr)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float, float]:
    """
    Validate the model.
    Returns (mean_val_loss, mean_psnr, mean_ssim).
    """
    model.eval()
    total_loss, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred = model(lr)
        loss, _ = criterion(pred, hr)
        total_loss += loss.item()

        # Per-image quality metrics
        for i in range(pred.size(0)):
            psnr_sum += compute_psnr(hr[i], pred[i])
            ssim_sum += compute_ssim(hr[i], pred[i])
            n += 1

    return total_loss / len(loader), psnr_sum / n, ssim_sum / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Checkpoint dir & CSV log
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "train_log.csv"

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = FolkArtSRDataset(args.data_root, split="train", scale=args.scale)
    val_ds = FolkArtSRDataset(args.data_root, split="val", scale=args.scale)
    print(f"Train: {len(train_ds)} images   Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = EDSR(
        scale=args.scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EDSR parameters: {total_params:,}")

    criterion = ReconstructionLoss(l1_weight=1.0, perceptual_weight=0.1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision (only on CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # CSV log initialisation
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr"]
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = math.inf
    patience_counter = 0
    best_ckpt_path = ckpt_dir / "edsr_best.pth"

    print("\n" + "=" * 65)
    print(
        f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'PSNR':>7} | {'SSIM':>7}"
    )
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
            f"{val_psnr:>7.2f} | {val_ssim:>7.4f}  [{elapsed:.1f}s]"
        )

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val_loss, val_psnr, val_ssim, current_lr]
            )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  ✓ New best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience})."
                )
                break

        # Also save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                ckpt_dir / f"edsr_epoch{epoch:03d}.pth",
            )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Training log:         {csv_path}")


if __name__ == "__main__":
    main()
