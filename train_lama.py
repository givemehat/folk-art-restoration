# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
train_lama.py
-------------
Training script for the LaMa-style inpainting GAN.

Usage:
  python train_lama.py --data_root ./data \
                       --ckpt_dir  ./checkpoints/lama \
                       --epochs 100 --batch_size 8

Trains Generator (U-Net + FFC) and PatchGAN Discriminator with:
  - Generator loss  = 10 × L1 + 0.1 × perceptual + 1.0 × adversarial (hinge)
  - Discriminator   = hinge loss
  - Separate Adam optimisers for G and D (lr = 1e-4)
  - Logs every step's losses to train_log.csv
  - Saves best checkpoint by lowest val reconstruction loss
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

from models.lama import Generator, PatchDiscriminator, FolkArtInpaintDataset
from models.losses import ReconstructionLoss, AdversarialLoss, PerceptualLoss

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(description="Train LaMa inpainting GAN")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints/lama")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate")
    p.add_argument(
        "--lr_d", type=float, default=1e-4, help="Discriminator learning rate"
    )
    p.add_argument("--lambda_l1", type=float, default=10.0)
    p.add_argument("--lambda_perc", type=float, default=0.1)
    p.add_argument(
        "--lambda_style", type=float, default=0.0, help="Style consistency loss weight"
    )
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_ffc", type=int, default=4, help="FFC blocks in bottleneck")
    p.add_argument(
        "--use_attention",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use Attention Gates in U-Net skip connections",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    gen,
    disc,
    loader,
    opt_g,
    opt_d,
    recon_loss_fn,
    adv_loss_fn,
    device,
    scaler,
    lambda_adv: float,
) -> dict:
    """
    Run one full training epoch.
    Returns dict with mean losses: g_total, g_l1, g_perc, g_adv, d_loss.
    """
    gen.train()
    disc.train()

    meters = {
        k: 0.0 for k in ["g_total", "g_l1", "g_perc", "g_style", "g_adv", "d_loss"]
    }
    n = 0

    for batch in loader:
        clean = batch["clean"].to(device, non_blocking=True)  # (B,3,H,W) [0,1]
        damaged = batch["damaged"].to(device, non_blocking=True)  # (B,3,H,W)
        mask = batch["mask"].to(device, non_blocking=True)  # (B,1,H,W) {0,1}

        # Input to generator: damaged + mask → 4 channels
        gen_input = torch.cat([damaged, mask], dim=1)  # (B,4,H,W)

        # ============================================================
        # 1. Update Discriminator
        # ============================================================
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            fake = gen(gen_input).detach()  # no grad through G here

            real_logits = disc(clean, mask)
            fake_logits = disc(fake, mask)
            d_loss = adv_loss_fn(real_logits, fake_logits, mode="discriminator")

        opt_d.zero_grad()
        if scaler:
            scaler.scale(d_loss).backward()
            scaler.unscale_(opt_d)
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler.step(opt_d)
        else:
            d_loss.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        # ============================================================
        # 2. Update Generator
        # ============================================================
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            fake = gen(gen_input)  # fresh forward pass with grad

            # Reconstruction loss (L1 + perceptual)
            g_recon, breakdown = recon_loss_fn(fake, clean)

            # Adversarial loss for G
            fake_logits_for_g = disc(fake, mask)
            g_adv = adv_loss_fn(fake_logits=fake_logits_for_g, mode="generator")

            g_total = g_recon + lambda_adv * g_adv

        opt_g.zero_grad()
        if scaler:
            scaler.scale(g_total).backward()
            scaler.unscale_(opt_g)
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler.step(opt_g)
            # We share the same scaler; update it once per iteration after all optimizer steps
            scaler.update()
        else:
            g_total.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_g.step()

        # Accumulate losses
        bs = clean.size(0)
        meters["g_total"] += g_total.item() * bs
        meters["g_l1"] += breakdown["l1"] * bs
        meters["g_perc"] += breakdown["perceptual"] * bs
        meters["g_style"] += breakdown.get("style", 0.0) * bs
        meters["g_adv"] += g_adv.item() * bs
        meters["d_loss"] += d_loss.item() * bs
        n += bs

    return {k: v / n for k, v in meters.items()}


# ---------------------------------------------------------------------------
# Validation (reconstruction loss only — no GAN component)
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(gen, loader, recon_loss_fn, device) -> float:
    gen.eval()
    total_loss, n = 0.0, 0

    for batch in loader:
        clean = batch["clean"].to(device)
        damaged = batch["damaged"].to(device)
        mask = batch["mask"].to(device)

        gen_input = torch.cat([damaged, mask], dim=1)
        fake = gen(gen_input)
        loss, _ = recon_loss_fn(fake, clean)
        total_loss += loss.item() * clean.size(0)
        n += clean.size(0)

    return total_loss / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "train_log.csv"

    # ------------------------------------------------------------------ Datasets
    train_ds = FolkArtInpaintDataset(args.data_root, split="train", augment=True)
    val_ds = FolkArtInpaintDataset(args.data_root, split="val", augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

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

    # ------------------------------------------------------------------ Models
    use_attn = args.use_attention == "True"
    gen = Generator(base_ch=args.base_ch, n_ffc=args.n_ffc, use_attention=use_attn).to(
        device
    )
    disc = PatchDiscriminator(base_ch=args.base_ch).to(device)

    g_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator params: {g_params:,}   Discriminator params: {d_params:,}")

    # ------------------------------------------------------------------ Losses
    recon_loss_fn = ReconstructionLoss(
        l1_weight=args.lambda_l1,
        perceptual_weight=args.lambda_perc,
        style_weight=args.lambda_style,
    ).to(device)
    adv_loss_fn = AdversarialLoss()

    # ------------------------------------------------------------------ Optimisers & Schedulers
    opt_g = Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    sched_g = CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-6)

    # Shared AMP scaler (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ------------------------------------------------------------------ CSV log
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "g_total",
                "g_l1",
                "g_perc",
                "g_style",
                "g_adv",
                "d_loss",
                "val_loss",
            ]
        )

    # ------------------------------------------------------------------ Training loop
    best_val_loss = math.inf
    best_ckpt = ckpt_dir / "lama_best.pth"

    header = f"{'Epoch':>6} | {'G-Total':>8} | {'G-L1':>8} | {'G-Adv':>7} | {'D-Loss':>7} | {'Val':>8}"
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            gen,
            disc,
            train_loader,
            opt_g,
            opt_d,
            recon_loss_fn,
            adv_loss_fn,
            device,
            scaler,
            lambda_adv=args.lambda_adv,
        )
        val_loss = validate(gen, val_loader, recon_loss_fn, device)

        sched_g.step()
        sched_d.step()

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {train_metrics['g_total']:>8.4f} | {train_metrics['g_l1']:>8.4f} | "
            f"{train_metrics['g_adv']:>7.4f} | {train_metrics['d_loss']:>7.4f} | "
            f"{val_loss:>8.4f}  [{elapsed:.1f}s]"
        )

        # CSV log
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_metrics["g_total"],
                    train_metrics["g_l1"],
                    train_metrics["g_perc"],
                    train_metrics["g_style"],
                    train_metrics["g_adv"],
                    train_metrics["d_loss"],
                    val_loss,
                ]
            )

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "gen_state_dict": gen.state_dict(),
                    "disc_state_dict": disc.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"  ✓ Best checkpoint saved (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "gen_state_dict": gen.state_dict()},
                ckpt_dir / f"lama_epoch{epoch:03d}.pth",
            )

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints:   {ckpt_dir}")
    print(f"Log:           {csv_path}")


if __name__ == "__main__":
    main()
