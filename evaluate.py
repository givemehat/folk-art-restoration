"""
evaluate.py
-----------
Runs the full test-set evaluation and prints a comparison table:

  Model                  | PSNR ↑  | SSIM ↑  | LPIPS ↓
  -----------------------|---------|---------|--------
  OpenCV Inpainting      |  xx.xx  |  0.xxx  |  0.xxx
  LaMa only              |  xx.xx  |  0.xxx  |  0.xxx
  EDSR + LaMa (ours)     |  xx.xx  |  0.xxx  |  0.xxx

Usage:
  python evaluate.py \
    --data_root   ./data \
    --edsr_ckpt   ./checkpoints/edsr/edsr_best.pth \
    --lama_ckpt   ./checkpoints/lama/lama_best.pth \
    --output_dir  ./data/restored \
    --scale 2

Baselines compared:
  1. OpenCV TELEA inpainting (classical)
  2. LaMa generator only (no EDSR upsampling)
  3. EDSR upsampling → LaMa inpainting (full pipeline)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from models.edsr import EDSR
from models.lama import Generator, FolkArtInpaintDataset
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
from utils.visualize import plot_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) float [0,1] tensor → (H,W,C) uint8 numpy."""
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H,W,C) uint8 → (C,H,W) float [0,1] tensor."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)


def opencv_inpaint(damaged_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """
    Run OpenCV TELEA inpainting.
    damaged_t : (3, H, W) float tensor [0,1]
    mask_t    : (1, H, W) float tensor {0, 1}
    Returns   : (3, H, W) float tensor [0,1]
    """
    # Convert to uint8 numpy
    img_np = tensor_to_numpy_uint8(damaged_t)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # OpenCV expects mask as uint8 (255 = inpaint here)
    mask_np = (mask_t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    # TELEA inpainting, radius = 3 pixels
    result_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return numpy_to_tensor(result_rgb)


# ---------------------------------------------------------------------------
# Model pipeline wrappers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_lama(gen: Generator, damaged: torch.Tensor, mask: torch.Tensor, device) -> torch.Tensor:
    """Run LaMa generator on a single (3,H,W) + (1,H,W) pair."""
    inp = torch.cat([damaged, mask], dim=0).unsqueeze(0).to(device)  # (1,4,H,W)
    out = gen(inp).squeeze(0).cpu()                                   # (3,H,W)
    return out.clamp(0, 1)


@torch.no_grad()
def run_edsr_lama(
    edsr: EDSR,
    gen: Generator,
    damaged: torch.Tensor,
    mask: torch.Tensor,
    scale: int,
    device,
) -> torch.Tensor:
    """
    Full two-stage pipeline:
      1. Upsample damaged LR image with EDSR
      2. Feed EDSR output + mask to LaMa generator
    """
    # Stage 1: LR → HR with EDSR
    lr = T.Resize(
        (damaged.shape[1] // scale, damaged.shape[2] // scale),
        interpolation=T.InterpolationMode.BICUBIC,
    )(damaged).unsqueeze(0).to(device)
    sr = edsr(lr).squeeze(0).cpu()                # (3, H, W)

    # Stage 2: Inpainting
    return run_lama(gen, sr, mask, device)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate(psnrs, ssims, lpipss) -> dict:
    a, b, c = np.array(psnrs), np.array(ssims), np.array(lpipss)
    return {
        "psnr":  (a.mean(), a.std()),
        "ssim":  (b.mean(), b.std()),
        "lpips": (c.mean(), c.std()),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="./data")
    p.add_argument("--edsr_ckpt",  type=str, default="./checkpoints/edsr/edsr_best.pth")
    p.add_argument("--lama_ckpt",  type=str, default="./checkpoints/lama/lama_best.pth")
    p.add_argument("--output_dir", type=str, default="./data/restored")
    p.add_argument("--scale",      type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1,   help="Keep at 1 for easy visualisation")
    p.add_argument("--num_workers",type=int, default=2)
    p.add_argument("--n_vis",      type=int, default=5,   help="Number of results to visualise")
    p.add_argument("--no_lpips",   action="store_true",   help="Skip slow LPIPS computation")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")

    # ------------------------------------------------------------------ Load models
    print("Loading EDSR …")
    edsr = EDSR(scale=args.scale).to(device).eval()
    ckpt = torch.load(args.edsr_ckpt, map_location=device)
    edsr.load_state_dict(ckpt.get("model_state_dict", ckpt))

    print("Loading LaMa generator …")
    gen = Generator().to(device).eval()
    ckpt_lama = torch.load(args.lama_ckpt, map_location=device)
    gen.load_state_dict(ckpt_lama.get("gen_state_dict", ckpt_lama))

    # ------------------------------------------------------------------ Test DataLoader
    test_ds = FolkArtInpaintDataset(args.data_root, split="test", augment=False)
    print(f"Test set: {len(test_ds)} images\n")
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # ------------------------------------------------------------------ Containers
    results = {
        "opencv": {"psnr": [], "ssim": [], "lpips": []},
        "lama":   {"psnr": [], "ssim": [], "lpips": []},
        "full":   {"psnr": [], "ssim": [], "lpips": []},
    }

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    use_lpips = not args.no_lpips

    # ------------------------------------------------------------------ Evaluation loop
    for idx, batch in enumerate(test_loader):
        clean   = batch["clean"][0]     # (3,H,W)
        damaged = batch["damaged"][0]   # (3,H,W)
        mask    = batch["mask"][0]      # (1,H,W)
        fname   = batch["filename"][0]

        # 1. OpenCV TELEA
        cv_out = opencv_inpaint(damaged, mask)

        # 2. LaMa only
        lama_out = run_lama(gen, damaged, mask, device)

        # 3. EDSR + LaMa
        full_out = run_edsr_lama(edsr, gen, damaged, mask, args.scale, device)

        # Collect metrics
        for tag, restored in [("opencv", cv_out), ("lama", lama_out), ("full", full_out)]:
            results[tag]["psnr"].append(compute_psnr(clean, restored))
            results[tag]["ssim"].append(compute_ssim(clean, restored))
            if use_lpips:
                results[tag]["lpips"].append(compute_lpips(clean, restored, device=device))

        # Visualise first n_vis samples
        if vis_count < args.n_vis:
            plot_results(
                clean, damaged, full_out,
                save_path=str(vis_dir / f"sample_{idx:04d}_{fname}"),
                fig_title=f"Sample {idx}: {fname}",
            )
            vis_count += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_ds)} images …")

    # ------------------------------------------------------------------ Summary table
    def _fmt(tag):
        p_m, p_s = np.mean(results[tag]["psnr"]),  np.std(results[tag]["psnr"])
        s_m, s_s = np.mean(results[tag]["ssim"]),  np.std(results[tag]["ssim"])
        if use_lpips and results[tag]["lpips"]:
            l_m, l_s = np.mean(results[tag]["lpips"]), np.std(results[tag]["lpips"])
            return p_m, p_s, s_m, s_s, l_m, l_s
        return p_m, p_s, s_m, s_s, float("nan"), float("nan")

    rows = [
        ("OpenCV Inpainting",  _fmt("opencv")),
        ("LaMa only",          _fmt("lama")),
        ("EDSR + LaMa (ours)", _fmt("full")),
    ]

    sep = "-" * 65
    print("\n" + sep)
    print(f"{'Model':<24} | {'PSNR ↑':>9} | {'SSIM ↑':>9} | {'LPIPS ↓':>9}")
    print(sep)
    for name, (pm, ps, sm, ss, lm, ls) in rows:
        lpips_str = f"{lm:.3f}±{ls:.3f}" if not np.isnan(lm) else "  N/A  "
        print(f"{name:<24} | {pm:>5.2f}±{ps:.2f} | {sm:>5.3f}±{ss:.3f} | {lpips_str:>9}")
    print(sep + "\n")

    # Save results JSON
    results_json = {}
    for tag, (name, vals) in zip(["opencv", "lama", "full"], rows):
        pm, ps, sm, ss, lm, ls = vals
        results_json[name] = {
            "psnr_mean": pm, "psnr_std": ps,
            "ssim_mean": sm, "ssim_std": ss,
            "lpips_mean": None if np.isnan(lm) else lm,
        }
    json_out = output_dir / "eval_results.json"
    with open(json_out, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved → {json_out}")
    print(f"Visualisations → {vis_dir}")


if __name__ == "__main__":
    main()
