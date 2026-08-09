"""
prepare_paper_results.py
------------------------
Automates generating high-resolution, publication-grade figures for the
three vivid sample folk art paintings (peacock, village life, Krishna).
Generates both 3-panel side-by-side figures and 4-panel residual error map
heatmap figures for each painting at 300 DPI.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add workspace root to python path to run local package imports
workspace_root = "/Users/rajnishsingh/Downloads/files (3)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from models.edsr import EDSR
from models.lama import Generator
from utils.degrade import apply_damage
from utils.visualize import plot_results
from app import plot_residual_map  # Reuse the new residual map logic


def main():
    print("=== RESEARCH PAPER RESULTS GENERATOR ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Initialize models and load 1-epoch checkpoints
    print("\nLoading models...")
    edsr = EDSR(scale=2).to(device).eval()
    lama = Generator(use_attention=True).to(device).eval()

    edsr_ckpt = Path(workspace_root) / "checkpoints" / "edsr" / "edsr_best.pth"
    lama_ckpt = Path(workspace_root) / "checkpoints" / "lama" / "lama_best.pth"

    if edsr_ckpt.exists():
        ckpt = torch.load(edsr_ckpt, map_location=device)
        edsr.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("  ✓ Loaded EDSR weights.")
    else:
        print("ERROR: EDSR weights not found. Run dry-run first.")
        return

    if lama_ckpt.exists():
        ckpt = torch.load(lama_ckpt, map_location=device)
        lama.load_state_dict(ckpt.get("gen_state_dict", ckpt))
        print("  ✓ Loaded LaMa weights.")
    else:
        print("ERROR: LaMa weights not found. Run dry-run first.")
        return

    # 2. Define our sample targets
    samples = [
        {
            "name": "madhubani_peacock",
            "damage_mode": "scratch",
            "title": "Madhubani Peacocks Restoration Analysis",
        },
        {
            "name": "warli_village_life",
            "damage_mode": "tear",
            "title": "Warli Tribal Village Restoration Analysis",
        },
        {
            "name": "pattachitra_krishna",
            "damage_mode": "stain",
            "title": "Pattachitra Scroll Painting Restoration Analysis",
        },
    ]

    samples_dir = Path(workspace_root) / "sample_inputs"
    visuals_dir = Path(workspace_root) / "paper_visuals"
    visuals_dir.mkdir(exist_ok=True)

    import torchvision.transforms.functional as TF
    import torchvision.transforms as T

    # 3. Process each sample
    for idx, sample in enumerate(samples):
        name = sample["name"]
        dmg_mode = sample["damage_mode"]
        title = sample["title"]

        img_path = samples_dir / f"{name}.png"
        if not img_path.exists():
            print(f"WARNING: Sample image not found: {img_path}")
            continue

        print(f"\nProcessing [{idx+1}/3] {name} with '{dmg_mode}' damage...")

        # Load clean and resize to 256x256
        clean_pil = (
            Image.open(img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
        )

        # Apply simulated damage
        damaged_pil, mask_np = apply_damage(clean_pil, mode=dmg_mode)

        clean_t = TF.to_tensor(clean_pil)
        damaged_t = TF.to_tensor(damaged_pil)
        mask_t = TF.to_tensor(Image.fromarray(mask_np).convert("L"))

        # Execute Restoration
        with torch.no_grad():
            lr = (
                T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC)(
                    damaged_t
                )
                .unsqueeze(0)
                .to(device)
            )
            sr = edsr(lr).squeeze(0).cpu()

            lama_inp = torch.cat([sr, mask_t], dim=0).unsqueeze(0).to(device)
            restored_t = lama(lama_inp).squeeze(0).cpu().clamp(0.0, 1.0)

        restored_pil = TF.to_pil_image(restored_t)

        # Generate 3-Panel Figure
        fig_3panel_path = visuals_dir / f"{name}_3panel_comparison.png"
        plot_results(
            original=clean_pil,
            damaged=damaged_pil,
            restored=restored_pil,
            save_path=str(fig_3panel_path),
            fig_title=f"{title} (EDSR + LaMa GAN)",
        )
        print(f"  ✓ Saved 3-Panel Figure -> {fig_3panel_path.name}")

        # Generate 4-Panel Error Map Figure
        fig_error_path = visuals_dir / f"{name}_error_heatmap.png"
        plot_residual_map(
            original=clean_pil,
            restored=restored_pil,
            damaged=damaged_pil,
            save_path=str(fig_error_path),
        )
        print(f"  ✓ Saved 4-Panel Error Map -> {fig_error_path.name}")

    print("\n=== ALL RESEARCH FIGURES SUCCESSFULLY PREPARED ===")
    print(f"View and download them at: {visuals_dir}")


if __name__ == "__main__":
    main()
