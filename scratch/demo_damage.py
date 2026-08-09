"""
demo_damage.py
--------------
Demonstration script to run the local damage simulator.
It creates a beautiful dummy geometric painting, applies the 5 degradation modes,
and saves the clean/damaged/mask triplets into a 'demo_outputs' folder.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Add workspace root to python path to run local package imports
workspace_root = "/Users/rajnishsingh/Downloads/files (3)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from utils.degrade import apply_damage


def main():
    print("=== Indian Folk Art Damage Simulator Demo ===")

    # Create a nice colorful dummy "painting" (representing a folk art design)
    w, h = 256, 256
    img = Image.new("RGB", (w, h), color=(245, 240, 220))  # Warm canvas background
    draw = ImageDraw.Draw(img)

    # Draw some bold geometric shapes inspired by traditional folk motifs
    draw.rectangle([20, 20, 236, 236], outline=(120, 30, 30), width=4)
    # Sun disk
    draw.ellipse([78, 78, 178, 178], fill=(235, 120, 40), outline=(20, 20, 20), width=2)
    # Triangle motifs
    draw.polygon(
        [(128, 30), (80, 110), (176, 110)],
        fill=(70, 130, 180),
        outline=(20, 20, 20),
        width=2,
    )
    draw.polygon(
        [(128, 226), (80, 146), (176, 146)],
        fill=(60, 140, 90),
        outline=(20, 20, 20),
        width=2,
    )

    output_dir = Path(workspace_root) / "demo_outputs"
    output_dir.mkdir(exist_ok=True)

    # Save original clean image
    clean_path = output_dir / "0_clean_original.png"
    img.save(clean_path)
    print(f"Saved clean original image -> {clean_path}\n")

    # Apply and save each damage type
    damage_modes = ["fade", "scratch", "tear", "stain", "combined"]
    for mode in damage_modes:
        damaged, mask = apply_damage(img, mode=mode)

        # Save damaged image
        dmg_path = output_dir / f"1_damaged_{mode}.png"
        damaged.save(dmg_path)

        # Save binary mask
        mask_path = output_dir / f"2_mask_{mode}.png"
        Image.fromarray(mask).save(mask_path)

        print(f"  ✓ Applied '{mode:<8}' damage -> Saved input & mask files.")

    print(f"\nDemo successfully completed! Check the output files in: {output_dir}")


if __name__ == "__main__":
    main()
