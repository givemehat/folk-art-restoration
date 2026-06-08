# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
utils/degrade.py
----------------
Simulates real-world damage on Indian folk art images to create
paired (clean, damaged) training data for the restoration pipeline.

Damage types:
  fade      - brightness/saturation reduction
  scratch   - random diagonal white lines
  tear      - rectangular region filled with noise
  stain     - circular brownish blotches
  combined  - 2 random damage types applied together
"""

import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Individual damage functions
# ---------------------------------------------------------------------------

def _fade(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Reduce brightness (40-60%) and desaturate colours (30%).
    Returns (damaged_image, mask) where mask is all-zeros (no localised region).
    """
    img = image.copy()

    # Reduce brightness
    brightness_factor = random.uniform(0.40, 0.60)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Desaturate colours
    color_factor = 1.0 - 0.30  # keep 70% saturation
    img = ImageEnhance.Color(img).enhance(color_factor)

    # Fade is global – mask is all zeros (nothing "missing")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    return img, mask


def _scratch(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Draw 3-8 random white diagonal lines of varying thickness over the image.
    Mask marks the pixels covered by the lines.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask_img = Image.fromarray(mask)
    mask_draw = ImageDraw.Draw(mask_img)

    num_scratches = random.randint(3, 8)
    w, h = image.size

    for _ in range(num_scratches):
        # Random start point anywhere on image edges or interior
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        # End point offset diagonally
        length = random.randint(50, max(w, h) // 2)
        angle = random.uniform(-60, 60)  # degrees from horizontal
        x1 = int(x0 + length * math.cos(math.radians(angle)))
        y1 = int(y0 + length * math.sin(math.radians(angle)))
        thickness = random.randint(1, 5)

        # White scratch on image
        draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 255), width=thickness)
        # Mark mask
        mask_draw.line([(x0, y0), (x1, y1)], fill=255, width=thickness + 2)

    mask = np.array(mask_img)
    return img, mask


def _tear(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Mask a random rectangular region (10-25% of image area) with white/grey noise.
    """
    img = image.copy()
    img_array = np.array(img)
    w, h = image.size
    total_area = w * h

    # Choose a rectangle covering 10-25% of the image
    target_area = random.uniform(0.10, 0.25) * total_area
    aspect = random.uniform(0.5, 2.0)  # rect aspect ratio
    rect_w = int(math.sqrt(target_area * aspect))
    rect_h = int(target_area / rect_w)

    # Clamp dimensions
    rect_w = min(rect_w, w - 1)
    rect_h = min(rect_h, h - 1)

    # Random top-left corner
    x0 = random.randint(0, w - rect_w)
    y0 = random.randint(0, h - rect_h)
    x1, y1 = x0 + rect_w, y0 + rect_h

    # Fill with white/grey noise
    noise = np.random.randint(200, 256, (rect_h, rect_w, 3), dtype=np.uint8)
    img_array[y0:y1, x0:x1] = noise

    # Build binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255

    return Image.fromarray(img_array), mask


def _stain(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Add 2-6 circular brownish/yellowish blotches at random positions.
    Blotches are semi-transparent to simulate old ink/water stains.
    """
    img = image.copy().convert("RGBA")
    w, h = image.size
    mask = np.zeros((h, w), dtype=np.uint8)
    mask_img = Image.fromarray(mask)
    mask_draw = ImageDraw.Draw(mask_img)

    num_stains = random.randint(2, 6)
    for _ in range(num_stains):
        # Brownish-yellow stain colour with variable opacity
        r = random.randint(100, 160)
        g = random.randint(70, 120)
        b = random.randint(20, 60)
        alpha = random.randint(80, 180)

        # Random centre and radius
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        radius = random.randint(15, min(w, h) // 5)

        # Draw stain as an ellipse on overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=(r, g, b, alpha),
        )
        # Blur to make it look organic
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius // 3))
        img = Image.alpha_composite(img, overlay)

        # Mask covers the stain ellipse area
        mask_draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=255,
        )

    mask = np.array(mask_img)
    return img.convert("RGB"), mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DAMAGE_MODES = ["fade", "scratch", "tear", "stain"]


def apply_damage(
    image: Image.Image, mode: str = "random"
) -> tuple[Image.Image, np.ndarray]:
    """
    Apply damage to a PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        Clean source image (RGB, 256x256 recommended).
    mode : str
        One of 'fade', 'scratch', 'tear', 'stain', 'combined', or 'random'.
        'random'   → picks one mode at random.
        'combined' → applies 2 randomly chosen modes in sequence.

    Returns
    -------
    damaged : PIL.Image.Image
        The degraded image.
    mask : np.ndarray  shape (H, W) uint8
        Binary mask (255 = damaged region, 0 = clean region).
        For global effects like fade the mask is all zeros.
    """
    image = image.convert("RGB")

    if mode == "random":
        mode = random.choice(DAMAGE_MODES)

    if mode == "combined":
        # Pick 2 distinct modes and apply sequentially
        chosen = random.sample(DAMAGE_MODES, 2)
        img, mask1 = _apply_single(image, chosen[0])
        img, mask2 = _apply_single(img, chosen[1])
        # Union of both masks
        mask = np.clip(mask1.astype(int) + mask2.astype(int), 0, 255).astype(np.uint8)
        return img, mask

    return _apply_single(image, mode)


def _apply_single(
    image: Image.Image, mode: str
) -> tuple[Image.Image, np.ndarray]:
    """Dispatch to the correct damage function."""
    dispatch = {
        "fade": _fade,
        "scratch": _scratch,
        "tear": _tear,
        "stain": _stain,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown damage mode '{mode}'. Choose from {list(dispatch)}")
    return dispatch[mode](image)


def create_damaged_dataset(
    input_dir: str,
    output_dir: str,
    mode: str = "random",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Walk *input_dir* for PNG/JPG images, apply damage to each, and save:
      - <output_dir>/clean/<split>/<filename>.png   – original
      - <output_dir>/damaged/<split>/<filename>.png – degraded
      - <output_dir>/masks/<split>/<filename>.png   – binary mask
      - <output_dir>/splits.json                    – train/val/test split info

    Parameters
    ----------
    input_dir  : path to folder containing clean images (any sub-folder depth).
    output_dir : root of the output dataset.
    mode       : damage mode passed to apply_damage().
    train_ratio / val_ratio : fractions; test = 1 - train - val.
    seed       : random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all image paths
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    all_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    if not all_paths:
        raise FileNotFoundError(f"No images found under {input_dir}")

    random.shuffle(all_paths)

    # Compute split indices
    n = len(all_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": [str(p) for p in all_paths[:n_train]],
        "val": [str(p) for p in all_paths[n_train : n_train + n_val]],
        "test": [str(p) for p in all_paths[n_train + n_val :]],
    }

    # Save split info
    splits_file = output_dir / "splits.json"
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved → {splits_file}")

    # Process each split
    for split_name, paths in splits.items():
        for src_path in tqdm(paths, desc=f"Processing {split_name}"):
            src_path = Path(src_path)
            try:
                img = Image.open(src_path).convert("RGB").resize((256, 256), Image.LANCZOS)
            except Exception as e:
                print(f"  [SKIP] {src_path}: {e}")
                continue

            damaged_img, mask = apply_damage(img, mode=mode)

            parent_name = src_path.parent.name
            stem = f"{parent_name}_{src_path.stem}"
            for subdir, data, is_img in [
                ("clean", img, True),
                ("damaged", damaged_img, True),
                ("masks", Image.fromarray(mask), False),
            ]:
                dest_dir = output_dir / subdir / split_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{stem}.png"
                data.save(dest_path)

    print(f"\nDataset created at {output_dir}")
    print(f"  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")
