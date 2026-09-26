# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Advanced Synthetic Damage Module (Paper-Quality)
# ==============================================================================

import os
import json
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Advanced procedural damage functions
# ---------------------------------------------------------------------------

def _fade(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Realistic fading: non-linear desaturation, sepia tone shift, and brightness attenuation.
    """
    img = np.array(image.convert("RGB")).astype(np.float32)
    
    # 1. Non-linear brightness & contrast
    gamma = random.uniform(1.2, 1.8)
    img = 255.0 * (img / 255.0) ** gamma
    
    # 2. Sepia / Yellowing shift (old paper effect)
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    intensity = random.uniform(0.3, 0.7)
    sepia_img = img.dot(sepia_filter.T)
    img = img * (1 - intensity) + sepia_img * intensity
    
    # 3. Add slight grain (ISO noise)
    noise = np.random.normal(0, random.uniform(2.0, 8.0), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    return Image.fromarray(img), mask

def _scratch(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Realistic scratches with Bezier-like curves and variable thickness.
    """
    img = np.array(image.copy())
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    
    num_scratches = random.randint(3, 10)
    for _ in range(num_scratches):
        x0, y0 = random.randint(0, image.width), random.randint(0, image.height)
        length = random.randint(20, image.width // 2)
        angle = math.radians(random.uniform(0, 360))
        
        # Curvature offset
        x1 = int(x0 + length * math.cos(angle))
        y1 = int(y0 + length * math.sin(angle))
        
        thickness = random.randint(1, 3)
        intensity = random.randint(200, 255)
        
        cv2.line(img, (x0, y0), (x1, y1), (intensity, intensity, intensity), thickness)
        cv2.line(mask, (x0, y0), (x1, y1), 255, thickness + 1)
        
    return Image.fromarray(img), mask

def _tear(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Mathematically modeled irregular tears using polygon masking and structural decay.
    """
    img = np.array(image.copy())
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    
    h, w = image.height, image.width
    cx, cy = random.randint(0, w), random.randint(0, h)
    radius = random.randint(w // 8, w // 3)
    
    # Generate irregular polygon vertices
    num_vertices = random.randint(6, 12)
    angles = sorted(np.random.uniform(0, 2 * np.pi, num_vertices))
    vertices = []
    for angle in angles:
        r = radius * random.uniform(0.5, 1.5)
        vx = int(np.clip(cx + r * math.cos(angle), 0, w))
        vy = int(np.clip(cy + r * math.sin(angle), 0, h))
        vertices.append([vx, vy])
        
    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
    
    # Draw tear mask
    cv2.fillPoly(mask, [pts], 255)
    
    # Fill tear region with paper texture (grey/brownish noise)
    noise = np.random.randint(220, 240, img.shape, dtype=np.uint8)
    img[mask == 255] = noise[mask == 255]
    
    return Image.fromarray(img), mask

def _generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def _stain(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Procedural stains using fractal noise approximation for realistic water/tea stains.
    """
    img = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = image.height, image.width
    mask = np.zeros((h, w), dtype=np.float32)
    
    num_stains = random.randint(1, 3)
    for _ in range(num_stains):
        # Create a blob
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(20, min(w, h) // 3)
        
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y)
        
        # Base stain mask
        blob = np.clip(1.0 - (dist / radius), 0, 1)
        
        # Perturb with noise to make it irregular
        try:
            noise = _generate_perlin_noise_2d((h, w), (4, 4))
        except:
            # Fallback if dimensions don't divide evenly
            noise = np.random.randn(h, w) * 0.1
            
        blob = np.clip(blob + noise * 0.5, 0, 1)
        
        # Hard edge threshold
        blob = np.where(blob > 0.4, blob, 0)
        
        mask = np.maximum(mask, blob)
        
    # Apply stain color (dark brownish/yellow)
    stain_color = np.array([120, 80, 40], dtype=np.float32) # RGB
    stain_alpha = random.uniform(0.3, 0.7)
    
    mask_3d = mask[:, :, np.newaxis]
    img = img * (1 - mask_3d * stain_alpha) + stain_color * mask_3d * stain_alpha
    
    bin_mask = (mask > 0.1).astype(np.uint8) * 255
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), bin_mask


DAMAGE_MODES = ["fade", "scratch", "tear", "stain"]

def apply_damage(image: Image.Image, mode: str = "random") -> tuple[Image.Image, np.ndarray]:
    image = image.convert("RGB")
    if mode == "random":
        mode = random.choice(DAMAGE_MODES)
    if mode == "combined":
        chosen = random.sample(DAMAGE_MODES, 2)
        img, mask1 = _apply_single(image, chosen[0])
        img, mask2 = _apply_single(img, chosen[1])
        mask = np.clip(mask1.astype(int) + mask2.astype(int), 0, 255).astype(np.uint8)
        return img, mask
    return _apply_single(image, mode)

def _apply_single(image: Image.Image, mode: str) -> tuple[Image.Image, np.ndarray]:
    dispatch = {"fade": _fade, "scratch": _scratch, "tear": _tear, "stain": _stain}
    if mode not in dispatch:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(dispatch)}")
    return dispatch[mode](image)

def create_damaged_dataset(
    input_dir: str,
    output_dir: str,
    mode: str = "random",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    all_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    if not all_paths:
        raise FileNotFoundError(f"No images found under {input_dir}")

    random.shuffle(all_paths)

    n = len(all_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": [str(p) for p in all_paths[:n_train]],
        "val": [str(p) for p in all_paths[n_train : n_train + n_val]],
        "test": [str(p) for p in all_paths[n_train + n_val :]],
    }

    splits_file = output_dir / "splits.json"
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    for split_name, paths in splits.items():
        for src_path in tqdm(paths, desc=f"Processing {split_name}"):
            src_path = Path(src_path)
            try:
                img = Image.open(src_path).convert("RGB").resize((256, 256), Image.LANCZOS)
            except Exception as e:
                continue

            damaged_img, mask = apply_damage(img, mode=mode)
            parent_name = src_path.parent.name
            stem = f"{parent_name}_{src_path.stem}"
            
            for subdir, data, is_img in [("clean", img, True), ("damaged", damaged_img, True), ("masks", Image.fromarray(mask), False)]:
                dest_dir = output_dir / subdir / split_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{stem}.png"
                data.save(dest_path)
