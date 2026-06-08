"""
run_one_epoch_training.py
-------------------------
End-to-end live dry-run of the entire folk art restoration pipeline.
1. Creates a tiny raw dataset of 4 images.
2. Creates damaged dataset splits (train/val/test).
3. Trains EDSR super-resolution network for 1 epoch.
4. Trains LaMa inpainting GAN for 1 epoch.
5. Runs restore.py using the newly saved checkpoints.
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

# Add workspace root to python path to run local package imports
workspace_root = "/Users/rajnishsingh/Downloads/files (3)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

def main():
    print("=== STARTING FULL END-TO-END DRY-RUN ===\n")
    
    # 1. Create a tiny raw dataset of 4 images
    raw_dir = Path(workspace_root) / "data" / "raw" / "Madhubani"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Generating tiny raw dataset...")
    for i in range(4):
        img = Image.new("RGB", (256, 256), color=(240 - i*15, 230 + i*5, 200 + i*10))
        draw = ImageDraw.Draw(img)
        # Draw some folk art geometric patterns
        draw.ellipse([60, 60, 196, 196], fill=(210, 90 + i*25, 45), outline=(10, 10, 10), width=2)
        draw.rectangle([30, 30, 226, 226], outline=(100, 20, 20), width=2)
        img.save(raw_dir / f"dummy_art_{i:04d}.png")
    print(f"  ✓ Created 4 raw images in {raw_dir}")
    
    # 2. Run dataset damage generator
    print("\n[2/5] Creating damaged train/val/test splits ...")
    from utils.degrade import create_damaged_dataset
    
    create_damaged_dataset(
        input_dir=str(Path(workspace_root) / "data" / "raw"),
        output_dir=str(Path(workspace_root) / "data"),
        mode="random",
        train_ratio=0.50,
        val_ratio=0.25,
        seed=42
    )
    
    # 3. Train EDSR for 1 epoch
    print("\n[3/5] Dry-running train_edsr.py for 1 epoch ...")
    result_edsr = subprocess.run(
        [
            ".venv/bin/python3", "train_edsr.py",
            "--data_root", "./data",
            "--ckpt_dir", "./checkpoints/edsr",
            "--scale", "2",
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0"
        ],
        capture_output=True, text=True
    )
    print(result_edsr.stdout)
    if result_edsr.returncode != 0:
        print("ERROR: train_edsr.py failed:")
        print(result_edsr.stderr)
        return
        
    # 4. Train LaMa GAN for 1 epoch
    print("\n[4/5] Dry-running train_lama.py for 1 epoch ...")
    result_lama = subprocess.run(
        [
            ".venv/bin/python3", "train_lama.py",
            "--data_root", "./data",
            "--ckpt_dir", "./checkpoints/lama",
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0",
            "--use_attention", "True",
            "--lambda_style", "0.0"  # Style loss off to speed up CPU dry-run
        ],
        capture_output=True, text=True
    )
    print(result_lama.stdout)
    if result_lama.returncode != 0:
        print("ERROR: train_lama.py failed:")
        print(result_lama.stderr)
        return
        
    # 5. Run single-image inference using these newly saved weights
    print("\n[5/5] Executing restore.py with the 1-epoch checkpoints ...")
    result_restore = subprocess.run(
        [
            ".venv/bin/python3", "restore.py",
            "--input", "demo_outputs/1_damaged_scratch.png",
            "--output", "demo_outputs/restored_1epoch.png",
            "--ground_truth", "demo_outputs/0_clean_original.png",
            "--edsr_ckpt", "./checkpoints/edsr/edsr_best.pth",
            "--lama_ckpt", "./checkpoints/lama/lama_best.pth",
            "--scale", "2"
        ],
        capture_output=True, text=True
    )
    print(result_restore.stdout)
    if result_restore.returncode != 0:
        print("ERROR: restore.py failed:")
        print(result_restore.stderr)
        return
        
    print("\n=== PIPELINE DRY-RUN SUCCESSFULLY COMPLETED ===")

if __name__ == "__main__":
    main()
