# ==============================================================================
# Indian Folk Art Restoration AI Pipeline
# ----------------------------------------
# Author / Lead Researcher: Rajnish Singh
# Institution: Computer Science & Engineering
# Environment: PyTorch / Mac & Linux
# Description: Custom implementation for Madhubani, Warli, and Pattachitra Restoration
# ==============================================================================

"""
app.py
------
Flask backend for the Indian Folk Art Restoration AI dashboard.
Provides local endpoints to upload images, apply simulated damage,
run live neural restoration (EDSR + LaMa) using our trained checkpoints,
calculate active image metrics (PSNR, SSIM), and download research figures.
"""

import os
import sys
import uuid
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add workspace root to python path to run local package imports
workspace_root = "/Users/rajnishsingh/Downloads/files (3)"
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from models.edsr import EDSR
from models.lama import Generator
from utils.degrade import apply_damage
from utils.metrics import compute_psnr, compute_ssim
from utils.visualize import plot_results

app = Flask(__name__)

def plot_residual_map(original, restored, damaged, save_path):
    """Generates a 4-panel publication figure including a glowing absolute error map heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    orig_np = np.array(original.convert("RGB"), dtype=np.float32) / 255.0
    rest_np = np.array(restored.convert("RGB"), dtype=np.float32) / 255.0
    
    # Compute absolute differences and average over channels
    diff = np.abs(orig_np - rest_np)
    error_map = np.mean(diff, axis=2)  # (H, W)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle("Neural Restoration Efficiency & Residual Error Map Analysis", fontsize=13, fontweight="bold", y=1.01)
    
    axes[0].imshow(original)
    axes[0].set_title("Original (Clean GT)", fontsize=9, fontweight="semibold")
    
    axes[1].imshow(damaged)
    axes[1].set_title("Damaged (Input)", fontsize=9, fontweight="semibold")
    
    axes[2].imshow(restored)
    axes[2].set_title("Restored (Output)", fontsize=9, fontweight="semibold")
    
    im = axes[3].imshow(error_map, cmap="jet", vmin=0.0, vmax=0.3)
    axes[3].set_title("Absolute Error Map (|GT - Out|)", fontsize=9, fontweight="semibold")
    
    # Add beautiful custom colorbar
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Reconstruction Error Scale", fontsize=7, fontweight="semibold")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
app.config['UPLOAD_FOLDER'] = os.path.join(workspace_root, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
(Path(workspace_root) / 'static' / 'css').mkdir(parents=True, exist_ok=True)
(Path(workspace_root) / 'static' / 'js').mkdir(parents=True, exist_ok=True)

# Cache models on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask Web Server initializing models on: {device}")

edsr_model = EDSR(scale=2).to(device).eval()
lama_model = Generator(use_attention=True).to(device).eval()

# Try loading the saved checkpoints (even the 1-epoch ones)
edsr_ckpt = Path(workspace_root) / "checkpoints" / "edsr" / "edsr_best.pth"
lama_ckpt = Path(workspace_root) / "checkpoints" / "lama" / "lama_best.pth"

if edsr_ckpt.exists():
    ckpt = torch.load(edsr_ckpt, map_location=device)
    edsr_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"Loaded EDSR weights from {edsr_ckpt}")

if lama_ckpt.exists():
    ckpt = torch.load(lama_ckpt, map_location=device)
    lama_model.load_state_dict(ckpt.get("gen_state_dict", ckpt))
    print(f"Loaded LaMa weights from {lama_ckpt}")


def auto_mask(damaged_tensor: torch.Tensor, threshold: float = 0.90) -> torch.Tensor:
    """Heuristic mask generator from restore.py"""
    bright = (damaged_tensor.min(dim=0, keepdim=True).values > threshold).float()
    brightness = damaged_tensor.mean(dim=0, keepdim=True)
    saturation = damaged_tensor.max(dim=0, keepdim=True).values - \
                 damaged_tensor.min(dim=0, keepdim=True).values
    grey_damaged = ((saturation < 0.05) & (brightness > 0.85)).float()
    mask = ((bright + grey_damaged) > 0).float()
    
    import torch.nn.functional as F
    mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    return mask


@app.route('/')
def index():
    # Gather any demo outputs to show on startup
    demo_dir = Path(workspace_root) / "demo_outputs"
    demos = []
    if demo_dir.exists():
        for f in sorted(demo_dir.glob("1_damaged_*.png")):
            demos.append(f.stem.replace("1_damaged_", ""))
    return render_template('index.html', demos=demos)


@app.route('/restore', methods=['POST'])
def restore():
    try:
        # Determine source image (file upload or demo mode)
        img_file = request.files.get('image')
        demo_mode = request.form.get('demo_mode')
        damage_type = request.form.get('damage_type', 'scratch')
        
        uid = str(uuid.uuid4())
        
        if img_file and img_file.filename != '':
            # User uploaded an image
            filename = f"{uid}_uploaded.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)
            
            clean_pil = Image.open(filepath).convert("RGB").resize((256, 256), Image.LANCZOS)
            # Save resized clean
            clean_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_clean.png")
            clean_pil.save(clean_path)
            
            # Apply simulated damage based on selected type
            damaged_pil, mask_np = apply_damage(clean_pil, mode=damage_type)
            damaged_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_damaged.png")
            damaged_pil.save(damaged_path)
            
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_mask.png")
            Image.fromarray(mask_np).save(mask_path)
            
        elif demo_mode:
            # Load from pre-generated demo_outputs
            demo_dir = Path(workspace_root) / "demo_outputs"
            clean_path_orig = demo_dir / "0_clean_original.png"
            damaged_path_orig = demo_dir / f"1_damaged_{demo_mode}.png"
            mask_path_orig = demo_dir / f"2_mask_{demo_mode}.png"
            
            if not clean_path_orig.exists():
                return jsonify({"error": "Demo outputs not generated. Run dry-run first."}), 400
                
            clean_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_clean.png")
            damaged_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_damaged.png")
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_mask.png")
            
            # Copy to upload folder
            Image.open(clean_path_orig).save(clean_path)
            Image.open(damaged_path_orig).save(damaged_path)
            Image.open(mask_path_orig).save(mask_path)
            
            clean_pil = Image.open(clean_path)
            damaged_pil = Image.open(damaged_path)
        else:
            return jsonify({"error": "No image source provided"}), 400

        # Load tensors for model forward pass
        clean_t = TF.to_tensor(clean_pil)
        damaged_t = TF.to_tensor(damaged_pil)
        
        # Load or generate mask
        if demo_mode:
            mask_t = TF.to_tensor(Image.open(mask_path).convert("L"))
        else:
            # Auto-mask uploaded inputs
            mask_t = auto_mask(damaged_t)
            # Update mask file on disk
            TF.to_pil_image(mask_t).save(mask_path)

        # --------------------------------------------------------- Neural Inference
        with torch.no_grad():
            # Stage 1: Downscale -> EDSR super-resolution upscale
            lr = T.Resize((128, 128), interpolation=T.InterpolationMode.BICUBIC)(damaged_t).unsqueeze(0).to(device)
            sr = edsr_model(lr).squeeze(0).cpu()  # (3, 256, 256)
            
            # Stage 2: LaMa inpainting on the SR output using the mask
            lama_inp = torch.cat([sr, mask_t.cpu()], dim=0).unsqueeze(0).to(device)
            restored_t = lama_model(lama_inp).squeeze(0).cpu().clamp(0.0, 1.0)
            
        # Save restored image
        restored_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_restored.png")
        restored_pil = TF.to_pil_image(restored_t)
        restored_pil.save(restored_path)
        
        # --------------------------------------------------------- Compute Metrics
        psnr_val = compute_psnr(clean_pil, restored_pil)
        ssim_val = compute_ssim(clean_pil, restored_pil)
        
        # --------------------------------------------------------- Generate Paper Figure (300 DPI)
        paper_fig_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_paper_figure.png")
        plot_results(
            original=clean_pil,
            damaged=damaged_pil,
            restored=restored_pil,
            save_path=paper_fig_path,
            fig_title=f"Indian Folk Art Restoration — Style Consistency and SR Pipeline",
        )
        
        # Generate the Residual Error Map figure (4-panel with colorbar)
        residual_fig_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_residual_figure.png")
        plot_residual_map(
            original=clean_pil,
            restored=restored_pil,
            damaged=damaged_pil,
            save_path=residual_fig_path
        )
        
        # Return web paths relative to static
        return jsonify({
            "clean_url": f"/static/uploads/{uid}_clean.png",
            "damaged_url": f"/static/uploads/{uid}_damaged.png",
            "mask_url": f"/static/uploads/{uid}_mask.png",
            "restored_url": f"/static/uploads/{uid}_restored.png",
            "figure_url": f"/static/uploads/{uid}_paper_figure.png",
            "residual_url": f"/static/uploads/{uid}_residual_figure.png",
            "psnr": f"{psnr_val:.2f}",
            "ssim": f"{ssim_val:.4f}",
            "restored_filename": f"{uid}_restored.png",
            "figure_filename": f"{uid}_paper_figure.png",
            "residual_filename": f"{uid}_residual_figure.png"
        })
        
    except Exception as e:
        print(f"Error during restoration: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
