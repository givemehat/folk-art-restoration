# Physics-Informed Dual-Stage Restoration of Heritage Folk Art: Coupling Spectral Inpainting with Detail-Preserving Super-Resolution

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target Venue](https://img.shields.io/badge/Target%20Journal-IEEE%20TIP%20%7C%20ACM%20JOCCH-purple.svg)]()

**Rajnish Singh**  
*Department of Computer Science & Engineering*

</div>

---

## 📜 Abstract & Theoretical Foundation

Traditional Indian folk paintings (e.g., **Madhubani**, **Warli**, **Pattachitra**, **Gond**, **Kalighat**) embody delicate hand-drawn linework, periodic geometric motifs, and organic mineral pigment washes. When subjected to centuries of mechanical stress, atmospheric humidity, and chemical oxidation, these artworks suffer from compound, heterogeneous degradations: **structural tears**, **abrasive scratches**, **aqueous fungal stains**, and **non-linear pigment fading**.

Standard convolutional inpainting architectures fail on this cultural heritage domain because localized spatial receptive fields cannot capture long-range periodic border motifs, resulting in blurry linework and color shifts. Conversely, super-resolution networks trained independently hallucinate high-frequency noise when applied directly to missing regions.

To address these limitations, we propose a **Physics-Informed Dual-Stage Restoration Framework**:
1. **Stage 1 (Global Structural Inpainting via Fast Fourier Convolutions):** Formulates inpainting in the 2D Fourier domain using spectral residual blocks and Attention Gates, providing an image-wide receptive field to seamlessly synthesize periodic motifs.
2. **Stage 2 (High-Frequency Detail Super-Resolution):** Employs an Enhanced Deep Residual Network (EDSR) stripped of batch-normalization layers to prevent color distortion while restoring micro-textures and stroke crispness.
3. **Frequency-Spatial Multi-Objective Supervision:** Combines multi-scale $\mathcal{L}_1$ pixel loss, 2D Fourier spectral discrepancy $\mathcal{L}_{\text{freq}}$, deep perceptual loss $\mathcal{L}_{\text{perc}}$, and PatchGAN adversarial hinge loss $\mathcal{L}_{\text{adv}}$.

```
[Degraded Image (3ch) + Binary Mask (1ch)]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Spectral Inpainting Generator                 │
│  - Dilated Spatial Encoder                             │
│  - Fast Fourier Convolution (FFC) Bottleneck (Global) │
│  - Attention-Gated Skip Connections                    │
└────────────────────────────────────────────────────────┘
                   │
         [Structurally Inpainted Art]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Detail-Preserving Super-Resolution (EDSR)     │
│  - Residual Blocks (without Batch Normalization)       │
│  - Sub-Pixel PixelShuffle Upsampling                   │
└────────────────────────────────────────────────────────┘
                   │
                   ▼
       [High-Fidelity Restored Folk Artwork]
```

---

## 📂 Repository Organization

```
folk-art-restoration/
├── configs/                       # Experiment & Architecture Configurations
│   ├── proposed_dual_stage.yaml   # Primary Dual-Stage Multi-Objective Config
│   ├── ablation_spatial_unet.yaml # Vanilla Spatial U-Net Baseline Config
│   └── ablation_no_frequency.yaml # Ablation Config without Spectral Loss
├── data/                          # Dataset Management
│   ├── full/processed/            # Paired clean, damaged, and mask splits
│   └── sample/                    # Lightweight testing samples
├── data_utils/                    # Ingestion & Acquisition Utilities
│   ├── download_dataset.py        # Wikimedia Commons & Museum Open API Scraper
│   └── download_kaggle.py         # Kaggle Indian Paintings Ingestion Script
├── models/                        # Core Neural Architecture Suite
│   ├── registry.py                # Unified Factory Pattern Model Loader
│   ├── lama.py                    # Fast Fourier Convolution + Attention Generator
│   ├── edsr.py                    # Enhanced Deep Residual Super-Resolution
│   ├── baseline_unet.py           # Vanilla Spatial U-Net (Ablation Baseline)
│   ├── losses.py                  # Multi-Objective Losses (L1, Spectral, VGG, GAN)
│   └── diffusion.py               # Diffusers Inpainting Baseline Adapter
├── results/                       # Empirical Evaluation Artifacts
│   ├── evaluation/                # Test metrics JSON & LaTeX tables
│   └── figures/                   # 300 DPI Camera-Ready Publication Figures
├── scripts/                       # Reproducibility & Benchmark Automation
│   ├── run_ablation.sh            # Automated Ablation Study Execution
│   └── generate_paper_figures.py  # 300 DPI Publication Visuals Generator
├── utils/                         # Research Utilities
│   ├── degrade.py                 # Mathematical Procedural Degradation Engine
│   ├── metrics.py                 # PSNR, SSIM, LPIPS, FID & Sharpness Metrics
│   └── profiler.py                # FLOPs, Parameter Counts & Latency Profiling
├── train.py                       # Unified Multi-Objective Trainer
├── evaluate.py                    # Quantitative Evaluation & LaTeX Table Generator
├── app.py                         # Interactive Flask Demonstration Web Panel
├── RESULTS.md                     # Comprehensive Scientific Results & Tables
├── requirements.txt               # Pinned Python Dependencies
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Environment Preparation
```bash
git clone https://github.com/givemehat/folk-art-restoration.git
cd folk-art-restoration
pip install -r requirements.txt
```

### 2. Dataset Preparation
The benchmark incorporates two curated datasets:
- **Dataset 1 (Indian Paintings Archive):** 665+ traditional paintings across 8 styles (Madhubani, Warli, Gond, Kalighat, Kangra, Kerala Mural, Mandana, Pichwai).
- **Dataset 2 (Real Paired Historical Restoration):** 93 authentic museum before/after restoration pairs.

To generate the procedurally damaged triplets (Clean, Damaged, Mask) with mathematically modeled tears, scratches, fading, and stains:
```bash
python -c 'from utils.degrade import create_damaged_dataset; create_damaged_dataset("data/full/archive", "data/full/processed", mode="random")'
```

---

## 🔬 Training & Ablation Execution

### Train Proposed Dual-Stage Pipeline
```bash
python train.py \
  --config configs/proposed_dual_stage.yaml \
  --data_root data/full/processed \
  --mode joint \
  --epochs 25 \
  --batch_size 8
```
*Supports hardware acceleration on NVIDIA GPUs (`cuda`), Apple Silicon (`mps`), and `cpu`.*

### Execute Full Ablation Suite
Run the automated ablation script comparing (1) SR Only, (2) Inpainting Only, (3) Vanilla Spatial U-Net, and (4) Proposed Dual-Stage:
```bash
bash scripts/run_ablation.sh
```

---

## 📊 Quantitative Benchmarking & LaTeX Export

Run the peer-review-grade evaluation suite on the isolated test set:
```bash
python evaluate.py \
  --data_root data/full/processed \
  --edsr_ckpt checkpoints/ablation/joint/best_sr.pth \
  --lama_ckpt checkpoints/ablation/joint/best_inpaint.pth \
  --output_dir results/evaluation \
  --max_test_samples 50
```

This generates:
1. Terminal summary table with mean $\pm$ standard deviation across PSNR, SSIM, LPIPS, FID, and Laplacian Sharpness.
2. `results/evaluation/table_results.tex` containing camera-ready `\begin{table*}` LaTeX markup for paper inclusion.
3. Side-by-side visual comparisons in `results/evaluation/visualisations/`.

---

## 🎨 Interactive Demonstration Dashboard

To explore live neural restoration on user-uploaded cultural artworks:
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 📑 Citation & Academic Attribution

If you use this codebase, architectures, or benchmark in your research, please cite:

```bibtex
@article{singh2026folkartrestoration,
  title={Physics-Informed Dual-Stage Restoration of Heritage Folk Art: Coupling Spectral Inpainting with Detail-Preserving Super-Resolution},
  author={Singh, Rajnish},
  journal={IEEE Transactions on Image Processing (Under Review)},
  year={2026}
}
```
