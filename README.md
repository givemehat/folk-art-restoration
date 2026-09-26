# Indian Folk Art Restoration AI

<div align="center">
  <img src="https://img.shields.io/github/repo-size/givemehat/folk-art-restoration?style=for-the-badge&color=blue" alt="Repository Size" />
  <img src="https://img.shields.io/github/license/givemehat/folk-art-restoration?style=for-the-badge&color=green" alt="License" />
</div>

## Overview
A deep learning project focused on restoring Indian folk art images, such as Madhubani, Warli, and Pattachitra. It utilizes a combination of **EDSR (Enhanced Deep Super-Resolution)** and **LaMa (Large Mask Inpainting)** models to address degradation, scratching, and other damage specific to historical art pieces.

The pipeline performs a two-stage restoration:
1. **EDSR Upsampling:** Enhances the resolution and fine-grained details of the damaged artwork.
2. **LaMa Inpainting:** Generates visually and structurally consistent fills for damaged or missing regions.

## Features
- **Full Neural Pipeline**: Integrated Super Resolution + Mask Inpainting.
- **Flask Web Dashboard**: Includes a fully interactive UI to upload art, simulate damage, and perform live neural restoration.
- **Metric Evaluation**: Built-in support for computing PSNR, SSIM, and LPIPS metrics.
- **Paper Visualizations**: Scripts in the `scratch/` directory easily reproduce charts and visual diagrams for research publication.

## Setup & Installation

Ensure you have Python 3.9+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/givemehat/folk-art-restoration.git
   cd folk-art-restoration
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Checkpoints:**
   Place the pre-trained checkpoints in the `checkpoints/` directory as follows:
   - `checkpoints/edsr/edsr_best.pth`
   - `checkpoints/lama/lama_best.pth`
   *(Note: The app will run without them but will use untrained weights)*

## Running the Web App
Start the Flask backend:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

## Evaluation & Training

- **Evaluate Test Set:**
  ```bash
  python evaluate.py --data_root ./data --output_dir ./results
  ```
- **Train EDSR Model:**
  ```bash
  python train_edsr.py --data_root ./data --epochs 50
  ```
- **Train LaMa Model:**
  ```bash
  python train_lama.py --data_root ./data --epochs 50
  ```

## Scratch Scripts
The `scratch/` directory contains helper scripts for preparing research paper results, generating diagrams, and running one-epoch demo trainings.

---
**Lead Researcher:** Rajnish Singh  
**Institution:** Computer Science & Engineering  
