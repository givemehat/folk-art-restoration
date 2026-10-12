#!/bin/bash
set -e

# ==============================================================================
# Empirical Ablation Study Execution Pipeline
# Generates model checkpoints across all comparative variants for the paper.
# ==============================================================================

DATA_ROOT="data/full/processed"
SAVE_DIR="checkpoints/ablation"

echo "=================================================================="
echo " Folk Art Restoration: Scientific Ablation Study Execution"
echo "=================================================================="

echo "[1/4] Variant A: Super-Resolution Only (EDSR Baseline)..."
python train.py --config configs/proposed_dual_stage.yaml --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode sr --epochs 10

echo "[2/4] Variant B: Inpainting Only (Spectral LaMa Baseline)..."
python train.py --config configs/proposed_dual_stage.yaml --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode inpaint --epochs 10

echo "[3/4] Variant C: Vanilla Spatial U-Net (Ablation: No Spectral Blocks)..."
python train.py --config configs/ablation_spatial_unet.yaml --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode inpaint --epochs 10

echo "[4/4] Proposed: Dual-Stage Framework (Joint EDSR + Spectral LaMa)..."
python train.py --config configs/proposed_dual_stage.yaml --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode joint --epochs 10

echo "=================================================================="
echo " Ablation Study Finished Successfully."
echo " Checkpoints preserved in $SAVE_DIR"
echo "=================================================================="
