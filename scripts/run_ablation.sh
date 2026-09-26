#!/bin/bash

# Folk Art Restoration Ablation Study Runner
# Executes training across different modes to study individual component contributions.

CONFIG="configs/baseline_edsr_lama.yaml"
DATA_ROOT="data/sample"
SAVE_DIR="results/ablations"

echo "========================================="
echo "Starting Ablation Study for Paper Results"
echo "========================================="

echo "[1/3] Training SR Only (EDSR Baseline)..."
python train.py --config $CONFIG --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode sr

echo "[2/3] Training Inpaint Only (LaMa Baseline)..."
python train.py --config $CONFIG --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode inpaint

echo "[3/3] Training Joint Pipeline (EDSR + LaMa)..."
python train.py --config $CONFIG --data_root $DATA_ROOT --save_dir $SAVE_DIR --mode joint

echo "========================================="
echo "Ablation Study Complete."
echo "Models saved in $SAVE_DIR"
echo "========================================="
