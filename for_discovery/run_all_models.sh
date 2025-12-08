#!/bin/bash
# Run all three PTN model variants (Simplified for overfitting check)
# Usage: bash run_all_models.sh
#
# CHANGES:
# - Disabled early stopping (force 300 epochs)
# - Disabled advanced regularization (mixup, smoothing)
# - Reduced dropout
# - Goal: Force models to overfit training data (Acc -> 100%) to verify capacity

set -e

echo "=========================================="
echo "ASL Pose Transformer Training (Sanity Check)"
echo "=========================================="
echo "Goal: Force overfitting on small dataset"
echo "  - Epochs: 300"
echo "  - Early Stopping: OFF"
echo "  - Regularization: Minimal"
echo "=========================================="

# Create models directory if it doesn't exist
mkdir -p models

# ============================================
# Model 1: MAE (Self-Supervised Pretraining)
# ============================================
echo ""
echo "[1/3] Training MAE (Self-Supervised) Model..."
echo "=============================================="
python3 src/train_ptn_pretrained.py \
    --data_root keypoints_data \
    --epochs 300 \
    --batch_size 16 \
    --lr 3e-4 \
    --model_type mae \
    --pretrain_epochs 50 \
    --save_path models/best_ptn_mae.pth

# ============================================
# Model 2: BERT-tiny (Pretrained, lightweight)
# ============================================
echo ""
echo "[2/3] Training BERT-tiny Model..."
echo "=============================================="
python3 src/train_ptn_pretrained.py \
    --data_root keypoints_data \
    --epochs 300 \
    --batch_size 16 \
    --lr 3e-4 \
    --model_type bert-tiny \
    --save_path models/best_ptn_bert.pth

# ============================================
# Model 3: GPT-2 (Pretrained, larger)
# ============================================
echo ""
echo "[3/3] Training GPT-2 Model (frozen backbone)..."
echo "=============================================="
python3 src/train_ptn_pretrained.py \
    --data_root keypoints_data \
    --epochs 300 \
    --batch_size 16 \
    --lr 3e-4 \
    --model_type gpt2 \
    --freeze_backbone \
    --save_path models/best_ptn_gpt2.pth

echo ""
echo "=========================================="
echo "All training complete!"
echo "Models saved in ./models/"
echo "  - best_ptn_mae.pth"
echo "  - best_ptn_bert.pth"
echo "  - best_ptn_gpt2.pth"
echo "==========================================""
