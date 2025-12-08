#!/bin/bash
# Training Pipeline Script
# Run this to execute steps 1-3 of the training pipeline

set -e

echo "=========================================="
echo "ASL Training Pipeline"
echo "=========================================="
echo ""

# Step 1: Train CNN on images_folder
echo "Step 1: Training CNN on images_folder..."
python3 src/train_cnn.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --save_path models/best_cnn_params.pth \
    --data_root images_folder

echo ""
echo "✓ Step 1 Complete: CNN trained and saved to models/best_cnn_params.pth"
echo ""

# Step 2: Already done (keypoints_data exists)
echo "Step 2: ✓ Already complete (keypoints_data exists)"
echo ""

# Step 3: Train PTN on keypoints_data
echo "Step 3: Training Pose Transformer Network on keypoints_data..."
python3 src/train_ptn.py \
    --data_root keypoints_data \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --save_path models/best_ptn_model.pth

echo ""
echo "✓ Step 3 Complete: PTN trained and saved to models/best_ptn_model.pth"
echo ""

echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next: Run notebooks/03_demo_cnn_transformer.ipynb"
echo "  It will automatically load:"
echo "    - models/best_cnn_params.pth (CNN weights)"
echo "    - models/best_ptn_model.pth (Transformer weights)"
echo ""
