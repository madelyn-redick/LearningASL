# Training Pipeline Setup

## Overview
This document describes the complete training pipeline that utilizes all your data sources:
- `images_folder/` → CNN pretraining (letter classification)
- `keypoints_data/` → Transformer pretraining (keypoint sequences)  
- `words/` → Final end-to-end training (frame sequences)

## Steps Completed

### ✅ Step 1: CNN Training Script Created
**File:** `src/train_cnn.py`

Trains `ASL_CNN` on `images_folder` (A-Z, DEL, SPACE letters) and saves weights to `models/best_cnn_params.pth`.

**Usage:**
```bash
python3 src/train_cnn.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-3 \
    --save_path models/best_cnn_params.pth \
    --data_root images_folder
```

### ✅ Step 2: Keypoints Data Ready
Your `keypoints_data/` folder already contains extracted keypoint sequences from MediaPipe.

### ✅ Step 3: PTN Training Script Updated
**File:** `src/train_ptn.py` (imports fixed)

Trains `PoseTransformer` on `keypoints_data` and saves weights to `models/best_ptn_model.pth`.

**Usage:**
```bash
python3 src/train_ptn.py \
    --data_root keypoints_data \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --save_path models/best_ptn_model.pth
```

### ✅ Step 4: Notebook Updated
**File:** `notebooks/03_demo_cnn_transformer.ipynb`

The notebook now:
- ✅ Loads pretrained CNN from `models/best_cnn_params.pth` (if exists)
- ✅ Loads pretrained Transformer from `models/best_ptn_model.pth` (if exists)
- ✅ Trains end-to-end with backpropagation through both CNN and Transformer
- ✅ Uses real `words/` dataset (no dummy fallback)
- ✅ Improved hyperparameters (20 epochs, LR=1e-4)

## Quick Start

### Option 1: Run All Steps Automatically
```bash
./run_training_pipeline.sh
```

### Option 2: Run Steps Individually

**1. Train CNN:**
```bash
python3 src/train_cnn.py --epochs 20 --batch_size 32 --save_path models/best_cnn_params.pth
```

**2. Train PTN:**
```bash
python3 src/train_ptn.py --data_root keypoints_data --epochs 50 --save_path models/best_ptn_model.pth
```

**3. Run Notebook:**
Open `notebooks/03_demo_cnn_transformer.ipynb` and run all cells. It will:
- Load pretrained weights (if available)
- Train on `words/` frame sequences
- Show training metrics and evaluation

## Data Flow

```
images_folder/ (40k+ images)
    ↓
[CNN Training] → models/best_cnn_params.pth
                    ↓
                [Feature Extractor]
                    ↓
keypoints_data/ (sequences)
    ↓
[PTN Training] → models/best_ptn_model.pth
                    ↓
                [Transformer Weights]
                    ↓
words/ (frame sequences)
    ↓
[End-to-End Training] → Final Model
```

## Notes

- **CNN Training**: Uses 80/20 train/val split from `images_folder`
- **PTN Training**: Uses `keypoints_data/train` and `keypoints_data/val`
- **Final Training**: Uses `words/train`, `words/val`, `words/test` frame sequences
- All models support both CPU and GPU (automatically detected)

## Troubleshooting

If you encounter architecture issues (e.g., ARM vs x86_64), ensure you're using the correct Python environment with compatible packages. Consider using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

