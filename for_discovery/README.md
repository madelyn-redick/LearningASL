# ASL Training on Discovery Cluster

This folder contains everything needed to train the ASL Pose Transformer models on a compute cluster.

## Contents

```
for_discovery/
├── keypoints_data/       # Preprocessed keypoint sequences (.npy files)
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Dataset loader
│   └── train_ptn_pretrained.py  # Training script with 3 model options
├── models/               # Output directory for trained models (created automatically)
├── requirements.txt      # Python dependencies
├── run_all_models.sh     # Script to train all 3 models
├── submit_job.slurm      # SLURM job submission script
└── README.md             # This file
```

## Quick Start

### Option 1: SLURM Cluster (e.g., Discovery)

```bash
# Upload this folder to the cluster
scp -r for_discovery/ username@discovery.neu.edu:~/

# SSH into the cluster
ssh username@discovery.neu.edu

# Navigate and submit job
cd for_discovery
sbatch submit_job.slurm

# Monitor job
squeue -u $USER
tail -f logs/asl_training_*.out
```

### Option 2: Any Linux Server with GPU

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training (all 3 models)
bash run_all_models.sh

# Or run individual models:
python3 src/train_ptn_pretrained.py --model_type mae --epochs 100
python3 src/train_ptn_pretrained.py --model_type bert-tiny --epochs 100
python3 src/train_ptn_pretrained.py --model_type gpt2 --freeze_backbone --epochs 100
```

### Option 3: Google Colab

Upload this folder to Google Drive, then in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/for_discovery
!pip install -r requirements.txt
!bash run_all_models.sh
```

## Models Being Trained

| Model | Description | Params | Notes |
|-------|-------------|--------|-------|
| `mae` | Self-supervised MAE pretraining | ~200K | Most fair - no external data |
| `bert-tiny` | Pretrained BERT-tiny encoder | ~4.4M | Lightweight pretrained |
| `gpt2` | Pretrained DistilGPT-2 (frozen) | ~82M | Only classifier trainable |

## Output

After training completes, download these model files:

```
models/
├── best_ptn_mae.pth
├── best_ptn_bert.pth
└── best_ptn_gpt2.pth
```

Download command:
```bash
scp username@discovery.neu.edu:~/for_discovery/models/*.pth ./models/
```

## Estimated Time

- MAE: ~30-45 min (50 pretrain + 100 finetune epochs)
- BERT-tiny: ~20-30 min (100 epochs)
- GPT-2: ~45-60 min (100 epochs, larger model)

Total: ~2-3 hours with GPU

## Troubleshooting

**Module not found errors:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA out of memory:**
```bash
# Reduce batch size
python3 src/train_ptn_pretrained.py --batch_size 8 ...
```

**SLURM partition issues:**
Edit `submit_job.slurm` and change `--partition=gpu` to match your cluster's GPU partition name.
