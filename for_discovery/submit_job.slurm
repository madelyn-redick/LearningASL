#!/bin/bash
#SBATCH --job-name=asl_training
#SBATCH --output=logs/asl_training_%j.out
#SBATCH --error=logs/asl_training_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# NOTE: No GPUs available on Discovery - use Explorer cluster for GPU training
#SBATCH --mail-type=END,FAIL

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs models

# Load modules - use anaconda3 which has Python 3.10+
module load anaconda3/2022.05
module load cuda/12.1

echo "Python version: $(python --version)"

# Setup virtual environment (delete old one if it exists with wrong Python)
if [ -d "venv" ]; then
    VENV_PYTHON_VERSION=$(./venv/bin/python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ "$VENV_PYTHON_VERSION" < "3.8" ]]; then
        echo "Removing old venv with Python $VENV_PYTHON_VERSION..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
bash run_all_models.sh

echo "Job completed at $(date)"
