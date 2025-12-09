#!/bin/bash
#SBATCH --job-name=videomae-asl
#SBATCH --output=logs/videomae_%j.out
#SBATCH --error=logs/videomae_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Create logs directory
mkdir -p logs

# Activate virtual environment if using one
# source venv/bin/activate

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Run training
python src/train_videomae_hf.py \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --fp16 \
    --output_dir output/videomae-asl

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
