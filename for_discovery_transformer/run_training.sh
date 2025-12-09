#!/bin/bash
# Run VideoMAE fine-tuning on ASL dataset
# Usage: ./run_training.sh [test|full]

set -e

MODE=${1:-"test"}

if [ "$MODE" == "test" ]; then
    echo "Running quick test with 100 samples..."
    python src/train_videomae_hf.py \
        --max_samples 100 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --output_dir output/test-run

elif [ "$MODE" == "full" ]; then
    echo "Running full training..."
    python src/train_videomae_hf.py \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --fp16 \
        --output_dir output/videomae-asl

else
    echo "Usage: ./run_training.sh [test|full]"
    echo "  test - Quick test with 100 samples, 1 epoch"
    echo "  full - Full training on entire dataset"
    exit 1
fi

echo "Done!"
