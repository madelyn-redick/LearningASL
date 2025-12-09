# VideoMAE Fine-tuning on ASL Dataset

Fine-tune a VideoMAE transformer on the [American Sign Language Dataset](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset) from Hugging Face.

## Dataset Info
- **108,618 videos** representing **2,207 ASL words**
- Each word has minimum 30 videos
- Videos are automatically downloaded from Hugging Face Hub

## Setup

### 1. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Quick test (50 samples, 1 epoch)
```bash
python src/train_videomae_hf.py \
    --max_samples 50 \
    --num_train_epochs 1 \
    --output_dir output/test-run
```

### Full training (GPU recommended)
```bash
python src/train_videomae_hf.py \
    --output_dir output/videomae-asl \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --fp16
```

### Training on CPU (slow, for testing only)
```bash
python src/train_videomae_hf.py \
    --max_samples 100 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --output_dir output/cpu-test
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_ckpt` | `MCG-NJU/videomae-base-finetuned-kinetics` | VideoMAE checkpoint |
| `--num_train_epochs` | 10 | Number of training epochs |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 4 | Accumulate gradients for larger effective batch |
| `--learning_rate` | 5e-5 | Learning rate |
| `--num_frames` | 16 | Frames to sample per video |
| `--max_samples` | None | Limit dataset size (for testing) |
| `--train_split` | 0.9 | Train/eval split ratio |
| `--fp16` | False | Enable mixed precision (GPU only) |
| `--output_dir` | `output/videomae-asl` | Where to save model |

## Output

After training, you'll find:
- `output/videomae-asl/final/` - Final model weights
- `output/videomae-asl/final/label2id.json` - Word to class ID mapping
- `output/videomae-asl/final/id2label.json` - Class ID to word mapping
- Checkpoints saved after each epoch

## SLURM (Cluster)

If running on a SLURM cluster:
```bash
sbatch submit_job.slurm
```

## Notes

- **Storage**: Videos are cached in `~/.cache/huggingface/`. Full dataset is ~100GB.
- **GPU Memory**: With batch_size=4 and 16 frames, expect ~12-16GB VRAM usage.
- **Training Time**: Full training on a single GPU takes several hours to days depending on hardware.

## Model Checkpoints

You can use different VideoMAE checkpoints:
- `MCG-NJU/videomae-base` - Base model (pretrained)
- `MCG-NJU/videomae-base-finetuned-kinetics` - Finetuned on Kinetics-400 (default, recommended)
- `MCG-NJU/videomae-large-finetuned-kinetics` - Larger model (more VRAM needed)
