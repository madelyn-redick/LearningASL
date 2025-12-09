"""
Fine-tune VideoMAE on the Hugging Face ASL dataset.

Dataset: https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset

Usage (GPU recommended):
    pip install -r requirements.txt
    python src/train_videomae_hf.py \
        --output_dir output/videomae-asl \
        --model_ckpt MCG-NJU/videomae-base-finetuned-kinetics
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from transformers import (
    AutoModelForVideoClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    VideoMAEImageProcessor,
)

REPO_ID = "ZahidYasinMittha/American-Sign-Language-Dataset"
CSV_FILENAME = "Aslense Dataset.csv"


def download_csv() -> str:
    """Download the labels CSV from the Hub."""
    return hf_hub_download(repo_id=REPO_ID, filename=CSV_FILENAME, repo_type="dataset")


def build_filename_to_path_mapping() -> Dict[str, str]:
    """Build a mapping from video filename to full repo path."""
    files = list_repo_files(REPO_ID, repo_type="dataset")
    mapping = {}
    for f in files:
        if f.endswith(".mp4"):
            basename = f.split("/")[-1]
            mapping[basename] = f
    return mapping


def download_video(repo_path: str) -> str:
    """Download a single video file and return local path."""
    return hf_hub_download(repo_id=REPO_ID, filename=repo_path, repo_type="dataset")


class ASLVideoDataset(Dataset):
    """Dataset that loads videos from HF Hub based on CSV labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        filename_to_path: Dict[str, str],
        label2id: Dict[str, int],
        processor: VideoMAEImageProcessor,
        num_frames: int = 16,
    ):
        self.df = df.reset_index(drop=True)
        self.filename_to_path = filename_to_path
        self.label2id = label2id
        self.processor = processor
        self.num_frames = num_frames

        # Filter to only rows where video exists in repo
        valid_mask = self.df["videos"].isin(filename_to_path.keys())
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"Dataset has {len(self.df)} valid samples")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        video_filename = row["videos"]
        word = row["word"]
        label = self.label2id[word]

        # Get repo path and download video
        repo_path = self.filename_to_path[video_filename]
        try:
            local_path = download_video(repo_path)
            # Read video frames using torchvision
            video, _, _ = read_video(local_path, pts_unit="sec")
            # video shape: (T, H, W, C)
            total_frames = video.shape[0]

            if total_frames == 0:
                raise ValueError("Empty video")

            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = video[indices].numpy()  # (num_frames, H, W, C)

            # Convert to PIL images for processor
            images = [Image.fromarray(f) for f in frames]

            # Process frames
            inputs = self.processor(images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)

        except Exception as e:
            print(f"Error loading video {video_filename}: {e}")
            # Return dummy tensor on error
            pixel_values = torch.zeros((self.num_frames, 3, 224, 224))

        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on ASL HF dataset")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        help="VideoMAE checkpoint to start from",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="output/videomae-asl")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size for quick testing",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.9, help="Fraction of data for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps for larger effective batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("VideoMAE Fine-tuning on ASL Dataset")
    print("=" * 60)

    print("\nDownloading CSV labels...")
    csv_path = download_csv()
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV with {df['word'].nunique()} unique words")

    print("\nBuilding filename to path mapping (this may take a moment)...")
    filename_to_path = build_filename_to_path_mapping()
    print(f"Found {len(filename_to_path)} video files in repo")

    # Build label mappings
    all_words = sorted(df["word"].unique())
    label2id = {w: i for i, w in enumerate(all_words)}
    id2label = {i: w for w, i in label2id.items()}
    num_labels = len(all_words)
    print(f"Number of classes: {num_labels}")

    # Limit samples if requested
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)
        print(f"Limited to {len(df)} samples for testing")

    # Split into train/eval
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")

    # Load processor and model
    print(f"\nLoading model: {args.model_ckpt}")
    processor = VideoMAEImageProcessor.from_pretrained(args.model_ckpt)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model_ckpt,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create datasets
    train_dataset = ASLVideoDataset(
        train_df, filename_to_path, label2id, processor, args.num_frames
    )
    eval_dataset = ASLVideoDataset(
        eval_df, filename_to_path, label2id, processor, args.num_frames
    )

    # Metrics
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues with HF downloads
        report_to="none",  # Disable wandb/tensorboard by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Evaluating best checkpoint...")
    print("=" * 60)
    results = trainer.evaluate()
    print(f"Eval results: {results}")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nSaved final model to: {final_dir}")

    # Save label mappings
    import json
    with open(os.path.join(final_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(final_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)
    print(f"Saved label mappings to: {final_dir}")


if __name__ == "__main__":
    main()
