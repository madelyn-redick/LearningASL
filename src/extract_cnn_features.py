"""
Extract CNN features from video frames using trained ASL_CNN model.

Reads frames from:
    words/<split>/<word>/<video_id>/frame_XXXX.jpg

Writes feature sequences to:
    cnn_features/<split>/<word>/<video_id>.npy

Each .npy file has shape:
    (T, 512) where T is number of frames and 512 is the CNN feature dimension.

Usage:
    python src/extract_cnn_features.py --model_path models/best_cnn_words.pth
"""

import os
import sys
import glob
import argparse
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cnn_model import ASL_CNN, ASL_CNN_FeatureExtractor

# ---- CONFIG -----------------------------------------------------------------

WORDS_ROOT = "words"
CNN_FEATURES_ROOT = "cnn_features"
DEFAULT_MODEL_PATH = "models/best_cnn_words.pth"

# Transform must match training
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---- UTILITIES --------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_feature_extractor(model_path: str, device: torch.device) -> ASL_CNN_FeatureExtractor:
    """Load the trained CNN and wrap it as a feature extractor."""
    # Load checkpoint to determine number of classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine num_classes from the final layer
    if '_base_model.fc.3.weight' in checkpoint:
        num_classes = checkpoint['_base_model.fc.3.weight'].shape[0]
    else:
        # Try to infer from state dict
        for key in checkpoint.keys():
            if 'fc' in key and 'weight' in key:
                num_classes = checkpoint[key].shape[0]
                break
        else:
            num_classes = 45  # Default
    
    print(f"Loading model with {num_classes} classes...")
    
    # Create and load model
    base_model = ASL_CNN(num_classes=num_classes)
    base_model.load_state_dict(checkpoint)
    
    # Wrap as feature extractor
    feature_extractor = ASL_CNN_FeatureExtractor(base_model)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    return feature_extractor


def extract_features_for_video(
    video_dir: str,
    feature_extractor: ASL_CNN_FeatureExtractor,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract CNN features for all frames in a video directory.
    
    Returns:
        np.ndarray of shape (T, 512) where T is number of frames
    """
    # Get all frame paths
    frame_paths = sorted(glob.glob(os.path.join(video_dir, "frame_*.jpg")))
    
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    
    # Load and transform all frames
    frames = []
    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path).convert('RGB')
            img_tensor = CNN_TRANSFORM(img)
            frames.append(img_tensor)
        except Exception as e:
            print(f"Warning: Could not load {frame_path}: {e}")
            # Use zero tensor for failed frames
            frames.append(torch.zeros(3, 128, 128))
    
    # Stack into batch
    frames_tensor = torch.stack(frames)  # (T, 3, 128, 128)
    
    # Extract features in batches
    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames_tensor), batch_size):
            batch = frames_tensor[i:i+batch_size].to(device)
            features = feature_extractor(batch)  # (batch, 512)
            all_features.append(features.cpu().numpy())
    
    # Concatenate all batches
    features_array = np.vstack(all_features)  # (T, 512)
    
    return features_array


def process_all_videos(
    words_root: str,
    output_root: str,
    feature_extractor: ASL_CNN_FeatureExtractor,
    device: torch.device,
    batch_size: int = 32
) -> dict:
    """
    Process all videos in words/ and extract CNN features.
    
    Returns statistics about the extraction process.
    """
    stats = {
        "train": {"videos": 0, "frames": 0},
        "val": {"videos": 0, "frames": 0},
        "test": {"videos": 0, "frames": 0},
        "skipped": 0,
        "errors": 0
    }
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = os.path.join(words_root, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory not found: {split_dir}, skipping.")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get all word classes
        words = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        
        for word in tqdm(words, desc=f"{split} words"):
            word_dir = os.path.join(split_dir, word)
            
            # Get all video directories
            video_ids = sorted([
                d for d in os.listdir(word_dir)
                if os.path.isdir(os.path.join(word_dir, d))
            ])
            
            for video_id in video_ids:
                video_dir = os.path.join(word_dir, video_id)
                
                # Output path
                output_dir = os.path.join(output_root, split, word)
                ensure_dir(output_dir)
                output_path = os.path.join(output_dir, f"{video_id}.npy")
                
                # Skip if already exists
                if os.path.exists(output_path):
                    stats["skipped"] += 1
                    continue
                
                try:
                    # Extract features
                    features = extract_features_for_video(
                        video_dir, feature_extractor, device, batch_size
                    )
                    
                    # Save
                    np.save(output_path, features)
                    
                    stats[split]["videos"] += 1
                    stats[split]["frames"] += features.shape[0]
                    
                except Exception as e:
                    print(f"\n[ERROR] Failed to process {video_dir}: {e}")
                    stats["errors"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract CNN features from video frames"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained CNN model (best_cnn_words.pth)"
    )
    parser.add_argument(
        "--words_root",
        type=str,
        default=WORDS_ROOT,
        help="Path to words/ directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=CNN_FEATURES_ROOT,
        help="Output directory for CNN features"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction"
    )
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print("Please train the CNN first or provide the correct path.")
        sys.exit(1)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load feature extractor
    print(f"\nLoading model from {args.model_path}...")
    feature_extractor = load_feature_extractor(args.model_path, device)
    print("Model loaded successfully!")
    
    # Test feature extraction
    print("\nTesting feature extractor...")
    test_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        test_output = feature_extractor(test_input)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    assert test_output.shape == (1, 512), f"Expected (1, 512), got {test_output.shape}"
    print("Feature extractor test passed!")
    
    # Process all videos
    print("\n" + "=" * 60)
    print("Extracting CNN Features")
    print("=" * 60)
    print(f"Source: {args.words_root}")
    print(f"Output: {args.output_root}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    stats = process_all_videos(
        args.words_root,
        args.output_root,
        feature_extractor,
        device,
        args.batch_size
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    total_videos = 0
    total_frames = 0
    for split in ["train", "val", "test"]:
        n_videos = stats[split]["videos"]
        n_frames = stats[split]["frames"]
        total_videos += n_videos
        total_frames += n_frames
        print(f"  {split:5s}: {n_videos:4d} videos, {n_frames:6d} frames")
    print("-" * 60)
    print(f"  Total: {total_videos:4d} videos, {total_frames:6d} frames")
    print(f"  Skipped (already exist): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    print("=" * 60)
    
    print(f"\nCNN features saved to: {args.output_root}/")
    print("Each .npy file has shape (T, 512)")


if __name__ == "__main__":
    main()
