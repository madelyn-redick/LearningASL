"""
Process HuggingFace ASL videos: extract frames -> CNN features.
Then train LSTM classifier.

Usage:
    python src/process_hf_videos.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import random

# --- CONFIG ---
HF_ROOT = "hf_asl_videos"
OUTPUT_ROOT = "hf_cnn_features"
MODEL_PATH = "models/best_cnn_params.pth"
MAX_FRAMES = 60
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


class ASL_CNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self._base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self._base_model.parameters():
            param.requires_grad = False
        in_features = self._base_model.fc.in_features
        self._base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self._base_model(x)

    def get_features(self, x):
        model = self._base_model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc[0](x)  # Linear(2048, 512)
        x = model.fc[1](x)  # ReLU
        return x


def extract_features_from_video(model, transform, video_path, max_frames=60):
    """Extract CNN features from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        return None

    # Sample frames if too many
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Extract features in batches
    batch = torch.stack([transform(f) for f in frames])
    with torch.no_grad():
        features = model.get_features(batch).numpy()

    return features


def main():
    print("=" * 60)
    print("Processing HuggingFace ASL Videos")
    print("=" * 60)

    # Load model
    print("\nLoading CNN model...")
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    num_classes = checkpoint['_base_model.fc.3.weight'].shape[0]
    model = ASL_CNN(num_classes=num_classes)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Loaded model with {num_classes} classes")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process all videos
    print(f"\nExtracting features from {HF_ROOT}...")
    words = sorted([d for d in os.listdir(HF_ROOT)
                    if os.path.isdir(os.path.join(HF_ROOT, d))])
    print(f"Found {len(words)} words")

    all_samples = []  # (features, word)
    for word in tqdm(words, desc="Processing words"):
        word_dir = os.path.join(HF_ROOT, word)
        videos = [f for f in os.listdir(word_dir) if f.endswith('.mp4')]

        for vid in videos:
            vid_path = os.path.join(word_dir, vid)
            features = extract_features_from_video(model, transform, vid_path, MAX_FRAMES)
            if features is not None and len(features) > 0:
                all_samples.append((features, word))

    print(f"\nExtracted features from {len(all_samples)} videos")

    # Split into train/val/test
    random.seed(42)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Save
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for split, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        split_dir = os.path.join(OUTPUT_ROOT, split)
        os.makedirs(split_dir, exist_ok=True)

        word_counts = {}
        for features, word in samples:
            word_dir = os.path.join(split_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            word_counts[word] = word_counts.get(word, 0) + 1
            out_path = os.path.join(word_dir, f"{word}_{word_counts[word]}.npy")
            np.save(out_path, features)

    print(f"\nSaved features to {OUTPUT_ROOT}/")
    print("Done!")


if __name__ == "__main__":
    main()
