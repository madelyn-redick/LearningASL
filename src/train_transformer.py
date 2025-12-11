"""
Train Transformer on combined CNN + Keypoint features for word recognition.

Uses:
- Keypoints from keypoints_data/<split>/<word>/<video_id>.npy (shape: T, 42)
- CNN features from cnn_features/<split>/<word>/<video_id>.npy (shape: T, 512)
- Combined input: (T, 554)

Usage:
    python src/train_transformer.py --epochs 50 --batch_size 32
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import CombinedFeatureDataset
from src.model import PoseTransformer

# ---- CONFIG -----------------------------------------------------------------

KEYPOINTS_ROOT = "keypoints_data"
CNN_FEATURES_ROOT = "cnn_features"
MODEL_SAVE_PATH = "models/best_combined_model.pth"
MAX_SEQ_LEN = 60

# Input dimensions
KEYPOINT_DIM = 42   # 21 landmarks * 2 (x, y)
CNN_FEATURE_DIM = 512
COMBINED_DIM = KEYPOINT_DIM + CNN_FEATURE_DIM  # 554


# ---- TRAINING FUNCTIONS -----------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    return epoch_loss, epoch_acc.item()


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer on combined features"
    )
    parser.add_argument("--keypoints_root", type=str, default=KEYPOINTS_ROOT,
                        help="Path to keypoints_data/")
    parser.add_argument("--cnn_features_root", type=str, default=CNN_FEATURES_ROOT,
                        help="Path to cnn_features/")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN,
                        help="Maximum sequence length")
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH,
                        help="Path to save best model")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--target_acc", type=float, default=0.95,
                        help="Stop early if validation accuracy exceeds this")
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check data directories exist
    if not os.path.isdir(args.keypoints_root):
        print(f"ERROR: Keypoints directory not found: {args.keypoints_root}")
        print("Please run: python src/extract_keypoints.py")
        sys.exit(1)
    
    if not os.path.isdir(args.cnn_features_root):
        print(f"ERROR: CNN features directory not found: {args.cnn_features_root}")
        print("Please run: python src/extract_cnn_features.py")
        sys.exit(1)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Train dataset (establishes class order)
    train_dataset = CombinedFeatureDataset(
        keypoints_root=args.keypoints_root,
        cnn_features_root=args.cnn_features_root,
        split="train",
        max_len=args.max_len
    )
    
    classes = train_dataset.classes
    num_classes = len(classes)
    
    # Val/Test datasets use same class mapping
    val_dataset = CombinedFeatureDataset(
        keypoints_root=args.keypoints_root,
        cnn_features_root=args.cnn_features_root,
        split="val",
        max_len=args.max_len,
        classes=classes
    )
    
    test_dataset = CombinedFeatureDataset(
        keypoints_root=args.keypoints_root,
        cnn_features_root=args.cnn_features_root,
        split="test",
        max_len=args.max_len,
        classes=classes
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print(f"Classes ({num_classes}): {classes}")
    
    if len(train_dataset) == 0:
        print("\nERROR: No training samples found!")
        print("Make sure keypoints and CNN features are extracted for the same videos.")
        sys.exit(1)
    
    # Create dataloaders
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers)
    
    # Initialize model
    print(f"\nInitializing PoseTransformer...")
    print(f"  Input dim: {COMBINED_DIM} (keypoints: {KEYPOINT_DIM} + CNN: {CNN_FEATURE_DIM})")
    print(f"  Model dim: {args.d_model}")
    print(f"  Heads: {args.nhead}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Classes: {num_classes}")
    
    model = PoseTransformer(
        input_dim=COMBINED_DIM,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        dropout=0.1,
        max_len=args.max_len
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting training for up to {args.epochs} epochs")
    print(f"Early stopping at {args.target_acc:.0%} validation accuracy")
    print("=" * 70)
    
    best_acc = 0.0
    best_model_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step(val_acc)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(best_model_state, args.save_path)
            print(f"-> New best model! (Val Acc: {best_acc:.4f})")
        
        # Early stopping
        if val_acc >= args.target_acc:
            print(f"\n*** Target accuracy {args.target_acc:.0%} reached! Stopping early. ***")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print("=" * 70)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(best_model_state)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 70)
    
    # Save class mapping
    class_mapping_path = os.path.join(os.path.dirname(args.save_path), "transformer_classes.txt")
    with open(class_mapping_path, "w") as f:
        for i, cls in enumerate(classes):
            f.write(f"{i},{cls}\n")
    print(f"\nClass mapping saved to: {class_mapping_path}")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
