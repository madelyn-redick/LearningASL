"""
Training script for Combined (CNN + Keypoints) Transformer model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import CombinedFeatureDataset
from src.model import PoseTransformer


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Combined (CNN + Keypoints) Transformer")
    parser.add_argument("--keypoints_root", type=str, default="keypoints_data", help="Path to keypoints data")
    parser.add_argument("--cnn_features_root", type=str, default="cnn_features", help="Path to cnn features")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=60, help="Sequence length")
    parser.add_argument("--save_path", type=str, default="models/best_combined_model.pth", help="Path to save best model")
    
    # Model Hyperparams
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create Datasets
    # We create the train dataset first to get the definitive list of classes
    print("Initializing Train Dataset...")
    train_dataset = CombinedFeatureDataset(
        keypoints_root=args.keypoints_root, 
        cnn_features_root=args.cnn_features_root,
        split="train", 
        max_len=args.max_len
    )
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    print("Initializing Validation Dataset...")
    val_dataset = CombinedFeatureDataset(
        keypoints_root=args.keypoints_root, 
        cnn_features_root=args.cnn_features_root,
        split="val", 
        max_len=args.max_len, 
        classes=classes
    )
    
    # Check if we have data
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check if keypoints and cnn_features exist.")
        return

    # 2. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Initialize Model
    # Input dim = 42 (Keypoints) + 512 (CNN) = 554
    INPUT_DIM = 42 + 512
    
    model = PoseTransformer(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        dropout=0.3,
        max_len=args.max_len
    ).to(device)

    print(f"Model created with input_dim={INPUT_DIM}, d_model={args.d_model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Ensure save dir exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 4. Training Loop
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> New best model saved! ({best_val_acc:.4f})")
            
        # Early stopping condition
        if val_acc >= 0.99:
            print(f"Validation accuracy reached {val_acc:.4f} >= 0.99. Stopping early.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
