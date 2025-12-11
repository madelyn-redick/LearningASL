"""
Train CNN on word frames for feature extraction.

Uses the flattened word_frames_flat/ structure created by prepare_word_frames.py.
Trains the ASL_CNN model on word classes instead of letters.

Usage:
    python src/train_cnn_words.py --epochs 15 --batch_size 32
    
For GPU training (recommended), run on Google Colab or a machine with CUDA.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.cnn_model import ASL_CNN

# ---- CONFIG -----------------------------------------------------------------

DATA_ROOT = "word_frames_flat"
MODEL_SAVE_PATH = "models/best_cnn_words.pth"

# Image transforms (must match inference time)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---- TRAINING FUNCTIONS -----------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, print_every=50):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
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
        
        if (batch_idx + 1) % print_every == 0:
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {current_loss:.4f} Acc: {current_acc:.4f}")
    
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
        for inputs, labels in dataloader:
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


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    print_every=50,
    target_acc=0.99
):
    """Full training loop with validation. Stops early if accuracy exceeds target_acc."""
    print("=" * 70)
    print(f"Starting training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Target accuracy for early stop: {target_acc:.2%}")
    print("=" * 70)
    
    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        print("Training...")
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, print_every
        )
        
        # Validation phase
        print("Validating...")
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! (Val Acc: {best_acc:.4f})")
        
        # Early stopping if accuracy exceeds target
        if val_acc >= target_acc:
            print(f"\n*** Target accuracy {target_acc:.2%} reached! Stopping early. ***")
            break
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Val Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {save_path}")
    print("=" * 70)
    
    return history, best_acc


def evaluate_test(model, dataloader, criterion, device):
    """Evaluate on test set."""
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    test_loss, test_acc = validate(model, dataloader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 70)
    
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN on word frames")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help="Path to word_frames_flat/")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for pretrained layers")
    parser.add_argument("--fc_lr", type=float, default=1e-3,
                        help="Learning rate for new FC layers")
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH,
                        help="Path to save best model")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print progress every N batches")
    parser.add_argument("--target_acc", type=float, default=0.99,
                        help="Stop training when val accuracy exceeds this (default: 0.99)")
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    test_dir = os.path.join(args.data_root, "test")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    val_dataset = datasets.ImageFolder(val_dir, transform=VAL_TRANSFORM)
    test_dataset = datasets.ImageFolder(test_dir, transform=VAL_TRANSFORM)
    
    num_classes = len(train_dataset.classes)
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print(f"Classes ({num_classes}): {train_dataset.classes}")
    
    # Save class mapping for later use
    class_mapping_path = os.path.join(os.path.dirname(args.save_path), "word_classes.txt")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(class_mapping_path, "w") as f:
        for i, cls in enumerate(train_dataset.classes):
            f.write(f"{i},{cls}\n")
    print(f"Class mapping saved to: {class_mapping_path}")
    
    # Create dataloaders
    num_workers = 2 if device.type == "cuda" else 0
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=num_workers),
    }
    
    # Initialize model
    print(f"\nInitializing ASL_CNN with {num_classes} classes...")
    model = ASL_CNN(num_classes=num_classes)
    model = model.to(device)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with different learning rates
    # Lower LR for pretrained layers, higher for new FC layers
    pretrained_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                fc_params.append(param)
            else:
                pretrained_params.append(param)
    
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': args.lr},
        {'params': fc_params, 'lr': args.fc_lr}
    ])
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Train
    history, best_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_path=args.save_path,
        print_every=args.print_every,
        target_acc=args.target_acc
    )
    
    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    evaluate_test(model, dataloaders['test'], criterion, device)
    
    print("\nTraining pipeline complete!")
    print(f"Model saved to: {args.save_path}")
    print(f"Use this model with ASL_CNN_FeatureExtractor to extract 512-dim features.")


if __name__ == "__main__":
    main()
