"""
Training script for CNN letter classification on images_folder.
"""

import os
import sys
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


def train_model(model, loss_fn, optimizer, scheduler, dataloader, device,
                best_model_path='best_model.pth', num_epochs=20, print_every=10):
    """Train the CNN model."""
    print("=" * 80)
    print(f"Starting training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Train batches: {len(dataloader['train'])}")
    print(f"Validation batches: {len(dataloader['validation'])}")
    print("=" * 80)

    model = model.to(device)
    torch.save(model.state_dict(), best_model_path)
    best_accuracy = 0.0

    import time
    overall_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\n{"="*80}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*80}')

        for phase in ['train', 'validation']:
            phase_start_time = time.time()

            if phase == 'train':
                model.train()
                print(f"\nTraining Phase")
            else:
                model.eval()
                print(f"\nValidation Phase")

            cumulative_loss = 0.0
            cumulative_corrects = 0
            cumulative_samples = 0
            dataset_size = len(dataloader[phase].dataset)
            num_batches = len(dataloader[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                cumulative_loss += loss.item() * inputs.size(0)
                cumulative_corrects += torch.sum(predictions == labels.data)
                cumulative_samples += inputs.size(0)

                # Print progress every N batches
                if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
                    current_loss = cumulative_loss / cumulative_samples
                    current_acc = cumulative_corrects.double() / cumulative_samples
                    progress = (batch_idx + 1) / num_batches * 100

                    print(f"  Batch [{batch_idx + 1:4d}/{num_batches:4d}] ({progress:5.1f}%) | "
                          f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}", flush=True)

            if phase == 'train':
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Learning rate: {current_lr:.6f}")

            epoch_loss = cumulative_loss / dataset_size
            epoch_accuracy = cumulative_corrects.double() / dataset_size
            phase_time = time.time() - phase_start_time

            print(f"\n  {phase.upper()} RESULTS:")
            print(f"  Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f} | Time: {phase_time:.1f}s")

            # Save best model
            if phase == 'validation' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved (Accuracy: {best_accuracy:.4f})")

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - overall_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs

        print(f"\nEpoch time: {epoch_time:.1f}s | Elapsed: {elapsed_time/60:.1f}m | ETA: {eta/60:.1f}m")

    total_time = time.time() - overall_start_time
    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Loading best model from {best_model_path}")
    print(f"{'='*80}\n")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CNN on images_folder")
    parser.add_argument("--data_root", type=str, default="images_folder", help="Path to images folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/best_cnn_params.pth", help="Path to save best model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets using ImageFolder (expects structure: data_root/class_name/image.jpg)
    # Since images_folder has structure: images_folder/A/A_1.jpg, we can use ImageFolder directly
    train_dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    
    # Split into train/val manually (80/20 split)
    import random
    random.seed(42)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'validation': val_loader
    }

    # Initialize model
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    model = ASL_CNN(num_classes=num_classes)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Train
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device,
                       best_model_path=args.save_path, num_epochs=args.epochs)
    
    print(f"Training complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()

