"""
Train Transformer on CNN features ONLY (no keypoints).
Uses class weights to handle imbalanced dataset.

Usage:
    python src/train_transformer_cnn_only.py --epochs 100 --batch_size 16
"""

import os
import sys
import glob
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---- CONFIG -----------------------------------------------------------------

CNN_FEATURES_ROOT = "cnn_features"
MODEL_SAVE_PATH = "models/best_transformer_cnn_only.pth"
MAX_SEQ_LEN = 60
CNN_FEATURE_DIM = 512


# ---- DATASET ----------------------------------------------------------------

class CNNFeatureDataset(Dataset):
    """Dataset that loads only CNN features (no keypoints)."""
    
    def __init__(self, cnn_features_root, split, max_len=60, classes=None):
        self.max_len = max_len
        self.samples = []  # (cnn_path, label_str)
        
        split_dir = os.path.join(cnn_features_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Get classes
        found_classes = sorted([d for d in os.listdir(split_dir) 
                                if os.path.isdir(os.path.join(split_dir, d))])
        
        if classes is not None:
            self.classes = classes
        else:
            self.classes = found_classes
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Collect samples
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            
            for f in os.listdir(cls_dir):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, f), cls_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cnn_path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]
        
        # Load CNN features
        cnn_seq = np.load(cnn_path).astype(np.float32)  # (T, 512)
        
        # Pad/truncate
        T, D = cnn_seq.shape
        if T >= self.max_len:
            cnn_seq = cnn_seq[:self.max_len]
        else:
            padding = np.zeros((self.max_len - T, D), dtype=np.float32)
            cnn_seq = np.vstack([cnn_seq, padding])
        
        return torch.from_numpy(cnn_seq), torch.tensor(label_idx, dtype=torch.long)
    
    def get_class_counts(self):
        """Get number of samples per class for computing weights."""
        counts = Counter(label for _, label in self.samples)
        return [counts.get(cls, 0) for cls in self.classes]


# ---- MODEL ------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """Simplified transformer for small datasets."""
    
    def __init__(self, input_dim=512, num_classes=45, d_model=128, nhead=4,
                 num_layers=2, dropout=0.3, max_len=60):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pe[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pool
        return self.classifier(x)


# ---- TRAINING ---------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)
    
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_features_root", type=str, default=CNN_FEATURES_ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CNNFeatureDataset(args.cnn_features_root, "train", MAX_SEQ_LEN)
    classes = train_dataset.classes
    num_classes = len(classes)
    
    val_dataset = CNNFeatureDataset(args.cnn_features_root, "val", MAX_SEQ_LEN, classes)
    test_dataset = CNNFeatureDataset(args.cnn_features_root, "test", MAX_SEQ_LEN, classes)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Classes: {num_classes}")
    
    # Compute class weights for imbalanced data
    class_counts = train_dataset.get_class_counts()
    total_samples = sum(class_counts)
    class_weights = torch.tensor([
        total_samples / (num_classes * max(count, 1)) for count in class_counts
    ], dtype=torch.float32).to(device)
    print(f"\nClass weights range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleTransformer(
        input_dim=CNN_FEATURE_DIM,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.4,
        max_len=MAX_SEQ_LEN
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    print("\n" + "=" * 60)
    print(f"Training for {args.epochs} epochs")
    print("=" * 60)
    
    best_val_acc = 0
    best_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} / {train_acc:.4f} | "
              f"Val: {val_loss:.4f} / {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(best_state, args.save_path)
            print(f"         -> New best! ({best_val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
    
    # Test
    print("\n" + "=" * 60)
    model.load_state_dict(best_state)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"TEST RESULTS: Loss={test_loss:.4f}, Accuracy={test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60)
    
    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()
