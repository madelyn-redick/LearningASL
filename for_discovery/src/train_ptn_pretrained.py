"""
Training script for Pose Transformer Network with pretrained backbone options.

Uses either:
1. A pretrained Time Series Transformer from HuggingFace (generic, not ASL-specific)
2. A pretrained GPT-2 adapted for sequence classification
3. Self-supervised MAE pretraining on keypoints (most fair approach)

Includes:
- Data augmentation for small datasets
- Label smoothing for better generalization
- Mixup regularization
- Early stopping
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset import KeypointDataset


# ============== Regularization Helpers ==============

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_preds)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return (-smooth_labels * log_preds).sum(dim=-1).mean()


def mixup_data(x, y, alpha=0.2):
    """Mixup: creates mixed inputs and targets for regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


class PretrainedPoseTransformer(nn.Module):
    """
    Uses a pretrained GPT-2 (tiny) as the sequence encoder.
    GPT-2 is trained on text but learns general sequence patterns.
    This is a "fair" approach - no ASL-specific pretraining.
    """
    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 40,
        pretrained_model: str = "distilgpt2",  # Smallest GPT-2 variant
        freeze_backbone: bool = False,
        dropout: float = 0.5  # Higher dropout for small datasets
    ):
        super().__init__()
        try:
            from transformers import GPT2Model, GPT2Config
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # Load pretrained GPT-2 (smallest version)
        self.backbone = GPT2Model.from_pretrained(pretrained_model)
        hidden_size = self.backbone.config.hidden_size  # 768 for distilgpt2
        
        # Input dropout for regularization
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Project keypoints to GPT-2's embedding dimension
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Optionally freeze backbone for transfer learning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """x: (batch, seq_len, 42)"""
        # Input dropout for regularization
        x = self.input_dropout(x)
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Pass through pretrained transformer
        outputs = self.backbone(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Global average pooling
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Classify
        logits = self.classifier(pooled)
        return logits


class LightweightPretrainedPTN(nn.Module):
    """
    A more lightweight approach using pretrained BERT encoder layers.
    Uses only a few layers from a pretrained model to reduce compute.
    """
    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 40,
        d_model: int = 256,
        num_layers: int = 4,
        dropout: float = 0.5  # Higher dropout
    ):
        super().__init__()
        try:
            from transformers import BertModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # Load pretrained BERT and extract just the encoder layers
        bert = BertModel.from_pretrained("prajjwal1/bert-tiny")  # 4.4M params, very small
        bert_hidden = bert.config.hidden_size  # 128 for bert-tiny
        
        # Input dropout for regularization
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Use BERT's encoder layers (pretrained attention patterns)
        self.encoder_layers = nn.ModuleList([
            bert.encoder.layer[i] for i in range(min(num_layers, len(bert.encoder.layer)))
        ])
        
        # Project input to BERT dimension
        self.input_proj = nn.Linear(input_dim, bert_hidden)
        
        # Learned positional embeddings (smaller than sinusoidal)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, bert_hidden) * 0.02)
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(bert_hidden),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden, bert_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden // 2, num_classes)
        )
    
    def forward(self, x):
        """x: (batch, seq_len, 42)"""
        batch_size, seq_len, _ = x.shape
        
        # Input dropout for regularization
        x = self.input_dropout(x)
        
        # Project and add position
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Pass through pretrained encoder layers
        for layer in self.encoder_layers:
            layer_output = layer(x)
            x = layer_output[0]
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        return self.classifier(pooled)


class SelfSupervisedPTN(nn.Module):
    """
    Self-supervised pretraining approach using Masked Autoencoder.
    This learns from YOUR data without labels first, then finetunes.
    Most "fair" approach - no external data at all.
    
    Improved with higher dropout for small datasets.
    """
    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 40,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.5,  # Higher dropout for small datasets
        mask_ratio: float = 0.4  # Higher mask ratio for better pretraining
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.dropout_rate = dropout
        
        # Encoder with input dropout
        self.input_dropout = nn.Dropout(dropout * 0.5)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for reconstruction (pretraining)
        self.decoder = nn.Linear(d_model, input_dim)
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.pretrain_mode = True
    
    def forward(self, x, pretrain=None):
        """
        x: (batch, seq_len, 42)
        If pretrain=True, returns reconstruction loss
        If pretrain=False, returns classification logits
        """
        pretrain = pretrain if pretrain is not None else self.pretrain_mode
        batch_size, seq_len, _ = x.shape
        
        if pretrain:
            # Mask some frames
            mask = torch.rand(batch_size, seq_len, device=x.device) < self.mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0
            
            # Encode
            h = self.input_proj(x_masked)
            h = h + self.pos_embed[:, :seq_len, :]
            h = self.encoder(h)
            
            # Reconstruct
            reconstruction = self.decoder(h)
            
            # Return both reconstruction and original for loss computation
            return reconstruction, x, mask
        else:
            # Classification mode with input dropout
            x = self.input_dropout(x)
            h = self.input_proj(x)
            h = h + self.pos_embed[:, :seq_len, :]
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.classifier(pooled)


def pretrain_mae(model, dataloader, epochs, device, lr=1e-3):
    """Self-supervised pretraining using masked reconstruction."""
    model.pretrain_mode = True
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    print("\n=== Self-Supervised Pretraining ===")
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in dataloader:  # Ignore labels during pretraining
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            reconstruction, original, mask = model(inputs, pretrain=True)
            
            # Only compute loss on masked positions
            loss = ((reconstruction - original) ** 2)
            loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * model.input_dim + 1e-8)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Pretrain Epoch [{epoch+1}/{epochs}] Reconstruction Loss: {total_loss/len(dataloader):.4f}")
    
    model.pretrain_mode = False
    print("=== Pretraining Complete ===\n")
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Standard training without mixup for now (sanity check)
        outputs = model(inputs, pretrain=False) if hasattr(model, 'pretrain_mode') else model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, pretrain=False) if hasattr(model, 'pretrain_mode') else model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train PTN with pretrained models")
    parser.add_argument("--data_root", type=str, default="keypoints_data")
    parser.add_argument("--epochs", type=int, default=150)  # More epochs with early stopping
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)  # Slightly lower LR
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--save_path", type=str, default="models/best_ptn_pretrained.pth")
    parser.add_argument("--model_type", type=str, default="mae",
                       choices=["gpt2", "bert-tiny", "mae"],
                       help="gpt2: pretrained GPT-2, bert-tiny: pretrained BERT-tiny, mae: self-supervised")
    parser.add_argument("--pretrain_epochs", type=int, default=100,  # More pretraining
                       help="Number of self-supervised pretraining epochs (only for mae)")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze pretrained weights (only for gpt2/bert)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                       help="Mixup alpha parameter")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    parser.add_argument("--no_augment", action="store_true",
                       help="Disable data augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Label smoothing: {args.label_smoothing}, Mixup alpha: {args.mixup_alpha}")
    print(f"Data augmentation: {'OFF' if args.no_augment else 'ON'}")

    # Create datasets WITH AUGMENTATION for training
    train_dataset = KeypointDataset(
        args.data_root, 
        split="train", 
        max_len=args.max_len,
        augment=not args.no_augment,  # Enable augmentation
        augmentation_config={
            'temporal_crop_ratio': 0.85,
            'scale_range': (0.85, 1.15),
            'noise_std': 0.015,
            'speed_range': (0.9, 1.1),
            'flip_prob': 0.5,
            'dropout_prob': 0.1,
            'shift_range': 0.08,
        }
    )
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    print(f"Training samples: {len(train_dataset)}")

    # Validation WITHOUT augmentation
    val_dataset = KeypointDataset(
        args.data_root, 
        split="val", 
        max_len=args.max_len, 
        classes=classes,
        augment=False  # No augmentation for validation
    )
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model based on type
    if args.model_type == "gpt2":
        print("Loading pretrained DistilGPT-2...")
        model = PretrainedPoseTransformer(
            input_dim=42,
            num_classes=num_classes,
            pretrained_model="distilgpt2",
            freeze_backbone=args.freeze_backbone,
            dropout=0.1  # Low dropout to force learning
        ).to(device)
    elif args.model_type == "bert-tiny":
        print("Loading pretrained BERT-tiny...")
        model = LightweightPretrainedPTN(
            input_dim=42,
            num_classes=num_classes,
            dropout=0.1  # Low dropout to force learning
        ).to(device)
    else:  # mae
        print("Using self-supervised MAE pretraining...")
        model = SelfSupervisedPTN(
            input_dim=42,
            num_classes=num_classes,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.1,  # Low dropout to force learning
            mask_ratio=0.3
        ).to(device)
        
        # Pretrain with masked autoencoder (using augmented data)
        model = pretrain_mae(model, train_loader, args.pretrain_epochs, device, lr=1e-3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Use standard CrossEntropy for now (easier optimization)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=0.001  # Lower weight decay
    )
    
    # Learning rate scheduler (simple cosine)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=False  # Disable mixup
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        if val_acc >= best_val_acc:  # Save even if equal
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'model_type': args.model_type,
                'best_val_acc': best_val_acc
            }, args.save_path)
            print(f"  -> New best model saved! ({best_val_acc:.4f})")
        
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

