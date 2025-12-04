"""
Pose Transformer Network (PTN) definition.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard Transformer sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) to broadcast across batch
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    """
    Transformer-based model for classifying sequences of pose keypoints.

    Input: (batch_size, seq_len, input_dim)
    Output: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 42,       # 21 landmarks * 2 (x, y)
        num_classes: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        super().__init__()

        # Linear projection from raw coordinate space to latent d_model space
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Inputs are (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        # We'll use global average pooling over the time dimension
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, input_dim)
        """
        # 1. Project inputs
        x = self.input_proj(x)  # (B, T, d_model)

        # 2. Add positional encoding
        x = self.pos_encoder(x) # (B, T, d_model)

        # 3. Transformer layers
        # Output is (B, T, d_model) since batch_first=True
        x = self.transformer_encoder(x)

        # 4. Global Average Pooling
        # Average across the time dimension (T) -> (B, d_model)
        # (Masking padding tokens would be better, but simple averaging works okay for fixed length)
        x = x.mean(dim=1)

        # 5. Classifier
        logits = self.classifier(x) # (B, num_classes)
        return logits

