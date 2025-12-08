"""
LearningASL - American Sign Language Recognition

This package provides models and utilities for ASL recognition:
- CNN-based letter classification (A-Z, DEL, SPACE)
- Transformer-based word recognition from video sequences
"""

from __future__ import annotations

import inspect

import torch

from .cnn_model import ASL_CNN, ASL_CNN_FeatureExtractor
from .model import PoseTransformer, ASLSequenceModel, PositionalEncoding
from .dataset import KeypointDataset, FrameSequenceDataset


# ---------------------------------------------------------------------------
# Compatibility: allow torch.load(..., weights_only=True) on older PyTorch
# ---------------------------------------------------------------------------
_load_sig = inspect.signature(torch.load)
if "weights_only" not in _load_sig.parameters:
    _torch_load_orig = torch.load

    def _torch_load_compat(*args, **kwargs):
        """
        Backwards-compatible wrapper for torch.load that ignores the
        'weights_only' keyword argument if it is not supported.
        """

        kwargs.pop("weights_only", None)
        return _torch_load_orig(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]


__all__ = [
    # CNN Models
    "ASL_CNN",
    "ASL_CNN_FeatureExtractor",
    # Transformer Models
    "PoseTransformer",
    "ASLSequenceModel",
    "PositionalEncoding",
    # Datasets
    "KeypointDataset",
    "FrameSequenceDataset",
]

