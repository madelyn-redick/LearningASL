"""
Dataset class for loading ASL keypoint sequences.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional


class KeypointDataset(Dataset):
    """
    Loads pre-computed .npy keypoint sequences for ASL words.
    Each sequence has shape (T, 42) representing 21 (x, y) landmarks over T frames.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        max_len: int = 60,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir (str): Path to keypoints_data/
            split (str): One of 'train', 'val', 'test'
            max_len (int): Fixed length for padding/truncating sequences
            classes (list): Optional list of class names to enforce specific label indices
        """
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len
        self.samples: List[Tuple[str, str]] = []  # (path, label_str)

        # Gather classes automatically if not provided
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        found_classes = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )

        if classes is not None:
            self.classes = classes
            # Filter samples to only include known classes
            valid_classes = set(classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.classes = found_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(found_classes)}

        # Collect all .npy files
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            npy_files = glob.glob(os.path.join(cls_dir, "*.npy"))
            for fpath in npy_files:
                self.samples.append((fpath, cls_name))

        if not self.samples:
            print(f"[WARN] No samples found in {split_dir} for classes {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]

        # Load sequence: (T, 42)
        sequence = np.load(path).astype(np.float32)
        T, D = sequence.shape

        # Padding / Truncation
        if T >= self.max_len:
            # Truncate (can also sample or center crop, but simple truncation is fine)
            sequence = sequence[:self.max_len, :]
        else:
            # Pad with zeros
            pad_len = self.max_len - T
            padding = np.zeros((pad_len, D), dtype=np.float32)
            sequence = np.vstack([sequence, padding])

        # Convert to tensor
        # PyTorch nn.Transformer expects input shape (SeqLen, Batch, Dim) by default,
        # or (Batch, SeqLen, Dim) if batch_first=True.
        # We'll return (SeqLen, Dim) here, and let DataLoader batch them to (Batch, SeqLen, Dim).
        x_tensor = torch.from_numpy(sequence)
        y_tensor = torch.tensor(label_idx, dtype=torch.long)

        return x_tensor, y_tensor

