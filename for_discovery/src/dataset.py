"""
Dataset class for loading ASL keypoint sequences and Frame sequences.
Includes data augmentation for improved generalization.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from PIL import Image
import random


class KeypointAugmentation:
    """
    Data augmentation for keypoint sequences to improve generalization.
    Critical for small datasets!
    """
    
    def __init__(
        self,
        temporal_crop_ratio: float = 0.8,  # Random crop 80% of sequence
        scale_range: Tuple[float, float] = (0.8, 1.2),  # Scale coordinates
        noise_std: float = 0.02,  # Add gaussian noise
        speed_range: Tuple[float, float] = (0.8, 1.2),  # Temporal speed change
        flip_prob: float = 0.5,  # Horizontal flip probability
        dropout_prob: float = 0.1,  # Randomly drop frames
        shift_range: float = 0.1,  # Random spatial shift
    ):
        self.temporal_crop_ratio = temporal_crop_ratio
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.speed_range = speed_range
        self.flip_prob = flip_prob
        self.dropout_prob = dropout_prob
        self.shift_range = shift_range
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random augmentations to keypoint sequence (T, 42)"""
        sequence = sequence.copy()
        T, D = sequence.shape
        
        # 1. Random temporal crop (take a random subsequence)
        if self.temporal_crop_ratio < 1.0 and T > 5:
            crop_len = max(3, int(T * self.temporal_crop_ratio))
            start = random.randint(0, T - crop_len)
            sequence = sequence[start:start + crop_len]
            T = sequence.shape[0]
        
        # 2. Random temporal speed change (interpolate to different length)
        if self.speed_range[0] != 1.0 or self.speed_range[1] != 1.0:
            speed = random.uniform(*self.speed_range)
            new_len = max(3, int(T / speed))
            if new_len != T:
                indices = np.linspace(0, T - 1, new_len)
                sequence = np.array([
                    np.interp(indices, np.arange(T), sequence[:, i])
                    for i in range(D)
                ]).T
                T = sequence.shape[0]
        
        # 3. Random frame dropout (drop some frames)
        if self.dropout_prob > 0 and T > 5:
            keep_mask = np.random.random(T) > self.dropout_prob
            keep_mask[0] = True  # Always keep first
            keep_mask[-1] = True  # Always keep last
            if keep_mask.sum() >= 3:
                sequence = sequence[keep_mask]
                T = sequence.shape[0]
        
        # 4. Random scale (zoom in/out)
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = random.uniform(*self.scale_range)
            # Center the keypoints, scale, then shift back
            center = sequence.mean(axis=0)
            sequence = (sequence - center) * scale + center
        
        # 5. Random spatial shift
        if self.shift_range > 0:
            shift_x = random.uniform(-self.shift_range, self.shift_range)
            shift_y = random.uniform(-self.shift_range, self.shift_range)
            sequence[:, 0::2] += shift_x  # x coordinates
            sequence[:, 1::2] += shift_y  # y coordinates
        
        # 6. Horizontal flip (mirror the hand)
        if random.random() < self.flip_prob:
            # Flip x coordinates around center (0.5)
            sequence[:, 0::2] = 1.0 - sequence[:, 0::2]
        
        # 7. Add gaussian noise
        if self.noise_std > 0:
            noise = np.random.randn(*sequence.shape) * self.noise_std
            sequence = sequence + noise
        
        # Clip to valid range
        sequence = np.clip(sequence, 0, 1)
        
        return sequence.astype(np.float32)


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes keypoints to be relative to the wrist (landmark 0) and scale invariant.
    sequence: (T, 42)
    """
    # Reshape to (T, 21, 2)
    T = sequence.shape[0]
    landmarks = sequence.reshape(T, 21, 2)
    
    # 1. Center around wrist (landmark 0)
    wrist = landmarks[:, 0:1, :]  # (T, 1, 2)
    landmarks = landmarks - wrist
    
    # 2. Scale normalization (divide by max distance from wrist)
    # This makes the hand size invariant
    distances = np.sqrt(np.sum(landmarks ** 2, axis=2))  # (T, 21)
    max_dist = np.max(distances, axis=1, keepdims=True) + 1e-6  # (T, 1)
    # Broadcast to (T, 21, 1) for division
    landmarks = landmarks / max_dist[..., None]
    
    return landmarks.reshape(T, 42)


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
        augment: bool = False,
        augmentation_config: Optional[dict] = None,
        normalize: bool = True,  # New flag
    ):
        """
        Args:
            root_dir (str): Path to keypoints_data/
            split (str): One of 'train', 'val', 'test'
            max_len (int): Fixed length for padding/truncating sequences
            classes (list): Optional list of class names to enforce specific label indices
            augment (bool): Whether to apply data augmentation (use for training only!)
            augmentation_config (dict): Optional config for augmentation parameters
            normalize (bool): Whether to normalize keypoints relative to wrist
        """
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len
        self.normalize = normalize
        self.samples: List[Tuple[str, str]] = []  # (path, label_str)
        
        # Setup augmentation
        self.augment = augment
        if augment:
            aug_config = augmentation_config or {}
            self.augmentor = KeypointAugmentation(**aug_config)
        else:
            self.augmentor = None

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
        else:
            print(f"[INFO] {split}: {len(self.samples)} samples, {len(self.classes)} classes" + 
                  (", augmentation ON" if augment else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]

        # Load sequence: (T, 42)
        sequence = np.load(path).astype(np.float32)
        
        # Apply normalization (CRITICAL for small datasets)
        if self.normalize:
            sequence = normalize_keypoints(sequence)
        
        # Apply augmentation (only during training)
        if self.augmentor is not None:
            sequence = self.augmentor(sequence)
        
        T, D = sequence.shape

        # Padding / Truncation
        if T >= self.max_len:
            # Random crop during training, center crop during eval
            if self.augment:
                start = random.randint(0, T - self.max_len)
                sequence = sequence[start:start + self.max_len, :]
            else:
                sequence = sequence[:self.max_len, :]
        else:
            # Pad with zeros
            pad_len = self.max_len - T
            padding = np.zeros((pad_len, D), dtype=np.float32)
            sequence = np.vstack([sequence, padding])

        # Convert to tensor
        x_tensor = torch.from_numpy(sequence)
        y_tensor = torch.tensor(label_idx, dtype=torch.long)

        return x_tensor, y_tensor


class FrameSequenceDataset(Dataset):
    """
    Loads sequences of image frames for ASL words.
    Expects structure: root_dir/split/class_name/video_id/frame_0000.jpg
    Returns: (SeqLen, Channels, Height, Width) tensor and label index.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        max_len: int = 60,
        transform=None,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir (str): Path to words/ (containing train/val/test)
            split (str): One of 'train', 'val', 'test'
            max_len (int): Fixed length for padding/truncating sequences
            transform (callable, optional): Transform to apply to each frame (e.g. resize, normalize)
            classes (list): Optional list of class names
        """
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []  # (video_dir_path, label_str)

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            # Fallback check if root_dir already includes split or is just the data root
            # Assuming standard structure words/split/
            if os.path.isdir(root_dir) and any(os.path.isdir(os.path.join(root_dir, d)) for d in os.listdir(root_dir)):
                 # Maybe root_dir is already the split dir? No, let's enforce structure.
                 pass
            # If fail, just warn later.
        
        if not os.path.isdir(split_dir):
             raise FileNotFoundError(f"Split directory not found: {split_dir}")

        found_classes = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )

        if classes is not None:
            self.classes = classes
            valid_classes = set(classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.classes = found_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(found_classes)}

        # Collect video directories
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            
            # Each subdirectory here is a video_id
            video_dirs = [
                os.path.join(cls_dir, d) 
                for d in os.listdir(cls_dir) 
                if os.path.isdir(os.path.join(cls_dir, d))
            ]
            
            for v_path in video_dirs:
                self.samples.append((v_path, cls_name))
        
        if not self.samples:
            print(f"[WARN] No video samples found in {split_dir} for classes {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]

        # Get all frame files
        # Assuming frames are named frame_0000.jpg, etc.
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        
        if not frame_files:
             # Handle empty video dir? Should not happen if data prep is good.
             # Return zero tensor of shape (max_len, 3, 128, 128)
             # assuming 128x128 default
             print(f"Warning: No frames in {video_path}")
             return torch.zeros((self.max_len, 3, 128, 128)), torch.tensor(label_idx, dtype=torch.long)

        # Load images
        frames_list = []
        for fpath in frame_files:
            try:
                img = Image.open(fpath).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                else:
                    # Default to tensor if no transform provided
                    import torchvision.transforms.functional as TF
                    img = TF.to_tensor(img)
                frames_list.append(img)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
        
        if not frames_list:
             return torch.zeros((self.max_len, 3, 128, 128)), torch.tensor(label_idx, dtype=torch.long)

        # Stack into (T_actual, C, H, W)
        sequence = torch.stack(frames_list)
        T = sequence.shape[0]

        # Pad / Truncate
        if T >= self.max_len:
            sequence = sequence[:self.max_len]
        else:
            pad_len = self.max_len - T
            # sequence shape: (T, C, H, W)
            # padding shape: (pad_len, C, H, W)
            padding = torch.zeros((pad_len, *sequence.shape[1:]), dtype=sequence.dtype)
            sequence = torch.cat([sequence, padding])

        return sequence, torch.tensor(label_idx, dtype=torch.long)
