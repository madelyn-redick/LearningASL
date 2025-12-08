"""
Dataset class for loading ASL keypoint sequences and Frame sequences.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from PIL import Image


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


class CombinedFeatureDataset(Dataset):
    """
    Loads both MediaPipe keypoints and CNN features for ASL words.
    Fuses them into a single feature vector per time step.
    Output shape: (T, 42 + 512) -> (T, 554)
    """

    def __init__(
        self,
        keypoints_root: str,
        cnn_features_root: str,
        split: str,
        max_len: int = 60,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            keypoints_root (str): Path to keypoints_data/
            cnn_features_root (str): Path to cnn_features/
            split (str): One of 'train', 'val', 'test'
            max_len (int): Fixed length for padding/truncating sequences
            classes (list): Optional list of class names
        """
        self.keypoints_root = keypoints_root
        self.cnn_features_root = cnn_features_root
        self.split = split
        self.max_len = max_len
        self.samples: List[Tuple[str, str, str]] = []  # (kp_path, cnn_path, label_str)

        # 1. Determine Classes
        split_dir_kp = os.path.join(keypoints_root, split)
        if not os.path.isdir(split_dir_kp):
            # Try without keypoints dir to check if user made a typo or path issue
            # But we must fail if keypoints missing.
            raise FileNotFoundError(f"Keypoints split directory not found: {split_dir_kp}")

        found_classes = sorted(
            d for d in os.listdir(split_dir_kp)
            if os.path.isdir(os.path.join(split_dir_kp, d))
        )

        if classes is not None:
            self.classes = classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        else:
            self.classes = found_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(found_classes)}

        # 2. Collect paired samples
        for cls_name in self.classes:
            kp_cls_dir = os.path.join(split_dir_kp, cls_name)
            cnn_cls_dir = os.path.join(cnn_features_root, split, cls_name)

            if not os.path.isdir(kp_cls_dir):
                continue
            
            # We iterate through keypoint files and look for corresponding CNN feature files
            kp_files = glob.glob(os.path.join(kp_cls_dir, "*.npy"))
            for kp_path in kp_files:
                filename = os.path.basename(kp_path) # e.g. "word_1.npy"
                cnn_path = os.path.join(cnn_cls_dir, filename)
                
                if os.path.isfile(cnn_path):
                    self.samples.append((kp_path, cnn_path, cls_name))
                else:
                    # If CNN feature missing, just skip or warn?
                    # print(f"Missing CNN features for {kp_path}")
                    pass

        if not self.samples:
            print(f"[WARN] No paired samples found in {split} for classes {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp_path, cnn_path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]

        # Load Keypoints: (T1, 42)
        kp_seq = np.load(kp_path).astype(np.float32)
        
        # Load CNN Features: (T2, 512)
        cnn_seq = np.load(cnn_path).astype(np.float32)
        
        # Synchronize lengths
        # In theory T1 == T2 if processed from same frames.
        # But if errors occurred, they might differ. We take the min length.
        min_len = min(kp_seq.shape[0], cnn_seq.shape[0])
        kp_seq = kp_seq[:min_len]
        cnn_seq = cnn_seq[:min_len]
        
        # Concatenate: (T, 554)
        combined_seq = np.concatenate([kp_seq, cnn_seq], axis=1)
        
        T, D = combined_seq.shape
        
        # Padding / Truncation
        if T >= self.max_len:
            combined_seq = combined_seq[:self.max_len, :]
        else:
            pad_len = self.max_len - T
            padding = np.zeros((pad_len, D), dtype=np.float32)
            combined_seq = np.vstack([combined_seq, padding])

        x_tensor = torch.from_numpy(combined_seq)
        y_tensor = torch.tensor(label_idx, dtype=torch.long)

        return x_tensor, y_tensor
