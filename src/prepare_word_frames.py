"""
Flatten word video frames into ImageFolder-compatible structure for CNN training.

Reads frames from:
    words/<split>/<word>/<video_id>/frame_XXXX.jpg

Creates symlinks (or copies) to:
    word_frames_flat/<split>/<word>/<video_id>_frame_XXXX.jpg

This allows using torchvision.datasets.ImageFolder for training.
"""

import os
import glob
import argparse
from typing import List, Tuple

# ---- CONFIG -----------------------------------------------------------------

WORDS_ROOT = "words"
OUTPUT_ROOT = "word_frames_flat"

# Set to True to copy files instead of creating symlinks
# Symlinks save disk space but may not work on all systems
USE_SYMLINKS = True


# ---- UTILITIES --------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_all_frames(words_root: str, split: str) -> List[Tuple[str, str, str, str]]:
    """
    Get all frame paths for a given split.
    
    Returns list of tuples: (frame_path, word, video_id, frame_name)
    """
    split_dir = os.path.join(words_root, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] Split directory not found: {split_dir}")
        return []
    
    frames = []
    for word in sorted(os.listdir(split_dir)):
        word_dir = os.path.join(split_dir, word)
        if not os.path.isdir(word_dir):
            continue
            
        for video_id in sorted(os.listdir(word_dir)):
            video_dir = os.path.join(word_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
                
            frame_files = sorted(glob.glob(os.path.join(video_dir, "frame_*.jpg")))
            for frame_path in frame_files:
                frame_name = os.path.basename(frame_path)
                frames.append((frame_path, word, video_id, frame_name))
    
    return frames


def flatten_frames(
    words_root: str,
    output_root: str,
    use_symlinks: bool = True
) -> dict:
    """
    Flatten the nested frame structure into ImageFolder format.
    
    Returns statistics about the flattening process.
    """
    stats = {
        "train": {"frames": 0, "classes": set()},
        "val": {"frames": 0, "classes": set()},
        "test": {"frames": 0, "classes": set()},
    }
    
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        frames = get_all_frames(words_root, split)
        
        if not frames:
            print(f"  No frames found for {split}")
            continue
        
        for frame_path, word, video_id, frame_name in frames:
            # Create output directory for this word class
            output_class_dir = os.path.join(output_root, split, word)
            ensure_dir(output_class_dir)
            
            # Create unique filename: video_id + frame_name
            # e.g., "about_1_frame_0000.jpg"
            unique_name = f"{video_id}_{frame_name}"
            output_path = os.path.join(output_class_dir, unique_name)
            
            # Skip if already exists
            if os.path.exists(output_path):
                stats[split]["frames"] += 1
                stats[split]["classes"].add(word)
                continue
            
            # Create symlink or copy
            if use_symlinks:
                # Use absolute path for symlink target
                abs_frame_path = os.path.abspath(frame_path)
                try:
                    os.symlink(abs_frame_path, output_path)
                except OSError as e:
                    # Fallback to copy if symlink fails
                    import shutil
                    shutil.copy2(frame_path, output_path)
            else:
                import shutil
                shutil.copy2(frame_path, output_path)
            
            stats[split]["frames"] += 1
            stats[split]["classes"].add(word)
        
        print(f"  {split}: {stats[split]['frames']} frames, {len(stats[split]['classes'])} classes")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Flatten word video frames for CNN training"
    )
    parser.add_argument(
        "--words_root",
        type=str,
        default=WORDS_ROOT,
        help="Path to words/ directory containing train/val/test splits"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=OUTPUT_ROOT,
        help="Output directory for flattened frames"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    args = parser.parse_args()
    
    use_symlinks = not args.copy
    
    print("=" * 60)
    print("Flattening Word Frames for CNN Training")
    print("=" * 60)
    print(f"Source: {args.words_root}")
    print(f"Output: {args.output_root}")
    print(f"Method: {'symlinks' if use_symlinks else 'copy'}")
    print("=" * 60)
    
    stats = flatten_frames(args.words_root, args.output_root, use_symlinks)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_frames = 0
    all_classes = set()
    for split in ["train", "val", "test"]:
        n_frames = stats[split]["frames"]
        n_classes = len(stats[split]["classes"])
        total_frames += n_frames
        all_classes.update(stats[split]["classes"])
        print(f"  {split:5s}: {n_frames:6d} frames, {n_classes:3d} classes")
    
    print("-" * 60)
    print(f"  Total: {total_frames:6d} frames, {len(all_classes):3d} unique classes")
    print("=" * 60)
    
    # Print class list
    print(f"\nClasses ({len(all_classes)}):")
    for i, cls in enumerate(sorted(all_classes)):
        print(f"  {i:2d}. {cls}")


if __name__ == "__main__":
    main()
