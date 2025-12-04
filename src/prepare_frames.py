"""
Prepare video frames for word-level ASL recognition.

This script:
  1. Downloads the WLASL-based dataset via kagglehub.
  2. Copies videos into a local 'videos_folder/' structure.
  3. Splits videos into train/val/test sets.
  4. Extracts frames from each video into 'words/<split>/<word>/<video_id>/frame_XXXX.jpg'.

This is mostly a refactor of the video/word pipeline from data_preprocessing.ipynb.
"""

import os
import shutil
import random
import string
import glob
from typing import Dict, List

import cv2
import pandas as pd
import kagglehub

# ---- CONFIG -----------------------------------------------------------------

# Root paths are relative to the project root
VIDEOS_ROOT = "videos_folder"
WORDS_ROOT = "words"
UNIGRAM_PATH = "unigram_freq.csv"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Number of most frequent words to keep
NUM_TOP_WORDS = 40

# Random seed for reproducibility
RANDOM_SEED = 42


# ---- UTILITIES --------------------------------------------------------------


def download_wlasl_dataset() -> str:
    """
    Download the WLASL dataset via kagglehub and return the local path
    where the SL videos live.
    """
    print("Downloading WLASL dataset via kagglehub...")
    download_path = kagglehub.dataset_download(
        "waseemnagahhenes/sign-language-dataset-wlasl-videos"
    )
    source_dir = os.path.join(download_path, "dataset", "SL")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Expected SL directory at {source_dir}, but it does not exist."
        )
    print(f"Downloaded dataset to: {download_path}")
    print(f"Found SL videos at: {source_dir}")
    return source_dir


def copy_videos_to_working_dir(source_dir: str, target_dir: str = VIDEOS_ROOT) -> None:
    """
    Copy or merge all files from the dataset SL directory into VIDEOS_ROOT.
    This mirrors the notebook behavior using shutil.copytree(..., dirs_exist_ok=True).
    """
    print(f"Copying videos from {source_dir} to {target_dir} ...")
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    print("Finished copying videos.")


def extract_frames(video_path: str, output_dir: str) -> None:
    """
    Extract frames from a single video and save them as JPGs in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    currentframe = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame_name = f"frame_{currentframe:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        currentframe += 1
    vid.release()
    cv2.destroyAllWindows()


def list_candidate_words(source_root: str) -> List[str]:
    """
    List candidate word directories under source_root, excluding single-letter
    folders (a-z) which are not words.
    """
    all_words = [
        d for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d))
    ]
    alphabet = list(string.ascii_lowercase)
    words_only = [w for w in all_words if w not in alphabet]
    return words_only


def filter_words_by_frequency(
    words: List[str],
    unigram_csv_path: str,
    top_n: int
) -> List[str]:
    """
    Intersect words in the dataset with the unigram frequency list and
    select the top_n most frequent words.
    """
    if not os.path.isfile(unigram_csv_path):
        raise FileNotFoundError(
            f"Unigram frequency file not found at {unigram_csv_path}"
        )

    word_freq = pd.read_csv(unigram_csv_path)
    word_freq = word_freq.loc[word_freq["word"].isin(words)].reset_index(drop=True)

    # Sort by frequency descending if 'count' or 'frequency' column exists.
    freq_col = None
    for candidate in ["count", "frequency", "freq"]:
        if candidate in word_freq.columns:
            freq_col = candidate
            break

    if freq_col is not None:
        word_freq = word_freq.sort_values(by=freq_col, ascending=False)

    selected_words = word_freq["word"].iloc[:top_n].tolist()
    return selected_words


def split_indices(
    n_total: int, train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, List[int]]:
    """
    Given total number of items and split ratios, return index lists for
    train/val/test.
    """
    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def create_words_structure() -> None:
    """
    Ensure the main train/val/test directories under WORDS_ROOT exist.
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(WORDS_ROOT, split), exist_ok=True)


def process_word(
    word: str,
    source_root: str,
    split_map: Dict[str, List[int]],
    video_files: List[str]
) -> None:
    """
    For a given word, split its videos into train/val/test and extract frames
    into the appropriate directory structure.
    """
    for split_name, indices in split_map.items():
        for split_idx, idx in enumerate(indices, start=1):
            video_path = video_files[idx]
            video_id = f"{word}_{split_idx}"
            output_dir = os.path.join(WORDS_ROOT, split_name, word, video_id)
            print(f"Extracting frames for {video_path} -> {output_dir}")
            extract_frames(video_path, output_dir)


# ---- MAIN PIPELINE ----------------------------------------------------------


def main() -> None:
    # 1. Download dataset and copy videos
    source_sl_dir = download_wlasl_dataset()
    copy_videos_to_working_dir(source_sl_dir, VIDEOS_ROOT)

    # 2. List all candidate words in the local videos folder
    candidate_words = list_candidate_words(VIDEOS_ROOT)
    print(f"Found {len(candidate_words)} candidate word folders in {VIDEOS_ROOT}.")

    # 3. Filter words by unigram frequency
    selected_words = filter_words_by_frequency(
        candidate_words,
        UNIGRAM_PATH,
        NUM_TOP_WORDS
    )
    print(f"Selected top {len(selected_words)} frequent words for training.")

    # 4. Ensure words/ train/val/test root folders exist
    create_words_structure()

    # 5. Iterate over selected words and process videos
    for word in selected_words:
        word_folder = os.path.join(VIDEOS_ROOT, word)
        if not os.path.exists(word_folder):
            print(f"[WARN] Word folder not found: {word_folder}, skipping.")
            continue

        # Collect all video files for that word
        video_files = [
            os.path.join(word_folder, f)
            for f in os.listdir(word_folder)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]

        if not video_files:
            print(f"[WARN] No videos found for word {word}, skipping.")
            continue

        print(f"Processing word '{word}' with {len(video_files)} videos.")

        # Compute split indices
        split_map = split_indices(
            n_total=len(video_files),
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=RANDOM_SEED,
        )

        # Process and extract frames
        process_word(word, VIDEOS_ROOT, split_map, video_files)

    print("Frame extraction pipeline complete.")


if __name__ == "__main__":
    main()

