"""
Download ASL videos from Hugging Face dataset for words in the local project.

This script downloads videos from:
https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset

It matches words from the dataset with the top N most frequent English words
from the local unigram_freq.csv file.

Videos are named like: part_X/{id}-{WORD}.mp4

Usage:
    # Without auth (rate limited):
    python download_hf_asl_videos.py

    # With Hugging Face token (recommended):
    export HF_TOKEN=your_token_here
    python download_hf_asl_videos.py
    
    # Or login via CLI first:
    huggingface-cli login
    python download_hf_asl_videos.py
"""

import os
import sys
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm
import string
import re
from collections import defaultdict
import shutil
import time

# ============== CONFIGURATION ==============

# Hugging Face dataset info
REPO_ID = "ZahidYasinMittha/American-Sign-Language-Dataset"
REPO_TYPE = "dataset"

# Local paths
UNIGRAM_PATH = "unigram_freq.csv"
OUTPUT_DIR = "hf_asl_videos"  # Where to save downloaded videos

# How many words to download
NUM_TOP_WORDS = 60  # Adjust this to match your needs

# How many videos per word (dataset has minimum 30 per word)
VIDEOS_PER_WORD = 30

# Rate limiting - delay between downloads (seconds)
# Increase this if you hit rate limits
DELAY_BETWEEN_DOWNLOADS = 0.5  # seconds

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds to wait before retrying after rate limit

# ============== FUNCTIONS ==============


def get_top_frequent_words(unigram_path: str, top_n: int) -> list:
    """
    Get the top N most frequent English words from unigram_freq.csv,
    excluding single letters.
    """
    if not os.path.isfile(unigram_path):
        raise FileNotFoundError(f"Unigram frequency file not found: {unigram_path}")
    
    df = pd.read_csv(unigram_path)
    
    # Sort by count (frequency) descending
    if "count" in df.columns:
        df = df.sort_values(by="count", ascending=False)
    
    # Filter out single letters
    alphabet = set(string.ascii_lowercase)
    df = df[~df["word"].isin(alphabet)]
    
    # Get top N words (uppercase to match HF dataset)
    top_words = [w.upper() for w in df["word"].head(top_n).tolist()]
    return top_words


def get_all_video_files() -> list:
    """
    Get list of all video files in the HF dataset.
    """
    print("Fetching file list from Hugging Face (this may take a moment)...")
    api = HfApi()
    all_files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )
    # Filter to only .mp4 files
    video_files = [f for f in all_files if f.endswith('.mp4')]
    return video_files


def build_word_to_videos_map(video_files: list) -> dict:
    """
    Parse video filenames to build a mapping of word -> list of video paths.
    
    Video naming convention: part_X/{id}-{WORD}.mp4
    Example: part_1/01653635897919692-YOU.mp4
    """
    word_to_videos = defaultdict(list)
    
    # Pattern to extract word from filename
    # Matches: {digits}-{WORD}.mp4 or just {WORD}.mp4
    pattern = re.compile(r'(?:\d+-)?(.+?)\.mp4$', re.IGNORECASE)
    
    for video_path in video_files:
        # Get the filename part
        filename = os.path.basename(video_path)
        
        match = pattern.match(filename)
        if match:
            word = match.group(1).upper()
            word_to_videos[word].append(video_path)
    
    return dict(word_to_videos)


def find_matching_words(target_words: list, available_words: dict) -> list:
    """
    Find which target words are available in the dataset.
    Returns list of matching words (uppercase).
    """
    target_set = set(w.upper() for w in target_words)
    available_set = set(available_words.keys())
    
    matching = target_set.intersection(available_set)
    missing = target_set - available_set
    
    print(f"\nWord matching results:")
    print(f"  - Target words: {len(target_words)}")
    print(f"  - Matching in dataset: {len(matching)}")
    print(f"  - Missing from dataset: {len(missing)}")
    
    if missing:
        print(f"\nMissing words: {sorted(missing)}")
    
    return sorted(matching)


def get_existing_videos(word: str, output_dir: str) -> set:
    """
    Get set of video filenames already downloaded for a word.
    """
    word_dir = os.path.join(output_dir, word.lower())
    if not os.path.exists(word_dir):
        return set()
    
    return set(f for f in os.listdir(word_dir) if f.endswith('.mp4'))


def download_with_retry(repo_id: str, filename: str, repo_type: str, max_retries: int = MAX_RETRIES) -> str | None:
    """
    Download a file with retry logic for rate limiting.
    """
    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"\n  Rate limited. Waiting {RETRY_DELAY}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\n  Rate limited. Max retries reached.")
                    return None
            else:
                # Other error, don't retry
                return None
    return None


def download_videos_for_word(
    word: str,
    video_paths: list,
    max_videos: int,
    output_dir: str
) -> tuple[int, int]:
    """
    Download up to max_videos for a given word.
    Returns (downloaded_count, skipped_count).
    """
    word_dir = os.path.join(output_dir, word.lower())
    os.makedirs(word_dir, exist_ok=True)
    
    # Get already downloaded videos
    existing = get_existing_videos(word, output_dir)
    
    # Limit to max_videos
    videos_to_download = video_paths[:max_videos]
    downloaded = 0
    skipped = 0
    
    for video_path in videos_to_download:
        filename = os.path.basename(video_path)
        
        # Skip if already downloaded
        if filename in existing:
            skipped += 1
            continue
        
        # Add delay to avoid rate limiting
        if downloaded > 0:
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
        
        try:
            local_path = download_with_retry(REPO_ID, video_path, REPO_TYPE)
            
            if local_path:
                # Copy to our output directory with organized structure
                dest_path = os.path.join(word_dir, filename)
                if not os.path.exists(dest_path):
                    shutil.copy2(local_path, dest_path)
                downloaded += 1
        except Exception as e:
            print(f"\n  [WARN] Failed to download {video_path}: {e}")
    
    return downloaded, skipped


def main():
    print("=" * 60)
    print("ASL Video Downloader from Hugging Face")
    print("=" * 60)
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("\n✓ Hugging Face token found in environment")
    else:
        print("\n⚠ No HF_TOKEN found. You may hit rate limits.")
        print("  To avoid rate limits, run: huggingface-cli login")
        print("  Or set: export HF_TOKEN=your_token")
    
    # 1. Get top frequent words from local unigram file
    print(f"\n[1/4] Loading top {NUM_TOP_WORDS} frequent words from {UNIGRAM_PATH}...")
    target_words = get_top_frequent_words(UNIGRAM_PATH, NUM_TOP_WORDS)
    print(f"Top words: {target_words[:10]}... (showing first 10)")
    
    # 2. Get all video files from HF and build word mapping
    print(f"\n[2/4] Building word-to-video mapping from Hugging Face...")
    video_files = get_all_video_files()
    print(f"Found {len(video_files)} video files in dataset")
    
    word_to_videos = build_word_to_videos_map(video_files)
    print(f"Mapped to {len(word_to_videos)} unique words")
    
    # 3. Find matching words
    print(f"\n[3/4] Finding matching words...")
    matching_words = find_matching_words(target_words, word_to_videos)
    
    if not matching_words:
        print("\nNo matching words found! Exiting.")
        return
    
    # Calculate expected downloads (accounting for existing)
    total_to_download = 0
    total_existing = 0
    print(f"\nMatching words and their status:")
    
    for word in matching_words:
        available = len(word_to_videos.get(word, []))
        existing = len(get_existing_videos(word, OUTPUT_DIR))
        to_download = min(available, VIDEOS_PER_WORD) - existing
        if to_download < 0:
            to_download = 0
        total_to_download += to_download
        total_existing += existing
        status = "✓" if existing >= VIDEOS_PER_WORD else "↓"
        print(f"  {status} {word}: {existing}/{min(available, VIDEOS_PER_WORD)} downloaded")
    
    print(f"\nSummary:")
    print(f"  - Words: {len(matching_words)}")
    print(f"  - Already downloaded: {total_existing}")
    print(f"  - To download: {total_to_download}")
    
    if total_to_download == 0:
        print("\nAll videos already downloaded!")
        return
    
    # Ask for confirmation
    response = input("\nProceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # 4. Download videos
    print(f"\n[4/4] Downloading videos to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_downloaded = 0
    total_skipped = 0
    
    for word in tqdm(matching_words, desc="Words"):
        videos = word_to_videos.get(word, [])
        downloaded, skipped = download_videos_for_word(
            word,
            videos,
            VIDEOS_PER_WORD,
            OUTPUT_DIR
        )
        total_downloaded += downloaded
        total_skipped += skipped
        
        if downloaded > 0:
            tqdm.write(f"  {word}: +{downloaded} new")
    
    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"  - New videos downloaded: {total_downloaded}")
    print(f"  - Previously downloaded: {total_skipped}")
    print(f"  - Total: {total_downloaded + total_skipped + total_existing}")
    print(f"Location: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
