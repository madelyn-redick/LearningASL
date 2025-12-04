"""
Extract hand keypoints from video frames using MediaPipe Hands.

Reads frame sequences from:
    words/<split>/<word>/<video_id>/frame_XXXX.jpg

Writes keypoint sequences to:
    keypoints_data/<split>/<word>/<video_id>.npy

Each .npy file has shape:
    (T, 42) where T is number of frames and 42 = 21 landmarks * 2 (x, y).
"""

import os
import glob
from typing import List, Optional

import numpy as np
import cv2
import mediapipe as mp

# ---- CONFIG -----------------------------------------------------------------

WORDS_ROOT = "words"
KEYPOINTS_ROOT = "keypoints_data"

# Whether to normalize landmarks to [0, 1] coordinates (relative to image size)
NORMALIZE_COORDS = True

# Maximum frames per sequence (optional downsampling/truncation per video)
MAX_FRAMES_PER_VIDEO = None  # or set e.g. 120


# ---- MEDIAPIPE SETUP --------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # not used, but handy for debugging


def extract_hand_keypoints_from_image(
    image_bgr,
    hands: mp_hands.Hands,
) -> Optional[np.ndarray]:
    """
    Run MediaPipe Hands on a single BGR image and return a (42,) array
    of (x, y) for 21 landmarks. If no hands are detected, returns None.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        return None

    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []
    for lm in hand_landmarks.landmark:
        if NORMALIZE_COORDS:
            # lm.x and lm.y are already normalized to [0, 1] w.r.t. image width/height
            coords.extend([lm.x, lm.y])
        else:
            # Convert to pixel coordinates if needed
            h, w, _ = image_bgr.shape
            coords.extend([lm.x * w, lm.y * h])

    return np.array(coords, dtype=np.float32)


def process_video_folder(
    video_folder: str,
    hands: mp_hands.Hands,
) -> np.ndarray:
    """
    Given a directory containing frames for one video, run hand
    keypoint extraction on each frame and build a (T, 42) array.
    Frames where no hand is detected will be zero-vectors.
    """
    frame_paths = sorted(
        glob.glob(os.path.join(video_folder, "frame_*.jpg"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {video_folder}")

    sequence = []
    count_frames = 0

    for frame_path in frame_paths:
        if MAX_FRAMES_PER_VIDEO is not None and count_frames >= MAX_FRAMES_PER_VIDEO:
            break

        image_bgr = cv2.imread(frame_path)
        if image_bgr is None:
            # Corrupt or unreadable image; use zeros
            keypoints = np.zeros(42, dtype=np.float32)
        else:
            kps = extract_hand_keypoints_from_image(image_bgr, hands)
            if kps is None:
                # If no hand detected, we use zeros
                keypoints = np.zeros(42, dtype=np.float32)
            else:
                keypoints = kps

        sequence.append(keypoints)
        count_frames += 1

    seq_array = np.stack(sequence, axis=0)  # (T, 42)
    return seq_array


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    """
    Walk through words/<split>/<word>/<video_id>/ and generate
    keypoints_data/<split>/<word>/<video_id>.npy for each video.
    """
    splits = ["train", "val", "test"]

    # Initialize MediaPipe Hands once and reuse
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as hands:
        for split in splits:
            split_root = os.path.join(WORDS_ROOT, split)
            if not os.path.isdir(split_root):
                print(f"[WARN] Split directory not found: {split_root}, skipping.")
                continue

            for word in sorted(os.listdir(split_root)):
                word_path = os.path.join(split_root, word)
                if not os.path.isdir(word_path):
                    continue

                for video_id in sorted(os.listdir(word_path)):
                    video_folder = os.path.join(word_path, video_id)
                    if not os.path.isdir(video_folder):
                        continue

                    rel_output_dir = os.path.join(KEYPOINTS_ROOT, split, word)
                    ensure_dir(rel_output_dir)
                    output_path = os.path.join(rel_output_dir, f"{video_id}.npy")

                    if os.path.isfile(output_path):
                        # Already processed
                        print(f"[INFO] Keypoints already exist: {output_path}, skipping.")
                        continue

                    try:
                        print(f"Processing {video_folder} -> {output_path}")
                        seq_array = process_video_folder(video_folder, hands)
                        np.save(output_path, seq_array)
                    except FileNotFoundError as e:
                        print(f"[WARN] {e}, skipping.")
                    except Exception as e:
                        print(f"[ERROR] Failed to process {video_folder}: {e}")


if __name__ == "__main__":
    main()

