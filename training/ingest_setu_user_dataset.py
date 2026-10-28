"""
SETU — Comprehensive Multi-Source Dataset Ingestion & Feature Extraction Pipeline

Extracts 63-D normalized hand landmark features from:
1. User-recorded JPEGs and MP4 video clips in `data/raw/setu_custom/SETU/`
2. Real-world signer photo archive in `data/raw/NEW ZIP/archive/` (especially for Y, Z, and full real-world coverage)

Applies:
- MediaPipe Hands with max_num_hands=2 for bimanual ISL sign detection
- Pure wrist-centered, scale- and rotation-invariant 63-D coordinate normalization
- Sagittal reflection data augmentation (both right- and left-handed execution invariance)
- Merges into `data/processed/realsign_landmarks_26class.npz` with stratified 80/10/10 splits
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.features.geometry import normalize_landmarks_to_63d

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_26)}

CUSTOM_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "setu_custom" / "SETU"
ARCHIVE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NEW ZIP" / "archive"
EXISTING_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"
ALT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "realsign" / "realsign_landmarks_26class.npz"


def extract_features_from_mediapipe_result(res: Any) -> List[np.ndarray]:
    """
    Extracts 63-D normalized hand vectors from a MediaPipe Hands result.
    Produces both normal and sagittal-reflected vectors for left/right invariance.
    If 2 hands are detected, extracts both hands with reflection augmentation.
    """
    if not res or not res.multi_hand_landmarks:
        return []

    features: List[np.ndarray] = []
    num_hands = len(res.multi_hand_landmarks)

    for i in range(num_hands):
        lms = res.multi_hand_landmarks[i]
        pts = np.array(
            [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms.landmark],
            dtype=np.float32,
        )
        if pts.shape != (21, 3) or not np.all(np.isfinite(pts)):
            continue

        # 1. Direct normalized vector
        vec_standard = normalize_landmarks_to_63d(pts, is_left_hand=False)
        if vec_standard is not None and vec_standard.shape == (63,) and np.all(np.isfinite(vec_standard)):
            features.append(vec_standard)

        # 2. Sagittal-reflected vector (mirror across wrist for left/right invariance)
        vec_mirrored = normalize_landmarks_to_63d(pts, is_left_hand=True)
        if vec_mirrored is not None and vec_mirrored.shape == (63,) and np.all(np.isfinite(vec_mirrored)):
            features.append(vec_mirrored)

    return features


def extract_from_image(
    image_path: Path,
    mp_hands: mp.solutions.hands.Hands,
) -> List[np.ndarray]:
    """Extracts 63-D landmark vectors (including reflections) from an image file."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(img_rgb)
    return extract_features_from_mediapipe_result(res)


def extract_from_video(
    video_path: Path,
    mp_hands: mp.solutions.hands.Hands,
    num_samples: int = 15,
) -> List[np.ndarray]:
    """Samples frames evenly across a video clip and extracts 63-D features."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    start_frame = max(0, int(total_frames * 0.10))
    end_frame = min(total_frames - 1, int(total_frames * 0.90))
    if end_frame <= start_frame:
        sample_indices = [total_frames // 2]
    else:
        sample_indices = np.linspace(start_frame, end_frame, min(num_samples, end_frame - start_frame + 1), dtype=int)

    features: List[np.ndarray] = []
    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(frame_rgb)
        feats = extract_features_from_mediapipe_result(res)
        features.extend(feats)

    cap.release()
    return features


def extract_all_sources(
    confidence: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Extracts features from both SETU custom user files and archive photos.
    Returns (X, y, per_class_counts).
    """
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=confidence,
    )

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    per_class_counts: Dict[str, int] = {c: 0 for c in CLASSES_26}

    t0 = time.time()
    print("[1/4] Ingesting SETU custom user recordings and archive photos...")

    for letter in CLASSES_26:
        cls_idx = CLASS_TO_IDX[letter]
        class_features: List[np.ndarray] = []

        # 1. SETU custom directory
        custom_class_dir = CUSTOM_RAW_DIR / letter
        if custom_class_dir.exists() and custom_class_dir.is_dir():
            images = [p for p in custom_class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            videos = [p for p in custom_class_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov")]

            for img_p in images:
                feats = extract_from_image(img_p, mp_hands)
                class_features.extend(feats)

            for vid_p in videos:
                feats = extract_from_video(vid_p, mp_hands, num_samples=15)
                class_features.extend(feats)

        # 2. Archive photos (high-res real photos, crucial for Y, Z, and full grounding)
        archive_class_dir = ARCHIVE_RAW_DIR / letter
        if archive_class_dir.exists() and archive_class_dir.is_dir():
            archive_images = [p for p in archive_class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            # Ingest all archive photos for Y and Z; sample up to 35 for other letters
            max_archive = len(archive_images) if letter in ("Y", "Z") else min(35, len(archive_images))
            for img_p in archive_images[:max_archive]:
                feats = extract_from_image(img_p, mp_hands)
                class_features.extend(feats)

        count = len(class_features)
        per_class_counts[letter] = count

        for f in class_features:
            all_features.append(f)
            all_labels.append(cls_idx)

        print(f"  Class '{letter}': {count:,} augmented vectors extracted")

    mp_hands.close()
    elapsed = time.time() - t0
    total = len(all_features)
    print(f"\nTotal real-world vectors extracted: {total:,} in {elapsed:.1f}s across 26 classes.")

    if total == 0:
        raise RuntimeError("No features extracted from any data source!")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    return X, y, per_class_counts


def merge_with_base_dataset(
    X_new: np.ndarray,
    y_new: np.ndarray,
    base_npz_path: Path = EXISTING_NPZ_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merges newly extracted features with existing dataset archive and deduplicates."""
    print(f"\n[2/4] Merging with base dataset from: {base_npz_path}")
    if not base_npz_path.exists():
        print(f"  Notice: Base dataset not found at {base_npz_path}. Using new features directly.")
        return X_new, y_new

    with np.load(base_npz_path, allow_pickle=True) as data:
        X_base = data["X"]
        y_base = data["y"]

    print(f"  Base dataset: {len(X_base):,} vectors")
    print(f"  New real-world dataset: {len(X_new):,} vectors")

    X_merged = np.vstack([X_base, X_new])
    y_merged = np.concatenate([y_base, y_new])
    print(f"  Combined total: {len(X_merged):,} vectors")

    # Deduplicate rounded to 5 decimal places
    print("  Deduplicating identical vectors...")
    X_rounded = np.round(X_merged, decimals=5)
    _, unique_idx = np.unique(X_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    X_dedup = X_merged[unique_idx]
    y_dedup = y_merged[unique_idx]
    print(f"  Retained {len(X_dedup):,} unique vectors ({len(X_merged) - len(X_dedup):,} duplicates removed)")

    return X_dedup, y_dedup


def generate_splits(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generates stratified 80% Train / 10% Validation / 10% Test splits."""
    print(f"\n[3/4] Generating stratified 80/10/10 splits across {len(y):,} samples...")
    n = len(y)
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed)

    splits = np.empty(n, dtype="<U10")
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"

    print(f"  Train: {len(train_idx):,} ({len(train_idx)/n*100:.1f}%)")
    print(f"  Val:   {len(val_idx):,} ({len(val_idx)/n*100:.1f}%)")
    print(f"  Test:  {len(test_idx):,} ({len(test_idx)/n*100:.1f}%)")
    return splits


def save_merged_archives(
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    classes: List[str] = CLASSES_26,
) -> None:
    """Saves the merged dataset to both primary and companion NPZ paths."""
    print("\n[4/4] Saving updated dataset archives...")
    for p in [EXISTING_NPZ_PATH, ALT_NPZ_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            X=X,
            y=y,
            splits=splits,
            classes=np.array(classes, dtype=str),
        )
        mb = p.stat().st_size / (1024 * 1024)
        print(f"  Saved: {p} ({mb:.2f} MB)")


def main():
    X_new, y_new, counts = extract_all_sources(confidence=0.30)
    X_merged, y_merged = merge_with_base_dataset(X_new, y_new)
    splits = generate_splits(y_merged)
    save_merged_archives(X_merged, y_merged, splits)
    print("\n[SUCCESS] Comprehensive dataset ingestion & merge completed successfully!")


if __name__ == "__main__":
    main()
