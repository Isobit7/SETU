"""
SETU — High-Throughput ISL 42K Bimanual Landmark Extraction & Dataset Ingestion

Extracts 126-D dual-hand landmark features from:
1. Authentic Indian Sign Language Dataset in `data/raw/NEW ZIP/Indian Sign Language_Dataset/` (A-Z)
2. Real-world signer archive in `data/raw/NEW ZIP/archive/` (A-Z)
3. Single-hand RealSign dataset in `data/raw/realsign/` (A-Z)

Features:
- Dual-hand 126-D representation:
  Left hand (63-D) in slots [0:63]
  Right hand (63-D) in slots [63:126]
- True bimanual ISL signs (A, B, D, E, G, K, M, N, P, T, X, Y, Z) capture both hands.
- Single-hand ISL signs (C, H, I, J, L, O, Q, R, S, U, V, W) populate active hand slot.
- Generates stratified Train/Val/Test splits (80/10/10).
- Produces `data/processed/isl_dualhand_26class.npz`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.features.geometry import normalize_wrist_coordinates, extract_dual_hand_126d

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_26)}

ISL_42K_DIR = PROJECT_ROOT / "data" / "raw" / "NEW ZIP" / "Indian Sign Language_Dataset"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "raw" / "NEW ZIP" / "archive"
REALSIGN_DIR = PROJECT_ROOT / "data" / "raw" / "realsign"
OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "isl_dualhand_26class.npz"


def process_single_image(img_path: Path) -> List[np.ndarray]:
    """
    Extracts 126-D dual-hand feature vector from an image using MediaPipe Hands.
    Returns standard and augmented vectors.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    # Local Hands instance for thread safety
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands_detector:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(rgb)

    if not res or not res.multi_hand_landmarks:
        return []

    num_hands = len(res.multi_hand_landmarks)
    pts_list = []
    labels_list = []

    for i, lms in enumerate(res.multi_hand_landmarks):
        pts = np.array(
            [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms.landmark],
            dtype=np.float32,
        )
        if pts.shape == (21, 3) and np.all(np.isfinite(pts)):
            lbl = "right"
            if res.multi_handedness and i < len(res.multi_handedness):
                h_meta = res.multi_handedness[i]
                c_name = h_meta.classification[0].label.lower() if h_meta.classification else "right"
                lbl = c_name
            pts_list.append(pts)
            labels_list.append(lbl)

    if not pts_list:
        return []

    vectors: List[np.ndarray] = []

    if len(pts_list) >= 2:
        # Bimanual sign: Sort by X-coordinate to reliably assign left hand vs right hand in frame
        # Left side of screen (viewer's left = smaller X) and right side of screen (larger X)
        h1, h2 = pts_list[0], pts_list[1]
        mean_x1 = np.mean(h1[:, 0])
        mean_x2 = np.mean(h2[:, 0])
        lh = h1 if mean_x1 < mean_x2 else h2
        rh = h2 if mean_x1 < mean_x2 else h1

        # 1. Standard dual-hand 126-D
        vec_126 = extract_dual_hand_126d(lh, rh)
        if vec_126.shape == (126,) and np.all(np.isfinite(vec_126)):
            vectors.append(vec_126)

        # 2. Hand-swap / mirror augmentation
        vec_swap = extract_dual_hand_126d(rh, lh)
        if vec_swap.shape == (126,) and np.all(np.isfinite(vec_swap)):
            vectors.append(vec_swap)

    elif len(pts_list) == 1:
        # Single-hand sign: Put in right hand slot [63:126], and also generate left hand slot [0:63]
        pts = pts_list[0]
        # Primary in right hand
        vec_rh = extract_dual_hand_126d(left_hand=None, right_hand=pts)
        if vec_rh.shape == (126,) and np.all(np.isfinite(vec_rh)):
            vectors.append(vec_rh)

        # Reflected / in left hand
        vec_lh = extract_dual_hand_126d(left_hand=pts, right_hand=None)
        if vec_lh.shape == (126,) and np.all(np.isfinite(vec_lh)):
            vectors.append(vec_lh)

    return vectors


def ingest_all(
    samples_per_class: int = 150,
    include_archive: bool = True,
    max_workers: int = 8,
) -> None:
    print("=" * 70)
    print("SETU — ISL 42K Dual-Hand Dataset Ingestion Pipeline")
    print(f"Target: {samples_per_class} samples/class from 42K dataset")
    print("=" * 70)

    all_X: List[np.ndarray] = []
    all_y: List[int] = []

    t0 = time.time()

    for letter in CLASSES_26:
        cls_idx = CLASS_TO_IDX[letter]
        class_dir = ISL_42K_DIR / letter
        if not class_dir.exists():
            print(f"Warning: Directory not found: {class_dir}")
            continue

        all_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        total_files = len(all_files)
        if total_files == 0:
            print(f"Letter {letter}: No images found.")
            continue

        # Sample evenly across the folder
        step = max(1, total_files // samples_per_class)
        sampled_files = all_files[::step][:samples_per_class]

        # Archive files
        if include_archive:
            arch_dir = ARCHIVE_DIR / letter
            if arch_dir.exists():
                arch_files = sorted(list(arch_dir.glob("*.jpg")) + list(arch_dir.glob("*.png")))
                sampled_files.extend(arch_files[:40])

        # Extract in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_single_image, sampled_files))

        class_vecs = []
        for res_list in batch_results:
            class_vecs.extend(res_list)

        for vec in class_vecs:
            all_X.append(vec)
            all_y.append(cls_idx)

        print(f"Letter {letter}: Processed {len(sampled_files)} images -> {len(class_vecs)} 126-D vectors")

    if not all_X:
        print("Error: No features extracted!")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    print("-" * 70)
    print(f"Total extracted ISL samples: {len(X):,} vectors of dimension {X.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Generate stratified 80/10/10 splits
    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=42)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=42)

    splits = np.empty(len(y), dtype="<U10")
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        splits=splits,
        classes=np.array(CLASSES_26),
    )
    print(f"Successfully saved ISL dataset to: {OUTPUT_NPZ}")
    print(f"Completed in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    ingest_all(samples_per_class=150, max_workers=8)
