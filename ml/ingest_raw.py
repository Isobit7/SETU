"""
SETU — End-to-End Raw Image Ingestion & Stratified Split Generation Pipeline

Recursively scans `data/raw/` for sign language image directories representing classes A–Z
(handling variations like 'A', 'letter_A', 'Class_A', 'A_1', etc.), extracts 63-D rotation-invariant
hand landmark features via MediaPipe (static_image_mode=True, max_num_hands=1) and
`ml.feature_extractor.extract_realsign_hand_features`, merges with existing dataset archives,
generates a stratified 80% Train / 10% Validation / 10% Test split across y, and saves to:
    `data/processed/realsign_landmarks_26class.npz`
retaining all four essential keys: 'X', 'y', 'splits', and 'classes'.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_extractor import extract_realsign_hand_features

# Standard 26-class uppercase alphabet
CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]
CLASS_SET_26: Set[str] = set(CLASSES_26)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_26)}

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_worker_hands: Optional[Any] = None


def init_extractor_worker(confidence: float) -> None:
    """Initializes a process-local MediaPipe Hands instance."""
    global _worker_hands
    _worker_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=confidence,
    )


def extract_features_from_image(image_path_str: str) -> Optional[np.ndarray]:
    """Worker function to process a single image file."""
    global _worker_hands
    if _worker_hands is None:
        _worker_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

    try:
        bgr = cv2.imread(image_path_str)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = _worker_hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lms = res.multi_hand_landmarks[0]
        feats = extract_realsign_hand_features(lms, dominant_hand="right")
        if feats is not None and feats.shape == (63,) and np.all(np.isfinite(feats)):
            return feats.astype(np.float32)
    except Exception:
        return None
    return None


def resolve_class_from_dir(dir_name: str) -> Optional[str]:
    """Extracts alphabet class letter ('A' through 'Z') from directory name."""
    s = dir_name.strip()
    if s.upper() in CLASS_SET_26:
        return s.upper()

    match = re.match(r"^(?:letter|class|sign|char)[_\-\s]*([A-Za-z])$", s, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in CLASS_SET_26:
            return letter

    match2 = re.match(r"^([A-Za-z])[_\-\s]+\d+$", s)
    if match2:
        letter = match2.group(1).upper()
        if letter in CLASS_SET_26:
            return letter

    subdirs = re.split(r"[_\-\s]+", s)
    for part in subdirs:
        if part.upper() in CLASS_SET_26:
            return part.upper()

    return None


def discover_raw_image_tasks(raw_root: Path) -> Dict[str, List[Path]]:
    """Discovers image files grouped by resolved alphabet class."""
    class_images: Dict[str, List[Path]] = {c: [] for c in CLASSES_26}
    if not raw_root.exists():
        return class_images

    for root, dirs, files in os.walk(raw_root):
        current_dir = Path(root)
        resolved_class = resolve_class_from_dir(current_dir.name)
        if resolved_class is None and current_dir.parent != raw_root:
            resolved_class = resolve_class_from_dir(current_dir.parent.name)

        if resolved_class in CLASS_SET_26:
            for f in files:
                p = current_dir / f
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    class_images[resolved_class].append(p)

    return class_images


def load_existing_dataset(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads existing dataset vectors and metadata."""
    if not path.exists():
        return None, None, None, None

    try:
        with np.load(path, allow_pickle=True) as data:
            X = np.array(data["X"], dtype=np.float32) if "X" in data else None
            y = np.array(data["y"]) if "y" in data else None
            splits = np.array(data["splits"], dtype=str) if "splits" in data else None
            classes = np.array(data["classes"], dtype=str) if "classes" in data else None
            return X, y, splits, classes
    except Exception as e:
        print(f"[-] Warning: Failed to read existing dataset at {path}: {e}")
        return None, None, None, None


def generate_stratified_splits(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Creates a stratified 80% Train / 10% Validation / 10% Test split across y."""
    n = len(y)
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed
    )

    splits = np.empty(n, dtype="<U10")
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    return splits


def run_ingestion_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_NPZ,
    num_workers: Optional[int] = None,
    min_confidence: float = 0.5,
    max_per_class: Optional[int] = None,
) -> bool:
    """Executes end-to-end dataset ingestion and split creation."""
    print("=" * 78)
    print("SETU — Raw Image Dataset Ingestion & Feature Extraction Pipeline")
    print(f"Raw Input Directory : {raw_dir.resolve()}")
    print(f"Output Dataset Path : {output_path.resolve()}")
    print("=" * 78)

    class_images = discover_raw_image_tasks(raw_dir)
    total_discovered = sum(len(v) for v in class_images.values())
    print(f"[+] Discovered {total_discovered:,} total raw images across {len(class_images)} classes.")

    new_X: List[np.ndarray] = []
    new_y: List[int] = []

    if total_discovered > 0:
        tasks = []
        for cls_name, paths in class_images.items():
            selected = paths[:max_per_class] if max_per_class else paths
            cls_idx = CLASS_TO_IDX[cls_name]
            for p in selected:
                tasks.append((str(p), cls_idx))

        workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
        print(f"[+] Extracting 63-D landmarks using {workers} parallel processes...")

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_extractor_worker,
            initargs=(min_confidence,),
        ) as executor:
            futures = [executor.submit(extract_features_from_image, t[0]) for t in tasks]
            for i, fut in enumerate(futures):
                feat = fut.result()
                if feat is not None:
                    new_X.append(feat)
                    new_y.append(tasks[i][1])

    # Load existing dataset if available
    old_X, old_y, _, _ = load_existing_dataset(output_path)

    all_X = []
    all_y = []

    if old_X is not None and old_y is not None and len(old_X) > 0:
        print(f"[+] Merging with {len(old_X):,} existing vectors from {output_path.name}...")
        all_X.append(old_X)
        all_y.append(old_y)

    if new_X:
        all_X.append(np.array(new_X, dtype=np.float32))
        all_y.append(np.array(new_y, dtype=np.int64))

    if not all_X:
        print("[-] Error: No feature vectors available to save.")
        return False

    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)

    # Convert string labels if needed
    if final_y.dtype.kind in ("U", "S", "O"):
        numeric_y = []
        for val in final_y:
            s_val = str(val).strip().upper()
            numeric_y.append(CLASS_TO_IDX.get(s_val, 0))
        final_y = np.array(numeric_y, dtype=np.int64)

    # Generate stratified splits
    splits = generate_stratified_splits(final_y)
    classes = np.array(CLASSES_26, dtype=str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=final_X,
        y=final_y,
        splits=splits,
        classes=classes,
    )
    print(f"[+] Dataset successfully saved to: {output_path} ({len(final_X):,} total samples)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw image datasets into standardized .npz format")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    run_ingestion_pipeline(raw_dir=args.raw_dir, output_path=args.output, num_workers=args.workers)
