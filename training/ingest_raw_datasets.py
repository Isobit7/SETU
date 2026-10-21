"""
SETU — End-to-End Raw Image Ingestion & Stratified Split Generation Pipeline

Recursively scans `data/raw/` for sign language image directories representing classes A–Z
(handling variations like 'A', 'letter_A', 'Class_A', 'A_1', etc.), extracts 63-D rotation-invariant
hand landmark features via MediaPipe (static_image_mode=True, max_num_hands=1) and
`setu.features.extractor.extract_realsign_hand_features`, merges with existing dataset archives,
generates a stratified 80% Train / 10% Validation / 10% Test split across y, and saves both archives:
  - `data/processed/realsign/realsign_landmarks_26class.npz`
  - `data/processed/realsign_landmarks_26class.npz`
retaining all four keys: 'X', 'y', 'splits', and 'classes'.
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setu.features.extractor import extract_realsign_hand_features

# Standard 26-class uppercase alphabet
CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]
CLASS_SET_26: Set[str] = set(CLASSES_26)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_26)}

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "realsign" / "realsign_landmarks_26class.npz"
DEFAULT_ALT_OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Global worker state for multiprocessing
_worker_hands: Optional[Any] = None


def init_extractor_worker(confidence: float) -> None:
    """Initializes a process-local MediaPipe Hands instance."""
    global _worker_hands
    _worker_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=confidence,
    )


def process_image_batch(
    batch: List[Tuple[str, str]],
) -> List[Tuple[np.ndarray, str, str]]:
    """
    Worker task: processes a batch of (image_path_str, class_name) tuples.
    Returns list of (feature_63d, class_name, image_path_str) for all detected hands.
    """
    global _worker_hands
    if _worker_hands is None:
        init_extractor_worker(0.4)

    results: List[Tuple[np.ndarray, str, str]] = []
    for path_str, class_name in batch:
        try:
            img = cv2.imread(path_str)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_res = _worker_hands.process(img_rgb)
            if mp_res and mp_res.multi_hand_landmarks and len(mp_res.multi_hand_landmarks) > 0:
                # Wrap hand landmarks and handedness for invariant extraction
                handedness_list = (
                    [[h.classification[0]] for h in mp_res.multi_handedness]
                    if mp_res.multi_handedness
                    else []
                )
                hand_target = {
                    "hand_landmarks": mp_res.multi_hand_landmarks,
                    "handedness": handedness_list,
                }
                feat_63 = extract_realsign_hand_features(hand_target)
                if feat_63 is not None and feat_63.shape == (63,) and np.all(np.isfinite(feat_63)):
                    results.append((feat_63, class_name, path_str))
        except Exception:
            continue

    return results


def match_class_from_dir_name(dir_name: str) -> Optional[str]:
    """Matches directory names representing classes A-Z."""
    s = dir_name.strip()
    if s.upper() in CLASS_SET_26:
        return s.upper()
    # Matches patterns like 'letter_A', 'class_A', 'alphabet_A', 'sign_A', 'letter-A', 'Class A'
    m = re.match(
        r"^(?:letter|class|alphabet|sign)[_\-\s]+([a-zA-Z])(?:[_\-\s\d].*)?$",
        s,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in CLASS_SET_26:
        return m.group(1).upper()
    # Matches patterns like 'A_1', 'A-1', 'A 1', 'A_sample', 'A_dataset'
    m2 = re.match(
        r"^([a-zA-Z])[_\-\s]+(?:\d+|sample|img|images|raw|data|part).*$",
        s,
        re.IGNORECASE,
    )
    if m2 and m2.group(1).upper() in CLASS_SET_26:
        return m2.group(1).upper()
    return None


def match_class_from_filename(filename: str) -> Optional[str]:
    """Matches image filenames representing classes A-Z (e.g. 'A.jpg', 'letter_A.png')."""
    stem = Path(filename).stem.strip()
    if stem.upper() in CLASS_SET_26:
        return stem.upper()
    m = re.match(
        r"^(?:letter|class|alphabet|sign)[_\-\s]+([a-zA-Z])(?:[_\-\s\d].*)?$",
        stem,
        re.IGNORECASE,
    )
    if m and m.group(1).upper() in CLASS_SET_26:
        return m.group(1).upper()
    return None


def scan_raw_image_datasets(
    raw_dir: Path,
    exclude_substrings: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """
    Recursively scans `raw_dir` for all image files belonging to classes A-Z.
    Returns {class_letter: [Path, Path, ...]}.
    """
    scanned: Dict[str, List[Path]] = {c: [] for c in CLASSES_26}
    exclude_patterns = [s.lower() for s in (exclude_substrings or [])]

    if not raw_dir.exists():
        return scanned

    for root, dirs, files in os.walk(raw_dir):
        p = Path(root)
        posix_norm = p.as_posix().lower()

        # Check exclusion filter
        if any(exc in posix_norm for exc in exclude_patterns):
            continue

        folder_cls = match_class_from_dir_name(p.name)
        img_files = [
            p / f for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not img_files:
            continue

        for img_path in img_files:
            cls = folder_cls or match_class_from_filename(img_path.name)
            if cls and cls in CLASS_SET_26:
                scanned[cls].append(img_path)

    return scanned


def extract_features_parallel(
    items: List[Tuple[Path, str]],
    confidence: float = 0.4,
    workers: int = 8,
    chunk_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts 63-D features from (image_path, class_name) list in parallel.
    Returns (X, y, detected_paths).
    """
    total = len(items)
    if total == 0:
        return np.empty((0, 63), dtype=np.float32), np.empty(0, dtype=np.int64), []

    str_items = [(str(p), c) for p, c in items]
    chunks = [str_items[i: i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"Dispatching {total:,} images in {len(chunks):,} batches across {workers} worker processes...")
    start_t = time.time()
    extracted_features: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []
    processed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_extractor_worker,
        initargs=(confidence,),
    ) as executor:
        for batch_results in executor.map(process_image_batch, chunks):
            for feat, cls_name, p_str in batch_results:
                extracted_features.append(feat)
                labels.append(CLASS_TO_IDX[cls_name])
                paths.append(p_str)

            processed += len(batch_results)
            # Log periodic progress
            elapsed = max(time.time() - start_t, 0.001)
            rate = len(extracted_features) / elapsed
            if len(extracted_features) % 2500 < len(batch_results) or len(extracted_features) == total:
                print(
                    f"  [Progress] Extracted {len(extracted_features):,} valid hand vectors "
                    f"({rate:.1f} vec/s) — Elapsed: {elapsed:.1f}s"
                )

    total_time = time.time() - start_t
    print(
        f"Feature extraction finished in {total_time:.1f}s. "
        f"Detected {len(extracted_features):,} valid hand landmarks out of {total:,} images "
        f"({len(extracted_features) / total * 100:.1f}% success rate)."
    )

    X = np.array(extracted_features, dtype=np.float32) if extracted_features else np.empty((0, 63), dtype=np.float32)
    y = np.array(labels, dtype=np.int64) if labels else np.empty(0, dtype=np.int64)
    return X, y, paths


def merge_and_deduplicate(
    X_existing: Optional[np.ndarray],
    y_existing: Optional[np.ndarray],
    X_new: np.ndarray,
    y_new: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merges new feature vectors with existing dataset vectors and deduplicates
    exact matches to prevent over-representation.
    """
    if X_existing is None or len(X_existing) == 0:
        X_all = X_new
        y_all = y_new
    elif len(X_new) == 0:
        X_all = X_existing
        y_all = y_existing
    else:
        print(f"Merging {len(X_new):,} new vectors with {len(X_existing):,} existing vectors...")
        X_all = np.vstack([X_existing, X_new])
        y_all = np.concatenate([y_existing, y_new])

    if len(X_all) == 0:
        return X_all, y_all

    # Deduplicate rounded to 5 decimals to catch identical coordinate extractions
    print(f"Deduplicating {len(X_all):,} combined vectors...")
    X_rounded = np.round(X_all, decimals=5)
    _, unique_idx = np.unique(X_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    X_dedup = X_all[unique_idx]
    y_dedup = y_all[unique_idx]
    print(f"Retained {len(X_dedup):,} unique vectors (dropped {len(X_all) - len(X_dedup):,} duplicate/redundant samples).")
    return X_dedup, y_dedup


def generate_stratified_splits(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Auto-generates a stratified 80% Train / 10% Validation / 10% Test split across y."""
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


def save_realsign_archive(
    save_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    classes: Sequence[str] = CLASSES_26,
    sync_to_alt: bool = True,
    alt_path: Optional[Path] = None,
) -> None:
    """
    Saves the dataset archive retaining X, y, splits, and classes keys,
    and synchronizes to the companion dataset path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    splits_str = np.array(splits, dtype=str)
    classes_str = np.array(classes, dtype=str)

    np.savez_compressed(
        save_path,
        X=X,
        y=y,
        splits=splits_str,
        classes=classes_str,
    )
    mb = save_path.stat().st_size / (1024 * 1024)
    print(f"Saved dataset archive to: {save_path} ({mb:.2f} MB, {len(X):,} samples)")

    if sync_to_alt:
        target_alt = alt_path or (
            DEFAULT_ALT_OUTPUT_NPZ
            if save_path.resolve() == DEFAULT_OUTPUT_NPZ.resolve()
            else DEFAULT_OUTPUT_NPZ
        )
        try:
            target_alt.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_alt,
                X=X,
                y=y,
                splits=splits_str,
                classes=classes_str,
            )
            alt_mb = target_alt.stat().st_size / (1024 * 1024)
            print(f"Synchronized companion dataset archive to: {target_alt} ({alt_mb:.2f} MB)")
        except Exception as e:
            print(f"Warning: Could not sync to companion archive {target_alt}: {e}")


def ingest_and_split_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT_NPZ,
    alt_output_path: Path = DEFAULT_ALT_OUTPUT_NPZ,
    workers: int = 8,
    confidence: float = 0.4,
    chunk_size: int = 100,
    seed: int = 42,
    include_existing_raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the end-to-end data ingestion and split generation pipeline.
    Returns (X, y, splits, classes).
    """
    print("=" * 75)
    print("SETU — RealSign 26-Class Raw Image Ingestion & Split Pipeline")
    print("=" * 75)

    # 1. Load existing vectors if available
    X_existing: Optional[np.ndarray] = None
    y_existing: Optional[np.ndarray] = None
    existing_source = output_path if output_path.exists() else (alt_output_path if alt_output_path.exists() else None)

    if existing_source:
        print(f"Loading existing processed dataset from: {existing_source}")
        with np.load(existing_source, allow_pickle=True) as data:
            if "X" in data and "y" in data:
                X_existing = data["X"]
                y_existing = data["y"]
                print(f"  Existing dataset contains: {len(X_existing):,} vectors across {len(np.unique(y_existing))} classes.")

    # 2. Scan raw images
    # If existing realsign dataset is already present, exclude 'realsign/extracted'
    # by default to avoid redundant re-extraction of the 26k images already in X_existing.
    exclude_dirs: List[str] = []
    if X_existing is not None and not include_existing_raw:
        exclude_dirs = [
            "realsign/extracted/training",
            "realsign/extracted/validation",
            "realsign/extracted/testing",
        ]
        print("Note: Excluding previously extracted 'data/raw/realsign/extracted/' to avoid redundant re-processing.")

    print(f"Scanning raw images in: {raw_dir}...")
    scanned_classes = scan_raw_image_datasets(raw_dir, exclude_substrings=exclude_dirs)
    
    total_scanned_imgs = sum(len(v) for v in scanned_classes.values())
    print(f"Found {total_scanned_imgs:,} raw candidate images across {len(scanned_classes)} classes.")

    # Flatten into list of (path, class_name)
    items_to_extract: List[Tuple[Path, str]] = []
    for cls_name in CLASSES_26:
        for p in scanned_classes.get(cls_name, []):
            items_to_extract.append((p, cls_name))

    # 3. Extract features in parallel
    X_new, y_new, _ = extract_features_parallel(
        items=items_to_extract,
        confidence=confidence,
        workers=workers,
        chunk_size=chunk_size,
    )

    # 4. Merge and deduplicate
    X_merged, y_merged = merge_and_deduplicate(
        X_existing=X_existing,
        y_existing=y_existing,
        X_new=X_new,
        y_new=y_new,
    )

    if len(X_merged) == 0:
        raise RuntimeError("No feature vectors available after extraction and merge!")

    # 5. Generate stratified 80/10/10 split
    print(f"Generating stratified 80% Train / 10% Val / 10% Test split for {len(y_merged):,} samples...")
    splits = generate_stratified_splits(y_merged, seed=seed)

    splits_clean = np.array(splits, dtype=str)
    train_count = np.sum(splits_clean == "train")
    val_count = np.sum(splits_clean == "val")
    test_count = np.sum(splits_clean == "test")

    print(f"  Train: {train_count:,} ({train_count / len(splits) * 100:.1f}%)")
    print(f"  Val:   {val_count:,} ({val_count / len(splits) * 100:.1f}%)")
    print(f"  Test:  {test_count:,} ({test_count / len(splits) * 100:.1f}%)")

    # 6. Save both datasets retaining all four keys
    classes_arr = np.array(CLASSES_26, dtype=str)
    save_realsign_archive(
        save_path=output_path,
        X=X_merged,
        y=y_merged,
        splits=splits,
        classes=CLASSES_26,
        sync_to_alt=True,
        alt_path=alt_output_path,
    )

    print("=" * 75)
    print("Ingestion and Split Generation Complete!")
    print("=" * 75)
    return X_merged, y_merged, splits, classes_arr


def main():
    parser = argparse.ArgumentParser(description="SETU RealSign Raw Dataset Ingestion & Split Pipeline")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Path to raw image directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_NPZ, help="Path to output NPZ")
    parser.add_argument("--alt-output", type=Path, default=DEFAULT_ALT_OUTPUT_NPZ, help="Path to companion NPZ")
    parser.add_argument("--workers", type=int, default=min(12, max(2, (os.cpu_count() or 4) - 2)), help="Worker processes")
    parser.add_argument("--confidence", type=float, default=0.4, help="Min detection confidence")
    parser.add_argument("--chunk-size", type=int, default=100, help="Images per worker task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits")
    parser.add_argument("--force-all-raw", action="store_true", help="Force re-extraction of all raw images")
    args = parser.parse_args()

    ingest_and_split_pipeline(
        raw_dir=args.raw_dir,
        output_path=args.output,
        alt_output_path=args.alt_output,
        workers=args.workers,
        confidence=args.confidence,
        chunk_size=args.chunk_size,
        seed=args.seed,
        include_existing_raw=args.force_all_raw,
    )


if __name__ == "__main__":
    main()
