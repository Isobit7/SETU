"""
SETU — Ingestion & Dual-Hand 126-D Extraction for Hugging Face ISL Dataset
Dataset: akritRihal/Indian_Sign_Language_dataset
Reads native parquet images, detects MediaPipe 3D hand landmarks,
and produces stratified 126-D vectors for model training.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.features.geometry import extract_dual_hand_126d

HF_DIR = PROJECT_ROOT / "data" / "raw" / "huggingface_isl"
OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "hf_isl_dualhand_26class.npz"

# HF Label Mapping (from dataset metadata)
HF_LABEL_MAP: Dict[int, str] = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "I",
    18: "K", 19: "L", 20: "M", 21: "N", 22: "O", 23: "P", 24: "Q", 25: "R",
    26: "S", 27: "T", 28: "U", 29: "W", 30: "X", 31: "Y", 32: "Z"
}

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES_26)}


def process_image_bytes(img_bytes: bytes) -> List[np.ndarray]:
    """Decodes image bytes and extracts 126-D dual-hand feature vectors."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as detector:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

    if not res or not res.multi_hand_landmarks:
        return []

    pts_list = []
    for lms in res.multi_hand_landmarks:
        pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms.landmark], dtype=np.float32)
        if pts.shape == (21, 3) and np.all(np.isfinite(pts)):
            pts_list.append(pts)

    if not pts_list:
        return []

    vectors = []
    if len(pts_list) >= 2:
        # Sort by horizontal coordinate: left screen vs right screen
        h1, h2 = pts_list[0], pts_list[1]
        lh = h1 if np.mean(h1[:, 0]) < np.mean(h2[:, 0]) else h2
        rh = h2 if np.mean(h1[:, 0]) < np.mean(h2[:, 0]) else h1

        v1 = extract_dual_hand_126d(lh, rh)
        if v1.shape == (126,) and np.all(np.isfinite(v1)):
            vectors.append(v1)

        v_swap = extract_dual_hand_126d(rh, lh)
        if v_swap.shape == (126,) and np.all(np.isfinite(v_swap)):
            vectors.append(v_swap)
    else:
        pts = pts_list[0]
        v_rh = extract_dual_hand_126d(left_hand=None, right_hand=pts)
        if v_rh.shape == (126,) and np.all(np.isfinite(v_rh)):
            vectors.append(v_rh)
        v_lh = extract_dual_hand_126d(left_hand=pts, right_hand=None)
        if v_lh.shape == (126,) and np.all(np.isfinite(v_lh)):
            vectors.append(v_lh)

    return vectors


def ingest_hf_dataset(max_samples_per_class: int = 200) -> None:
    """Ingests parquet files from Hugging Face and extracts 126-D features."""
    train_pq = HF_DIR / "train-00000-of-00001.parquet"
    test_pq = HF_DIR / "test-00000-of-00001.parquet"

    if not train_pq.exists() or not test_pq.exists():
        print(f"Error: Parquet files not found in {HF_DIR}")
        return

    print("Loading Hugging Face parquet dataframes...")
    t0 = time.time()
    df_test = pd.read_parquet(test_pq)
    df_train = pd.read_parquet(train_pq)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Loaded {len(df_all)} total examples in {time.time() - t0:.2f}s")

    # Map integer labels to letter names
    df_all["char"] = df_all["label"].map(HF_LABEL_MAP)
    # Filter to 26 alphabetic classes (A-Z)
    df_alphabet = df_all[df_all["char"].isin(CLASS_TO_IDX.keys())].copy()
    print(f"Filtered to {len(df_alphabet)} alphabet examples across {df_alphabet['char'].nunique()} classes")

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for char_name, group in df_alphabet.groupby("char"):
        class_idx = CLASS_TO_IDX[char_name]
        selected = group.head(max_samples_per_class)
        print(f"Processing class {char_name} ({len(selected)} samples)...", end="", flush=True)
        extracted = 0

        for _, row in selected.iterrows():
            img_dict = row["image"]
            b = img_dict.get("bytes") if isinstance(img_dict, dict) else None
            if not b:
                continue
            vecs = process_image_bytes(b)
            for v in vecs:
                features_list.append(v)
                labels_list.append(class_idx)
                extracted += 1

        print(f" extracted {extracted} vectors")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    print(f"\nTotal extracted vectors: {X.shape[0]}, Feature dimension: {X.shape[1]}")
    if len(X) == 0:
        print("No features extracted.")
        return

    # Stratified Train/Val/Test Split (80/10/10)
    indices = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.20, random_state=42, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    splits = np.empty(len(y), dtype=object)
    splits[idx_train] = "train"
    splits[idx_val] = "val"
    splits[idx_test] = "test"

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        splits=splits,
        classes=np.array(CLASSES_26),
    )
    print(f"Successfully saved {len(X)} dual-hand 126-D samples to {OUTPUT_NPZ}")
    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")


if __name__ == "__main__":
    ingest_hf_dataset(max_samples_per_class=120)
