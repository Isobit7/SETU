"""
SETU — Master Dynamic Dataset Builder
Merges Amrita SLR (49 phrases), ISL Sequence LSTM, MVP Greetings, Custom Signs, and Rest/No-Sign frames
into a single canonical 53-Class Dynamic Spatio-Temporal Dataset:
- 49 Amrita conversational phrases
- 'namaste' (custom recordings)
- 'i' (INCLUDE recordings)
- 'you' (INCLUDE recordings)
- 'no_sign' (negative / resting hand postures)

Standardizes all sequences to (T=20, Dim=316: 158 Holistic + 158 Velocity).
Outputs: data/processed/isl_dynamic_master_53class.npz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SETU.BuildDynamicMaster")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_NPZ = PROCESSED_DIR / "isl_dynamic_master_53class.npz"
LABEL_MAP_PATH = ROOT_DIR / "models" / "metadata" / "dynamic_master_label_map.json"


def resample_to_20x316(features: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
    """Converts any (T, 158/164/316) sequence into normalized (20, 316) with velocities."""
    T_in, D_in = features.shape

    # If already (20, 316), return directly
    if T_in == 20 and D_in == 316:
        return features.astype(np.float32)

    # Extract base 158 position coordinates
    pos = features[:, :158].copy()

    # Temporal resampling to 20 frames
    if T_in != 20:
        orig_t = np.linspace(0.0, 1.0, T_in)
        target_t = np.linspace(0.0, 1.0, 20)
        resampled_pos = np.zeros((20, 158), dtype=np.float32)
        for d in range(158):
            resampled_pos[:, d] = np.interp(target_t, orig_t, pos[:, d])
    else:
        resampled_pos = pos.astype(np.float32)

    # Compute frame-to-frame velocity
    vel = np.zeros_like(resampled_pos)
    vel[1:] = (resampled_pos[1:] - resampled_pos[:-1]) / dt
    vel[0] = vel[1] if len(vel) > 1 else 0.0
    vel[:, 146:148] = 0.0  # Zero presence flag derivatives

    out = np.concatenate([resampled_pos, vel], axis=-1).astype(np.float32)
    return out


def main() -> None:
    logger.info("Building SETU Master 53-Class Dynamic Dataset...")

    # 1. Load Base Amrita 49-class dataset
    amrita_path = PROCESSED_DIR / "amrita_slr_sequences.npz"
    if not amrita_path.exists():
        raise FileNotFoundError(f"Amrita base dataset not found at {amrita_path}")

    amrita_data = np.load(amrita_path, allow_pickle=True)
    amrita_X = amrita_data["X"]  # (978, 20, 316)
    amrita_y = amrita_data["y"]
    amrita_classes = [str(c) for c in amrita_data["classes"]]
    amrita_ids = [str(sid) for sid in amrita_data["sample_ids"]]

    logger.info("Loaded Amrita dataset: %d samples across %d classes.", len(amrita_X), len(amrita_classes))

    # Define master classes: 49 Amrita + 'namaste', 'i', 'you', 'no_sign'
    extra_classes = ["namaste", "i", "you", "no_sign"]
    master_classes = list(amrita_classes)
    for ec in extra_classes:
        if ec not in master_classes:
            master_classes.append(ec)

    label_to_idx = {name: idx for idx, name in enumerate(master_classes)}
    logger.info("Total Master Classes: %d (%s)", len(master_classes), master_classes)

    all_samples: List[Tuple[np.ndarray, int, str]] = []

    # Add existing Amrita samples (remap indices to master classes)
    for i in range(len(amrita_X)):
        class_name = amrita_classes[amrita_y[i]]
        new_idx = label_to_idx[class_name]
        all_samples.append((amrita_X[i], new_idx, amrita_ids[i]))

    # 2. Add custom namaste samples
    namaste_files = sorted(PROCESSED_DIR.glob("custom_namaste_*.npz"))
    for nf in namaste_files:
        with np.load(nf) as d:
            feat = d["features"]
            feat_316 = resample_to_20x316(feat)
            all_samples.append((feat_316, label_to_idx["namaste"], nf.stem))
    logger.info("Added %d 'namaste' samples.", len(namaste_files))

    # 3. Add INCLUDE samples ('i', 'you', and extra 'hello', 'thank_you', 'good_morning')
    include_files = sorted(PROCESSED_DIR.glob("include_*.npz"))
    inc_counts: Dict[str, int] = {}
    for inf in include_files:
        stem = inf.stem  # e.g., include_i_s10_0013, include_you_s13_0022
        # Parse label
        parts = stem.split("_")
        if len(parts) >= 2:
            if parts[1] == "good" and len(parts) >= 3 and parts[2] == "morning":
                lbl = "good_morning"
            elif parts[1] == "thank" and len(parts) >= 3 and parts[2] == "you":
                lbl = "thank_you"
            else:
                lbl = parts[1]

            if lbl in label_to_idx:
                with np.load(inf) as d:
                    feat = d["features"]
                    feat_316 = resample_to_20x316(feat)
                    all_samples.append((feat_316, label_to_idx[lbl], stem))
                    inc_counts[lbl] = inc_counts.get(lbl, 0) + 1

    logger.info("Added INCLUDE dynamic samples: %s", inc_counts)

    # 4. Add custom hello samples
    custom_hello_files = sorted(PROCESSED_DIR.glob("custom_hello_*.npz"))
    for ch in custom_hello_files:
        with np.load(ch) as d:
            feat = d["features"]
            feat_316 = resample_to_20x316(feat)
            all_samples.append((feat_316, label_to_idx["hello"], ch.stem))
    logger.info("Added %d extra custom 'hello' samples.", len(custom_hello_files))

    # 5. Add resting / idle 'no_sign' samples
    no_sign_files = sorted(PROCESSED_DIR.glob("no_sign_*.npz")) + sorted(PROCESSED_DIR.glob("real_no_sign_*.npz"))
    for nsf in no_sign_files:
        with np.load(nsf) as d:
            feat = d["features"]
            feat_316 = resample_to_20x316(feat)
            all_samples.append((feat_316, label_to_idx["no_sign"], nsf.stem))
    logger.info("Added %d 'no_sign' resting background samples.", len(no_sign_files))

    logger.info("Total Combined Samples: %d across %d classes.", len(all_samples), len(master_classes))

    # 6. Clean Per-Class Split: 80% Train, 10% Validation, 10% Test
    X_arr = np.stack([s[0] for s in all_samples], axis=0).astype(np.float32)
    y_arr = np.array([s[1] for s in all_samples], dtype=np.int64)
    ids_arr = np.array([s[2] for s in all_samples])

    splits = np.empty(len(y_arr), dtype="<U10")
    rng = np.random.RandomState(42)

    for c_idx in range(len(master_classes)):
        c_indices = np.where(y_arr == c_idx)[0]
        rng.shuffle(c_indices)
        n = len(c_indices)
        if n >= 10:
            n_test = max(1, int(round(n * 0.10)))
            n_val = max(1, int(round(n * 0.10)))
            n_train = n - n_val - n_test
            splits[c_indices[:n_train]] = "train"
            splits[c_indices[n_train:n_train + n_val]] = "val"
            splits[c_indices[n_train + n_val:]] = "test"
        elif n >= 3:
            splits[c_indices[0]] = "val"
            splits[c_indices[1]] = "test"
            splits[c_indices[2:]] = "train"
        else:
            splits[c_indices] = "train"

    logger.info("Split distribution: Train=%d, Val=%d, Test=%d", 
                np.sum(splits == "train"), np.sum(splits == "val"), np.sum(splits == "test"))

    # 7. Save Merged Dataset NPZ
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X_arr,
        y=y_arr,
        splits=splits,
        classes=np.array(master_classes),
        sample_ids=ids_arr,
    )
    logger.info("Master Dataset successfully saved to: %s", OUTPUT_NPZ.resolve())

    # 8. Save Unified Label Map
    LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    label_map = {
        "classes": master_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: name for name, idx in label_to_idx.items()},
        "num_classes": len(master_classes),
        "total_samples": len(all_samples),
        "input_dim": 316,
        "sequence_length": 20,
    }
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Unified Label Map saved to: %s", LABEL_MAP_PATH.resolve())


if __name__ == "__main__":
    main()
