"""
SETU — Physics-Based Multi-Signer Landmark Augmentation Engine.

Generates a robust 50,000+ sample landmark dataset by simulating varied human
hand physiologies (aspect ratios, finger lengths, palm widths, 3D rotations, and
sensor noise) to protect against real-world domain shifts and overfitting.

Addresses:
- PROB-04: Landmark 9 Canonical Coordinate Drift
- PROB-10: Lack of Diverse Anthropometric Signer Morphology
- PROB-03: Single-Hand vs Dual-Hand Training Distribution Mismatch (Hand Dropout)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np

from training.features.geometry import normalize_wrist_coordinates

SINGLE_HAND_CLASSES = {
    "A", "B", "C", "D", "E", "I", "L", "M", "N", "O", "U", "V", "W", "X", "Y"
}


def random_rotation_matrix_3d(max_angle_deg: float = 12.0) -> np.ndarray:
    """Generates a random 3D rotation matrix with angles bounded within +/- max_angle_deg."""
    angles = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    return Rz @ Ry @ Rx


def augment_hand_landmarks(
    hand_21x3: np.ndarray,
    scale_jitter: Tuple[float, float] = (0.88, 1.12),
    aspect_ratio_jitter: Tuple[float, float] = (0.90, 1.10),
    max_rot_deg: float = 12.0,
    noise_sigma: float = 0.005,
) -> np.ndarray:
    """
    Applies physiological morphology and 3D spatial perturbations to a single hand:
    - Segmental anthropometric finger lengths (PROB-10)
    - Palm aspect ratio variation (PROB-10)
    - 3D spatial camera rotation
    - Sensor noise simulation
    - Strict canonical coordinate projection (PROB-04)
    """
    if np.all(hand_21x3 == 0):
        return hand_21x3.copy()

    augmented = hand_21x3.copy()

    # 1. Segmental Anthropometric Scaling (PROB-10)
    # Scale individual finger chains relative to their MCP joint
    finger_chains = [
        (1, [2, 3, 4]),       # Thumb
        (5, [6, 7, 8]),       # Index
        (9, [10, 11, 12]),    # Middle
        (13, [14, 15, 16]),   # Ring
        (17, [18, 19, 20]),   # Pinky
    ]
    for mcp, phalanges in finger_chains:
        s_digit = float(np.random.uniform(0.88, 1.12))
        for p in phalanges:
            augmented[p] = augmented[mcp] + (augmented[p] - augmented[mcp]) * s_digit

    # 2. Palm aspect ratio (width vs height)
    w_palm = float(np.random.uniform(*aspect_ratio_jitter))
    augmented[:, 0] *= w_palm

    # 3. Overall scale perturbation
    scale = float(np.random.uniform(*scale_jitter))
    augmented *= scale

    # 4. 3D spatial rotation (camera angle / wrist tilt variation)
    R = random_rotation_matrix_3d(max_angle_deg=max_rot_deg)
    augmented = (R @ augmented.T).T

    # 5. Sensor noise / tremor simulation
    noise = np.random.normal(0.0, noise_sigma, size=augmented.shape).astype(np.float32)
    augmented += noise

    # 6. Strict Canonical Projection (PROB-04)
    # Re-normalize canonically so Landmark 0 is (0,0,0) and Landmark 9 is strictly (0, -1, 0)
    canonical_norm, scale_val = normalize_wrist_coordinates(augmented)
    if scale_val > 1e-6:
        return canonical_norm.astype(np.float32)

    augmented[0] = 0.0
    return augmented


def generate_multisigner_dataset(
    source_npz: str = "data/processed/isl_master_clean_126d.npz",
    output_npz: str = "data/processed/isl_multisigner_augmented_26class.npz",
    target_train_size: int = 40000,
    random_seed: int = 42,
) -> None:
    """
    Expands the clean training set into a multi-signer augmented training distribution
    with strict canonical projection, anthropometric variation, and single-hand dropout.
    """
    np.random.seed(random_seed)
    src_path = Path(source_npz)
    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {src_path.resolve()}")

    with np.load(src_path, allow_pickle=True) as data:
        X_orig = data["X"]
        y_orig = data["y"]
        splits_orig = data["splits"]
        classes = data["classes"]

    train_idx = np.where(splits_orig == "train")[0]
    val_idx = np.where(splits_orig == "val")[0]
    test_idx = np.where(splits_orig == "test")[0]

    X_train_orig = X_orig[train_idx]
    y_train_orig = y_orig[train_idx]

    num_orig_train = len(train_idx)
    multiplier = max(1, math.ceil(target_train_size / num_orig_train))

    print(f"Original train size: {num_orig_train}. Multiplier: {multiplier}x.")

    augmented_X_list = [X_train_orig]
    augmented_y_list = [y_train_orig]

    # Generate multi-signer variations
    for m in range(multiplier - 1):
        batch_X = np.zeros_like(X_train_orig)
        for i in range(num_orig_train):
            c_name = str(classes[y_train_orig[i]])
            lh = X_train_orig[i, :63].reshape(21, 3)
            rh = X_train_orig[i, 63:].reshape(21, 3)

            # Augment left and right hands with independent physiological variation
            lh_aug = augment_hand_landmarks(lh)
            rh_aug = augment_hand_landmarks(rh)

            # PROB-03: Single-Hand Dropout on resting hand
            if c_name in SINGLE_HAND_CLASSES:
                # 50% probability: signer drops left hand (slots 0..62 set to 0.0)
                if np.random.rand() < 0.50:
                    lh_aug = np.zeros((21, 3), dtype=np.float32)
                # 15% probability: signer drops right hand (left-handed signer mode)
                elif np.random.rand() < 0.15:
                    rh_aug = np.zeros((21, 3), dtype=np.float32)

            batch_X[i, :63] = lh_aug.flatten()
            batch_X[i, 63:] = rh_aug.flatten()

        augmented_X_list.append(batch_X)
        augmented_y_list.append(y_train_orig)

    X_train_full = np.vstack(augmented_X_list)[:target_train_size]
    y_train_full = np.concatenate(augmented_y_list)[:target_train_size]
    splits_train_full = np.array(["train"] * len(X_train_full), dtype=object)

    # Preserve genuine validation and test sets intact
    X_val = X_orig[val_idx]
    y_val = y_orig[val_idx]
    splits_val = np.array(["val"] * len(X_val), dtype=object)

    X_test = X_orig[test_idx]
    y_test = y_orig[test_idx]
    splits_test = np.array(["test"] * len(X_test), dtype=object)

    # Combine full dataset
    X_final = np.vstack([X_train_full, X_val, X_test]).astype(np.float32)
    y_final = np.concatenate([y_train_full, y_val, y_test]).astype(np.int64)
    splits_final = np.concatenate([splits_train_full, splits_val, splits_test])

    out_path = Path(output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X_final,
        y=y_final,
        splits=splits_final,
        classes=classes,
    )

    print(f"Generated multi-signer dataset saved to: {out_path.resolve()}")
    print(f"Total samples: {len(X_final):,} (Train: {len(X_train_full):,}, Val: {len(X_val):,}, Test: {len(X_test):,})")
    print(f"Class count: {len(classes)} classes.")


if __name__ == "__main__":
    generate_multisigner_dataset()
