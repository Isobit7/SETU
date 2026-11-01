"""
SETU — Physics-Based Multi-Signer Landmark Augmentation Engine.

Generates a robust 50,000+ sample landmark dataset by simulating varied human
hand physiologies (aspect ratios, finger lengths, palm widths, 3D rotations, and
sensor noise) to protect against real-world domain shifts and overfitting.
"""

import math
from pathlib import Path
from typing import Tuple
import numpy as np


def random_rotation_matrix_3d(max_angle_deg: float = 15.0) -> np.ndarray:
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
    max_rot_deg: float = 15.0,
    noise_sigma: float = 0.008,
) -> np.ndarray:
    """Applies physiological morphology and 3D spatial perturbations to a single hand."""
    if np.all(hand_21x3 == 0):
        return hand_21x3.copy()

    augmented = hand_21x3.copy()

    # 1. Morphological Aspect Ratio (simulate longer/shorter fingers vs wider/narrower palms)
    sx = np.random.uniform(*aspect_ratio_jitter)
    sy = np.random.uniform(*aspect_ratio_jitter)
    sz = np.random.uniform(0.95, 1.05)
    augmented[:, 0] *= sx
    augmented[:, 1] *= sy
    augmented[:, 2] *= sz

    # 2. Overall scale perturbation (hand size variation across signers)
    scale = np.random.uniform(*scale_jitter)
    augmented *= scale

    # 3. 3D spatial rotation (camera angle / wrist tilt variation)
    R = random_rotation_matrix_3d(max_angle_deg=max_rot_deg)
    augmented = (R @ augmented.T).T

    # 4. Sensor noise / tremor simulation
    noise = np.random.normal(0.0, noise_sigma, size=augmented.shape).astype(np.float32)
    augmented += noise

    # Maintain wrist anchor at (0, 0, 0)
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
    while strictly preserving the authentic held-out test and validation sets.
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
            lh = X_train_orig[i, :63].reshape(21, 3)
            rh = X_train_orig[i, 63:].reshape(21, 3)

            # Augment left and right hands with correlated but independent physiological variation
            lh_aug = augment_hand_landmarks(lh)
            rh_aug = augment_hand_landmarks(rh)

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

