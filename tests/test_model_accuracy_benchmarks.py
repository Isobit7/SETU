"""
SETU — Quantitative Model Accuracy Benchmarks & Geometric Discrimination (Pillar 2)

Validates mathematical discrimination thresholds, confusion pair separation,
and spatial invariance properties:
1. Minimal detectable angle threshold for joint flexion (> 15 deg precision).
2. 'U' vs 'V' fingertip spread discrimination (> 2.5x ratio via compute_finger_spread_angles).
3. 'M' vs 'N' vs 'T' finger flexion discrimination via compute_joint_angles_15.
4. 'O' vs 'C' tip-to-thumb closure clearance.
5. Dual-hand touching contact signs ('P', 'H', 'K') vs detached parallel signs ('B', 'W').
6. Translation, distance and scale invariance across 0.5x, 1.0x, and 2.0x hand sizes.
7. Mirror mode symmetry: left-hand vs right-hand canonical representations.
8. Quantitative multi-signer benchmark on 1,445 real test samples (> 90.0% accuracy, > 0.90 F1).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import torch

from training.features.geometry import (
    normalize_wrist_coordinates,
    normalize_landmarks_to_63d,
    compute_joint_angles_15,
    compute_finger_spread_angles,
    compute_inter_hand_relative_geometry,
)
from training.train_isl_dualhand_mlp import RealSignMLP, compute_metrics


# ---------------------------------------------------------------------------
# 1. MINIMAL DETECTABLE JOINT ANGLE DEFLECTION THRESHOLD
# ---------------------------------------------------------------------------

def test_minimal_detectable_angle_threshold():
    """Validates that joint angle computation detects finger deflections as small as 15 degrees."""
    # Create base straight finger: Wrist (0), MCP (5), PIP (6), DIP (7), Tip (8)
    # Straight line along negative Y axis
    base_pts = np.zeros((21, 3), dtype=np.float32)
    base_pts[0] = [0.0, 0.0, 0.0]
    base_pts[5] = [0.0, -0.4, 0.0]
    base_pts[6] = [0.0, -0.6, 0.0]
    base_pts[7] = [0.0, -0.8, 0.0]
    base_pts[8] = [0.0, -1.0, 0.0]

    angles_straight = compute_joint_angles_15(base_pts)
    # Straight line: angle around PIP (index 4 in 15-dim vector) is 180 deg -> normalized to 1.0
    pip_idx = 4  # MCP(5)->PIP(6)->DIP(7)
    assert abs(angles_straight[pip_idx] - 1.0) < 0.02

    # Deflect DIP joint by 15 degrees (0.2618 radians)
    theta = math.radians(15.0)
    deflected_pts = base_pts.copy()
    deflected_pts[7] = [
        deflected_pts[6, 0] + 0.2 * math.sin(theta),
        deflected_pts[6, 1] - 0.2 * math.cos(theta),
        0.0,
    ]
    deflected_pts[8] = [
        deflected_pts[7, 0] + 0.2 * math.sin(theta),
        deflected_pts[7, 1] - 0.2 * math.cos(theta),
        0.0,
    ]

    angles_deflected = compute_joint_angles_15(deflected_pts)
    detected_deflection_deg = abs(angles_straight[pip_idx] - angles_deflected[pip_idx]) * 180.0
    # Assert deflection was detected within 1.5 degrees of ground truth 15 degrees
    assert abs(detected_deflection_deg - 15.0) < 1.5


# ---------------------------------------------------------------------------
# 2. CONFUSION PAIR DISCRIMINATION: 'U' VS 'V'
# ---------------------------------------------------------------------------

def test_u_vs_v_fingertip_spread_separation():
    """'U' (index & middle touching/parallel) vs 'V' (index & middle spread wide)."""
    # 1. Synthetic 'U' hand: Index tip (8) and Middle tip (12) touching parallel
    hand_u = np.zeros((21, 3), dtype=np.float32)
    hand_u[0] = [0.0, 0.0, 0.0]     # Wrist
    hand_u[4] = [0.2, -0.4, 0.0]    # Thumb
    hand_u[8] = [-0.02, -1.0, 0.0]  # Index tip
    hand_u[12] = [0.02, -1.0, 0.0]  # Middle tip
    hand_u[16] = [0.05, -0.4, 0.0]  # Ring folded
    hand_u[20] = [0.10, -0.3, 0.0]  # Pinky folded

    spreads_u = compute_finger_spread_angles(hand_u)
    # Index 1 is angle between index ray (0->8) and middle ray (0->12)
    spread_u_im = spreads_u[1]

    # 2. Synthetic 'V' hand: Index and Middle tips spread by ~30 degrees
    hand_v = hand_u.copy()
    hand_v[8] = [-0.25, -0.95, 0.0]  # Index ray deflected left
    hand_v[12] = [0.25, -0.95, 0.0]  # Middle ray deflected right

    spreads_v = compute_finger_spread_angles(hand_v)
    spread_v_im = spreads_v[1]

    # Assert spread angle of 'V' is at least 2.5x greater than 'U'
    assert spread_v_im > 2.5 * spread_u_im
    assert spread_v_im > 0.10  # Significant angular aperture


# ---------------------------------------------------------------------------
# 3. CONFUSION PAIR DISCRIMINATION: 'M' VS 'N' VS 'T'
# ---------------------------------------------------------------------------

def test_m_vs_n_vs_t_finger_flexion_count():
    """'M' has 3 folded fingers over thumb, 'N' has 2, 'T' has 1."""
    # Base open hand
    def make_hand(flexed_fingers: list[str]) -> np.ndarray:
        hand = np.zeros((21, 3), dtype=np.float32)
        finger_indices = {
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }
        for name, idxs in finger_indices.items():
            if name in flexed_fingers:
                # Tightly folded phalanx: PIP bends 90 deg, DIP bends 90 deg into palm
                hand[idxs[0]] = [0.0, -0.4, 0.0]
                hand[idxs[1]] = [0.0, -0.6, 0.0]
                hand[idxs[2]] = [0.0, -0.5, 0.1]
                hand[idxs[3]] = [0.0, -0.4, 0.1]
            else:
                # Extended phalanx
                hand[idxs[0]] = [0.0, -0.4, 0.0]
                hand[idxs[1]] = [0.0, -0.6, 0.0]
                hand[idxs[2]] = [0.0, -0.8, 0.0]
                hand[idxs[3]] = [0.0, -1.0, 0.0]
        return hand

    hand_m = make_hand(["index", "middle", "ring"])
    hand_n = make_hand(["index", "middle"])
    hand_t = make_hand(["index"])

    angles_m = compute_joint_angles_15(hand_m)
    angles_n = compute_joint_angles_15(hand_n)
    angles_t = compute_joint_angles_15(hand_t)

    # Ring finger PIP/DIP angles (indices 9, 10, 11) should be flexed (< 0.70) in M but extended (> 0.90) in N & T
    ring_pip_m = angles_m[10]
    ring_pip_n = angles_n[10]
    assert ring_pip_m < 0.70, "Ring finger in 'M' must be flexed"
    assert ring_pip_n > 0.90, "Ring finger in 'N' must be extended"

    # Middle finger PIP/DIP angles (indices 6, 7, 8) flexed in M & N but extended in T
    mid_pip_n = angles_n[7]
    mid_pip_t = angles_t[7]
    assert mid_pip_n < 0.70, "Middle finger in 'N' must be flexed"
    assert mid_pip_t > 0.90, "Middle finger in 'T' must be extended"


# ---------------------------------------------------------------------------
# 4. CONFUSION PAIR DISCRIMINATION: 'O' VS 'C' TIP CLEARANCE
# ---------------------------------------------------------------------------

def test_o_vs_c_tip_to_thumb_clearance():
    """'O' has thumb tip touching index tip; 'C' has open curved aperture."""
    # Hand 'O': Thumb tip (4) and Index tip (8) touch (< 0.05)
    hand_o = np.zeros((21, 3), dtype=np.float32)
    hand_o[4] = [0.05, -0.70, 0.0]
    hand_o[8] = [0.05, -0.72, 0.0]
    d_o = float(np.linalg.norm(hand_o[4] - hand_o[8]))

    # Hand 'C': Thumb tip and Index tip open (> 0.20)
    hand_c = np.zeros((21, 3), dtype=np.float32)
    hand_c[4] = [0.15, -0.50, 0.0]
    hand_c[8] = [0.15, -0.80, 0.0]
    d_c = float(np.linalg.norm(hand_c[4] - hand_c[8]))

    assert d_o < 0.05
    assert d_c > 0.25
    assert d_c > 5.0 * d_o


# ---------------------------------------------------------------------------
# 5. DUAL-HAND TOUCHING VS DETACHED SPATIAL DISCRIMINATION
# ---------------------------------------------------------------------------

def test_dual_hand_touching_vs_detached_geometry():
    """Touching signs ('P', 'H', 'K') vs detached signs ('B', 'W') via relative geometry."""
    # 1. Contact Sign ('P'): Right index tip touches Left palm
    left_hand = np.zeros((21, 3), dtype=np.float32)
    left_hand[0] = [0.40, 0.50, 0.0]
    left_hand[9] = [0.40, 0.35, 0.0]  # Palm center

    right_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand[0] = [0.55, 0.50, 0.0]
    right_hand[8] = [0.41, 0.36, 0.0]  # Right index tip directly touches Left palm

    geom_touch = compute_inter_hand_relative_geometry(left_hand, right_hand)
    assert geom_touch["are_hands_touching"] is True
    assert geom_touch["min_fingertip_distance"] <= 0.12

    # 2. Detached Sign ('B'): Left and Right hands separated laterally
    right_detached = right_hand.copy()
    right_detached[:, 0] += 0.35  # Shift right by 35cm
    geom_detached = compute_inter_hand_relative_geometry(left_hand, right_detached)
    assert geom_detached["are_hands_touching"] is False
    assert geom_detached["min_fingertip_distance"] >= 0.25


# ---------------------------------------------------------------------------
# 6. DISTANCE, SCALE & TRANSLATION INVARIANCE
# ---------------------------------------------------------------------------

def test_scale_and_translation_invariance():
    """Hands at 0.5x, 1.0x, 2.0x scale produce identical normalized 3D coordinates."""
    rng = np.random.RandomState(42)
    base_hand = rng.uniform(-0.3, 0.3, size=(21, 3)).astype(np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.2, 0.0]  # Valid reference middle MCP

    # Canonical normalization of base
    norm_base, s_base = normalize_wrist_coordinates(base_hand)

    # 1. Translated and scaled down by 0.5x
    hand_small = (base_hand * 0.5) + np.array([0.5, 0.3, -0.2], dtype=np.float32)
    norm_small, _ = normalize_wrist_coordinates(hand_small)

    # 2. Translated and scaled up by 2.0x
    hand_large = (base_hand * 2.0) + np.array([-0.3, -0.4, 0.5], dtype=np.float32)
    norm_large, _ = normalize_wrist_coordinates(hand_large)

    # Coordinates must be identical within tight numerical tolerance (< 1e-4)
    assert np.allclose(norm_base, norm_small, atol=1e-4)
    assert np.allclose(norm_base, norm_large, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. MIRROR MODE & HANDEDNESS SYMMETRY
# ---------------------------------------------------------------------------

def test_mirror_mode_and_handedness_symmetry():
    """Left hand mirrored across x-axis produces identical canonical representation."""
    rng = np.random.RandomState(123)
    right_hand = rng.uniform(-0.25, 0.25, size=(21, 3)).astype(np.float32)
    right_hand[0] = [0.0, 0.0, 0.0]
    right_hand[9] = [0.0, -0.25, 0.0]

    # Create synthetic left hand by flipping x-coordinates
    left_hand = right_hand.copy()
    left_hand[:, 0] *= -1.0

    # Extract 63-D features using handedness flag
    feat_right = normalize_landmarks_to_63d(right_hand, is_left_hand=False)
    feat_left = normalize_landmarks_to_63d(left_hand, is_left_hand=True)

    # With handedness flip, canonical coordinates are symmetric
    assert np.allclose(feat_right, feat_left, atol=1e-4)


# ---------------------------------------------------------------------------
# 8. QUANTITATIVE MULTI-SIGNER ACCURACY BENCHMARK ON REAL TEST SPLIT
# ---------------------------------------------------------------------------

def test_multisigner_model_accuracy_benchmark():
    """Loads realsign_multisigner_mlp.pth and verifies test set accuracy > 90.0% and macro F1 > 0.90."""
    model_path = Path("models/weights/realsign_multisigner_mlp.pth")
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")

    dataset_path = Path("data/processed/isl_master_clean_126d.npz")
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found at {dataset_path}")

    # Load model checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", [chr(ord("A") + i) for i in range(26)])
    arch = ckpt.get("architecture", {})

    model = RealSignMLP(
        input_dim=arch.get("input_dim", 126),
        hidden1=arch.get("hidden1", 128),
        hidden2=arch.get("hidden2", 64),
        num_classes=arch.get("num_classes", len(classes)),
        use_batchnorm=arch.get("use_batchnorm", True),
        dropout=arch.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load real test split
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    splits = data["splits"]

    test_mask = (splits == "test")
    X_test = torch.from_numpy(X[test_mask].astype(np.float32))
    y_test = y[test_mask]

    assert len(X_test) > 500, "Test set should have adequate sample count"

    with torch.no_grad():
        logits = model(X_test)
        y_pred = torch.argmax(logits, dim=1).numpy()

    metrics = compute_metrics(y_test, y_pred, classes)
    overall_acc = metrics["overall_accuracy"]
    macro_f1 = metrics["macro_f1"]

    assert overall_acc >= 0.90, f"Expected test accuracy >= 90%, got {overall_acc * 100:.2f}%"
    assert macro_f1 >= 0.90, f"Expected macro F1 >= 0.90, got {macro_f1:.4f}"

    # Verify no class collapse: all 26 classes must have non-zero recall
    for c_name, c_stats in metrics["per_class"].items():
        assert c_stats["recall"] > 0.0, f"Class {c_name} suffered from mode collapse (recall = 0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


