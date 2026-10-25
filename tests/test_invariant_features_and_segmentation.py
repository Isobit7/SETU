"""
Unit Tests for Invariant Hand Feature Extraction, Time-Normalized Velocity,
and Kinetic Energy Gesture Segmentation.
"""

import numpy as np
import pytest

from training.feature_extractor import (
    normalize_wrist_coordinates,
    compute_joint_angles_15,
    compute_fingertip_to_wrist_distances,
    compute_fingertip_to_thumb_distances,
    compute_pairwise_distances,
    extract_invariant_hand_features,
)
from training.feature_extractor import (
    FeatureExtractor158,
    compute_velocity_features,
    extract_realsign_hand_features,
)


def _create_test_hand(seed: int = 42) -> np.ndarray:
    """Creates a realistic synthetic hand landmark array (21, 3)."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.6, 0.0)
    pts[0] = [0.5, 0.6, 0.0]

    # Thumb: 1, 2, 3, 4
    for i, idx in enumerate([1, 2, 3, 4]):
        pts[idx] = pts[0] + np.array([-0.02 * (i + 1), -0.015 * (i + 1), 0.005 * i], dtype=np.float32)

    # Fingers: Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
    bases = [(-0.015, -0.04), (0.0, -0.05), (0.015, -0.045), (0.025, -0.035)]
    start_indices = [5, 9, 13, 17]

    for (bx, by), start_idx in zip(bases, start_indices):
        pts[start_idx] = pts[0] + np.array([bx, by, 0.0], dtype=np.float32)
        for step in range(1, 4):
            pts[start_idx + step] = pts[start_idx] + np.array([bx * 0.3 * step, by * 0.4 * step, 0.0], dtype=np.float32)

    return pts


# ============================================================================
# 1. Fully Invariant Hand Normalization Tests
# ============================================================================

def test_normalize_wrist_coordinates_canonical_alignment():
    """Wrist must be at origin and Middle MCP must align strictly with (0, -1, 0)."""
    hand = _create_test_hand()
    norm_pts, scale = normalize_wrist_coordinates(hand)

    assert norm_pts.shape == (21, 3)
    assert scale > 0.0
    # Landmark 0 (wrist) at (0, 0, 0)
    assert np.allclose(norm_pts[0], [0.0, 0.0, 0.0], atol=1e-6)
    # Landmark 9 (middle MCP) strictly at (0, -1, 0)
    assert np.allclose(norm_pts[9], [0.0, -1.0, 0.0], atol=1e-5)
    # Distance from wrist to middle MCP is exactly 1.0
    assert np.isclose(np.linalg.norm(norm_pts[9] - norm_pts[0]), 1.0, atol=1e-5)


def test_normalize_wrist_coordinates_translation_invariance():
    """Translating hand in 3D must produce identical normalized coordinates."""
    hand = _create_test_hand()
    norm_base, scale_base = normalize_wrist_coordinates(hand)

    offsets = [
        np.array([10.5, -20.3, 5.0], dtype=np.float32),
        np.array([-100.0, 50.0, -25.0], dtype=np.float32),
        np.array([0.001, -0.002, 0.003], dtype=np.float32),
    ]
    for offset in offsets:
        translated_hand = hand + offset
        norm_trans, scale_trans = normalize_wrist_coordinates(translated_hand)
        assert np.isclose(scale_trans, scale_base, atol=1e-4)
        assert np.allclose(norm_trans, norm_base, atol=2e-4)


def test_normalize_wrist_coordinates_scale_invariance():
    """Scaling hand uniformly must produce identical normalized coordinates."""
    hand = _create_test_hand()
    norm_base, scale_base = normalize_wrist_coordinates(hand)

    scale_factors = [0.25, 0.5, 1.8, 3.5, 10.0]
    for sf in scale_factors:
        scaled_hand = hand * sf
        norm_scaled, scale_scaled = normalize_wrist_coordinates(scaled_hand)
        assert np.isclose(scale_scaled, scale_base * sf, atol=1e-4)
        assert np.allclose(norm_scaled, norm_base, atol=1e-5)


def test_normalize_wrist_coordinates_inplane_rotation_invariance():
    """Rotating hand in-plane (Z-axis) must yield identical canonical coordinates."""
    hand = _create_test_hand()
    norm_base, _ = normalize_wrist_coordinates(hand)

    angles = [np.pi / 6, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3, -2 * np.pi / 3]
    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)

        # Rotate around hand's wrist
        wrist = hand[0]
        rotated_hand = (hand - wrist) @ R_z.T + wrist
        norm_rot, _ = normalize_wrist_coordinates(rotated_hand)

        assert np.allclose(norm_rot[0], [0.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(norm_rot[9], [0.0, -1.0, 0.0], atol=1e-5)
        assert np.allclose(norm_rot, norm_base, atol=1e-4)


def test_normalize_wrist_coordinates_edge_cases():
    """Handles None, empty, zero palm length gracefully."""
    zeros, scale = normalize_wrist_coordinates(None)
    assert zeros.shape == (21, 3) and scale == 0.0
    assert np.all(zeros == 0.0)

    # Hand with zero palm length (wrist == middle MCP)
    hand_zero = np.zeros((21, 3), dtype=np.float32)
    zeros, scale = normalize_wrist_coordinates(hand_zero)
    assert scale == 0.0
    assert np.all(zeros == 0.0)


# ============================================================================
# 2. Joint Angles & Pairwise Distances Tests
# ============================================================================

def test_compute_joint_angles_15():
    """Computes 15 angles in [0, 1] that are invariant to rigid transformations."""
    hand = _create_test_hand()
    angles_base = compute_joint_angles_15(hand)

    assert angles_base.shape == (15,)
    assert angles_base.dtype == np.float32
    assert np.all(angles_base >= 0.0) and np.all(angles_base <= 1.0)

    # Invariant to translation
    translated = hand + np.array([5.0, -3.0, 2.0], dtype=np.float32)
    angles_trans = compute_joint_angles_15(translated)
    assert np.allclose(angles_trans, angles_base, atol=5e-4)

    # Invariant to uniform scale
    scaled = hand * 2.5
    angles_scaled = compute_joint_angles_15(scaled)
    assert np.allclose(angles_scaled, angles_base, atol=1e-5)

    # Invariant to rotation
    theta = 0.75
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    rotated = hand @ R.T
    angles_rot = compute_joint_angles_15(rotated)
    assert np.allclose(angles_rot, angles_base, atol=1e-5)


def test_pairwise_distances():
    """Verifies 5 fingertip-to-wrist and 4 fingertip-to-thumb distances."""
    hand = _create_test_hand()
    norm_hand, _ = normalize_wrist_coordinates(hand)

    d_wrist = compute_fingertip_to_wrist_distances(norm_hand)
    assert d_wrist.shape == (5,)
    assert np.all(d_wrist > 0.0)

    d_thumb = compute_fingertip_to_thumb_distances(norm_hand, include_self=False)
    assert d_thumb.shape == (4,)
    assert np.all(d_thumb > 0.0)

    d_combined = compute_pairwise_distances(norm_hand)
    assert d_combined.shape == (9,)
    assert np.allclose(d_combined[:5], d_wrist)
    assert np.allclose(d_combined[5:], d_thumb)


def test_extract_invariant_hand_features_87d():
    """extract_invariant_hand_features produces full 87-D representation."""
    hand = _create_test_hand()
    feat_87 = extract_invariant_hand_features(hand)

    assert feat_87.shape == (87,)
    assert feat_87.dtype == np.float32
    # 63 coords + 15 angles + 5 wrist dists + 4 thumb dists = 87
    coords_63 = feat_87[:63]
    angles_15 = feat_87[63:78]
    wrist_5 = feat_87[78:83]
    thumb_4 = feat_87[83:87]

    assert np.allclose(coords_63[:3], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.all(angles_15 >= 0.0) and np.all(angles_15 <= 1.0)
    assert np.all(wrist_5 > 0.0)
    assert np.all(thumb_4 > 0.0)

    # Invariant to rotation + translation + scaling
    theta = 1.2
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    transformed = (hand * 1.8) @ R.T + np.array([12.0, -8.0, 1.0], dtype=np.float32)
    feat_trans = extract_invariant_hand_features(transformed)
    assert np.allclose(feat_trans, feat_87, atol=5e-4)


# ============================================================================
# 3. Time-Normalized Velocity Computation Tests
# ============================================================================

def test_time_normalized_velocity_computation():
    """Velocity must scale inversely with delta t: V_t = (F_t - F_{t-1}) / max(dt, 0.001)."""
    T = 10
    seq = np.zeros((T, 164), dtype=np.float32)
    # Add steady landmark movement of 0.1 per frame
    for t in range(T):
        seq[t, :10] = t * 0.1

    # 1. Constant dt = 0.033 (30 FPS nominal)
    vel_30fps = compute_velocity_features(seq, target_dim=328, dt=0.033)
    # 2. Constant dt = 0.066 (15 FPS nominal) -> 2x longer step -> velocity per sec should be half
    vel_15fps = compute_velocity_features(seq, target_dim=328, dt=0.066)

    # Expected velocity for frame 1..T-1: delta F / dt = 0.1 / dt
    expected_v_30 = 0.1 / 0.033
    expected_v_15 = 0.1 / 0.066

    assert np.isclose(vel_30fps[1, 164], expected_v_30, atol=1e-4)
    assert np.isclose(vel_15fps[1, 164], expected_v_15, atol=1e-4)
    assert np.isclose(vel_30fps[1, 164] / vel_15fps[1, 164], 2.0, atol=1e-4)


def test_time_normalized_velocity_timestamps_array():
    """Accepts explicit non-uniform timestamps array and converts ms to seconds if needed."""
    T = 5
    seq = np.zeros((T, 164), dtype=np.float32)
    for t in range(T):
        seq[t, 0] = t * 0.05

    # Variable timestamps in seconds: [0.0, 0.02, 0.05, 0.10, 0.12]
    timestamps = [0.0, 0.02, 0.05, 0.10, 0.12]
    vel = compute_velocity_features(seq, target_dim=328, timestamps=timestamps)

    # For step 1 (0 -> 1): dt = 0.02, delta = 0.05 -> vel = 0.05 / 0.02 = 2.5
    assert np.isclose(vel[1, 164], 0.05 / 0.02, atol=1e-4)
    # For step 2 (1 -> 2): dt = 0.03, delta = 0.05 -> vel = 0.05 / 0.03 = 1.6667
    assert np.isclose(vel[2, 164], 0.05 / 0.03, atol=1e-4)


def test_time_normalized_velocity_safe_zero_dt():
    """Handles sub-millisecond or zero dt safely using max(dt, 0.001)."""
    seq = np.ones((5, 164), dtype=np.float32)
    seq[1, 0] = 2.0  # delta = 1.0

    # dt = 0.0 or 1e-6 -> clamped to 0.001 -> vel = 1.0 / 0.001 = 1000.0
    vel = compute_velocity_features(seq, target_dim=328, dt=0.0)
    assert np.isclose(vel[1, 164], 1000.0, atol=1e-3)
    assert not np.any(np.isnan(vel))
    assert not np.any(np.isinf(vel))


# ============================================================================
# 4. RealSign Dedicated Feature Invariance
# ============================================================================

def test_extract_realsign_hand_features_invariant():
    """extract_realsign_hand_features yields scale and rotation invariant 63-D features."""
    hand = _create_test_hand()
    feat_base = extract_realsign_hand_features(hand)

    assert feat_base is not None
    assert feat_base.shape == (63,)
    assert np.allclose(feat_base[:3], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feat_base[27:30], [0.0, -1.0, 0.0], atol=1e-5)

    # Rotate hand by 45 degrees
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    rotated = hand @ R.T
    feat_rot = extract_realsign_hand_features(rotated)
    assert np.allclose(feat_rot, feat_base, atol=1e-4)
