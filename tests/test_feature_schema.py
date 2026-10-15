"""
SETU - Comprehensive Feature Schema & Extractor Unit Tests
Tests every slice, dimension boundary, normalization invariant, and handedness condition for the 158-feature representation.
"""

import pytest
import numpy as np

from setu.features.schema import (
    SCHEMA_VERSION,
    FEATURE_DIM,
    RIGHT_HAND_COORDS_SLICE,
    RIGHT_HAND_DISTANCES_SLICE,
    RIGHT_HAND_ANGLES_SLICE,
    RIGHT_HAND_GEOM_SLICE,
    RIGHT_HAND_TOTAL_SLICE,
    LEFT_HAND_COORDS_SLICE,
    LEFT_HAND_DISTANCES_SLICE,
    LEFT_HAND_ANGLES_SLICE,
    LEFT_HAND_GEOM_SLICE,
    LEFT_HAND_TOTAL_SLICE,
    PRESENCE_FLAGS_SLICE,
    RIGHT_PRESENCE_INDEX,
    LEFT_PRESENCE_INDEX,
    INTER_HAND_TOTAL_SLICE,
    INTER_WRIST_VECTOR_SLICE,
    INTER_WRIST_DISTANCE_INDEX,
    INTER_INDEX_VECTOR_SLICE,
    INTER_INDEX_DISTANCE_INDEX,
    INTER_THUMB_DISTANCE_INDEX,
    CONTACT_SCORE_INDEX,
    validate_schema_vector
)
from setu.features.extractor import FeatureExtractor158
from setu.utils.geometry import (
    compute_joint_angle,
    normalize_hand_landmarks,
    compute_contact_score
)

def create_synthetic_hand(wrist=(0.5, 0.5, 0.0), scale=0.1):
    """Generates synthetic 21 3D landmarks for a hand with standard geometry."""
    hand = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    hand[0] = [wx, wy, wz] # Wrist
    
    # Thumb (1-4)
    for i in range(1, 5):
        hand[i] = [wx - i * 0.02 * (scale / 0.1), wy - i * 0.01 * (scale / 0.1), wz]
        
    # Index (5-8)
    for i, idx in enumerate(range(5, 9), 1):
        hand[idx] = [wx - 0.01 * (scale / 0.1), wy - i * 0.03 * (scale / 0.1), wz]
        
    # Middle (9-12)
    for i, idx in enumerate(range(9, 13), 1):
        hand[idx] = [wx, wy - i * 0.035 * (scale / 0.1), wz]
        
    # Ring (13-16)
    for i, idx in enumerate(range(13, 17), 1):
        hand[idx] = [wx + 0.01 * (scale / 0.1), wy - i * 0.03 * (scale / 0.1), wz]
        
    # Pinky (17-20)
    for i, idx in enumerate(range(17, 21), 1):
        hand[idx] = [wx + 0.02 * (scale / 0.1), wy - i * 0.025 * (scale / 0.1), wz]
        
    return hand

def test_schema_version_and_immutability():
    """Verifies that schema version is explicitly defined as 1.0.0."""
    assert SCHEMA_VERSION == "1.0.0"

def test_contiguous_partition_coverage():
    """
    Exhaustive regression test: verifies that the 158 indices are partitioned
    into contiguous, non-overlapping segments with exactly 158 covered indices.
    """
    covered = set()
    slices_and_indices = [
        ("RH_COORDS", range(RIGHT_HAND_COORDS_SLICE.start, RIGHT_HAND_COORDS_SLICE.stop)),
        ("RH_DIST", range(RIGHT_HAND_DISTANCES_SLICE.start, RIGHT_HAND_DISTANCES_SLICE.stop)),
        ("RH_ANGLES", range(RIGHT_HAND_ANGLES_SLICE.start, RIGHT_HAND_ANGLES_SLICE.stop)),
        ("LH_COORDS", range(LEFT_HAND_COORDS_SLICE.start, LEFT_HAND_COORDS_SLICE.stop)),
        ("LH_DIST", range(LEFT_HAND_DISTANCES_SLICE.start, LEFT_HAND_DISTANCES_SLICE.stop)),
        ("LH_ANGLES", range(LEFT_HAND_ANGLES_SLICE.start, LEFT_HAND_ANGLES_SLICE.stop)),
        ("RH_PRESENCE", [RIGHT_PRESENCE_INDEX]),
        ("LH_PRESENCE", [LEFT_PRESENCE_INDEX]),
        ("INTER_WRIST_VEC", range(INTER_WRIST_VECTOR_SLICE.start, INTER_WRIST_VECTOR_SLICE.stop)),
        ("INTER_WRIST_DIST", [INTER_WRIST_DISTANCE_INDEX]),
        ("INTER_INDEX_VEC", range(INTER_INDEX_VECTOR_SLICE.start, INTER_INDEX_VECTOR_SLICE.stop)),
        ("INTER_INDEX_DIST", [INTER_INDEX_DISTANCE_INDEX]),
        ("INTER_THUMB_DIST", [INTER_THUMB_DISTANCE_INDEX]),
        ("CONTACT_SCORE", [CONTACT_SCORE_INDEX]),
    ]
    
    total_elements = 0
    for name, idx_range in slices_and_indices:
        for idx in idx_range:
            assert idx not in covered, f"Index {idx} in {name} overlaps with another segment!"
            covered.add(idx)
            total_elements += 1
            
    assert total_elements == 158
    assert covered == set(range(158))

def test_feature_dimension_and_slice_bounds():
    """Verifies that all slice definitions partition exactly 158 contiguous, non-overlapping dimensions."""
    assert FEATURE_DIM == 158
    
    # Right Hand
    assert RIGHT_HAND_COORDS_SLICE == slice(0, 63)
    assert RIGHT_HAND_DISTANCES_SLICE == slice(63, 68)
    assert RIGHT_HAND_ANGLES_SLICE == slice(68, 73)
    assert RIGHT_HAND_GEOM_SLICE == slice(63, 73)
    assert RIGHT_HAND_TOTAL_SLICE == slice(0, 73)
    
    # Left Hand
    assert LEFT_HAND_COORDS_SLICE == slice(73, 136)
    assert LEFT_HAND_DISTANCES_SLICE == slice(136, 141)
    assert LEFT_HAND_ANGLES_SLICE == slice(141, 146)
    assert LEFT_HAND_GEOM_SLICE == slice(136, 146)
    assert LEFT_HAND_TOTAL_SLICE == slice(73, 146)
    
    # Presence Flags
    assert PRESENCE_FLAGS_SLICE == slice(146, 148)
    assert RIGHT_PRESENCE_INDEX == 146
    assert LEFT_PRESENCE_INDEX == 147
    
    # Inter-Hand Features
    assert INTER_HAND_TOTAL_SLICE == slice(148, 158)
    assert INTER_WRIST_VECTOR_SLICE == slice(148, 151)
    assert INTER_WRIST_DISTANCE_INDEX == 151
    assert INTER_INDEX_VECTOR_SLICE == slice(152, 155)
    assert INTER_INDEX_DISTANCE_INDEX == 155
    assert INTER_THUMB_DISTANCE_INDEX == 156
    assert CONTACT_SCORE_INDEX == 157
    
    # Ensure partition sum equals 158
    assert (63 + 5 + 5 + 63 + 5 + 5 + 2 + 3 + 1 + 3 + 1 + 1 + 1) == 158

def test_empty_hands_vector():
    """When no hands are detected, extractor must return 158 zeros with valid schema."""
    vec = FeatureExtractor158.extract_from_hands(right_hand=None, left_hand=None)
    assert vec.shape == (158,)
    assert vec.dtype == np.float32
    assert np.all(vec == 0.0)
    assert vec[RIGHT_PRESENCE_INDEX] == 0.0
    assert vec[LEFT_PRESENCE_INDEX] == 0.0
    assert validate_schema_vector(vec) is True

def test_one_handed_right_only():
    """One-handed right sign: right hand populated, left hand zero, inter-hand zero."""
    rh = create_synthetic_hand(wrist=(0.6, 0.5, 0.0), scale=0.1)
    vec = FeatureExtractor158.extract_from_hands(right_hand=rh, left_hand=None)
    
    assert vec.shape == (158,)
    assert vec[RIGHT_PRESENCE_INDEX] == 1.0
    assert vec[LEFT_PRESENCE_INDEX] == 0.0
    
    # Right hand features populated
    assert np.any(vec[RIGHT_HAND_COORDS_SLICE] != 0.0)
    assert np.all(vec[RIGHT_HAND_DISTANCES_SLICE] > 0.0)
    assert np.all(vec[RIGHT_HAND_ANGLES_SLICE] >= 0.0)
    
    # Left hand and inter-hand strictly zero
    assert np.all(vec[LEFT_HAND_TOTAL_SLICE] == 0.0)
    assert np.all(vec[INTER_HAND_TOTAL_SLICE] == 0.0)
    assert validate_schema_vector(vec) is True

def test_one_handed_left_only():
    """One-handed left sign: left hand populated, right hand zero, inter-hand zero."""
    lh = create_synthetic_hand(wrist=(0.4, 0.5, 0.0), scale=0.1)
    vec = FeatureExtractor158.extract_from_hands(right_hand=None, left_hand=lh)
    
    assert vec.shape == (158,)
    assert vec[RIGHT_PRESENCE_INDEX] == 0.0
    assert vec[LEFT_PRESENCE_INDEX] == 1.0
    
    assert np.all(vec[RIGHT_HAND_TOTAL_SLICE] == 0.0)
    assert np.any(vec[LEFT_HAND_COORDS_SLICE] != 0.0)
    assert np.all(vec[LEFT_HAND_DISTANCES_SLICE] > 0.0)
    assert np.all(vec[INTER_HAND_TOTAL_SLICE] == 0.0)
    assert validate_schema_vector(vec) is True

def test_two_handed_inter_hand_features():
    """Two-handed sign: both hands populated, inter-hand delta vectors and contact score active."""
    rh = create_synthetic_hand(wrist=(0.6, 0.5, 0.0), scale=0.1)
    lh = create_synthetic_hand(wrist=(0.4, 0.5, 0.0), scale=0.1)
    
    vec = FeatureExtractor158.extract_from_hands(right_hand=rh, left_hand=lh)
    
    assert vec[RIGHT_PRESENCE_INDEX] == 1.0
    assert vec[LEFT_PRESENCE_INDEX] == 1.0
    
    # Inter-wrist delta vector: Wrist_R (0.6, 0.5, 0.0) - Wrist_L (0.4, 0.5, 0.0) = (0.2, 0.0, 0.0)
    delta_w = vec[INTER_WRIST_VECTOR_SLICE]
    assert np.isclose(delta_w[0], 0.2, atol=1e-5)
    assert np.isclose(delta_w[1], 0.0, atol=1e-5)
    assert np.isclose(vec[INTER_WRIST_DISTANCE_INDEX], 0.2, atol=1e-5)
    
    # Inter-hand contact score
    c_score = vec[CONTACT_SCORE_INDEX]
    assert 0.0 <= c_score <= 1.0
    assert validate_schema_vector(vec) is True

def test_contact_score_high_when_touching():
    """Contact score must approach 1.0 when fingertips touch."""
    # Place fingertips at identical coordinate (0.5, 0.4, 0.0)
    rh = create_synthetic_hand(wrist=(0.55, 0.5, 0.0), scale=0.1)
    lh = create_synthetic_hand(wrist=(0.45, 0.5, 0.0), scale=0.1)
    rh[8] = [0.5, 0.4, 0.0] # Index tip R
    lh[8] = [0.5, 0.4, 0.0] # Index tip L
    
    vec = FeatureExtractor158.extract_from_hands(right_hand=rh, left_hand=lh)
    assert np.isclose(vec[INTER_INDEX_DISTANCE_INDEX], 0.0, atol=1e-6)
    assert np.isclose(vec[CONTACT_SCORE_INDEX], 1.0, atol=1e-3)

def test_scale_invariance_normalized_hand():
    """Translating and scaling a hand must produce identical normalized coordinates."""
    hand1 = create_synthetic_hand(wrist=(0.3, 0.3, 0.0), scale=0.1)
    # Translate by (+0.4, +0.2) and scale up by 2x
    hand2 = create_synthetic_hand(wrist=(0.7, 0.5, 0.0), scale=0.2)
    
    norm1, scale1 = normalize_hand_landmarks(hand1)
    norm2, scale2 = normalize_hand_landmarks(hand2)
    
    assert np.allclose(norm1, norm2, atol=1e-4)

def test_validation_rejects_corrupted_vectors():
    """validate_schema_vector must catch incorrect lengths, NaNs, and out-of-bound values."""
    # Wrong size (150 like aju22 instead of 158)
    with pytest.raises(ValueError, match="must have length 158"):
        validate_schema_vector(np.zeros(150, dtype=np.float32))
        
    # Contains NaN
    nan_vec = np.zeros(158, dtype=np.float32)
    nan_vec[10] = np.nan
    with pytest.raises(ValueError, match="contains NaN"):
        validate_schema_vector(nan_vec)
        
    # Invalid presence flag
    bad_presence = np.zeros(158, dtype=np.float32)
    bad_presence[RIGHT_PRESENCE_INDEX] = 2.5
    with pytest.raises(ValueError, match="presence flag must be 0.0 or 1.0"):
        validate_schema_vector(bad_presence)

if __name__ == '__main__':
    pytest.main([__file__, "-v"])
