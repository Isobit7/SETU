"""
SETU - Comprehensive Feature Schema & Extractor Unit Tests
Tests every slice, dimension boundary, normalization invariant, and handedness condition for the 158-feature representation.
"""

import pytest
import numpy as np

from training.feature_extractor import (
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
from training.feature_extractor import FeatureExtractor158
from training.feature_extractor import (
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


def test_facial_expression_feature_fallback():
    """Missing or empty face landmarks must safely return [0.0] * 6."""
    from training.feature_extractor import extract_facial_expression_features

    assert extract_facial_expression_features(None) == [0.0] * 6
    assert extract_facial_expression_features([]) == [0.0] * 6
    assert extract_facial_expression_features([{"x": 0.5, "y": 0.5}] * 10) == [0.0] * 6


def test_facial_expression_ratio_math():
    """Verifies that the 6 scale-invariant facial ratios compute accurately."""
    from training.feature_extractor import extract_facial_expression_features

    # Create synthetic 468 face mesh landmarks
    synthetic_mesh = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(468)]

    # 1. Mouth Aspect Ratio: ||P_13 - P_14|| / ||P_61 - P_291||
    synthetic_mesh[13] = {"x": 0.5, "y": 0.65, "z": 0.0}   # Upper inner lip
    synthetic_mesh[14] = {"x": 0.5, "y": 0.70, "z": 0.0}   # Lower inner lip (dy = 0.05)
    synthetic_mesh[61] = {"x": 0.45, "y": 0.675, "z": 0.0} # Mouth corner left
    synthetic_mesh[291] = {"x": 0.55, "y": 0.675, "z": 0.0} # Mouth corner right (dx = 0.10)
    # Expected MAR = 0.05 / 0.10 = 0.50

    # Face height normalization: ||P_10 - P_152|| (Forehead top to chin)
    synthetic_mesh[10] = {"x": 0.5, "y": 0.1, "z": 0.0}
    synthetic_mesh[152] = {"x": 0.5, "y": 0.9, "z": 0.0} # dy = 0.80

    # 2. Left Eyebrow: ||P_105 - P_159||
    synthetic_mesh[105] = {"x": 0.4, "y": 0.22, "z": 0.0}
    synthetic_mesh[159] = {"x": 0.4, "y": 0.30, "z": 0.0} # dy = 0.08 -> left_eer = 0.08 / 0.80 = 0.10

    # 3. Right Eyebrow: ||P_334 - P_386||
    synthetic_mesh[334] = {"x": 0.6, "y": 0.22, "z": 0.0}
    synthetic_mesh[386] = {"x": 0.6, "y": 0.30, "z": 0.0} # dy = 0.08 -> right_eer = 0.08 / 0.80 = 0.10

    # 5. Left Eye Aspect Ratio (EAR)
    synthetic_mesh[33] = {"x": 0.35, "y": 0.35, "z": 0.0}
    synthetic_mesh[133] = {"x": 0.45, "y": 0.35, "z": 0.0} # width = 0.10
    synthetic_mesh[160] = {"x": 0.40, "y": 0.33, "z": 0.0}
    synthetic_mesh[144] = {"x": 0.40, "y": 0.37, "z": 0.0} # height1 = 0.04
    synthetic_mesh[158] = {"x": 0.40, "y": 0.33, "z": 0.0}
    synthetic_mesh[153] = {"x": 0.40, "y": 0.37, "z": 0.0} # height2 = 0.04
    # Expected Left EAR = (0.04 + 0.04) / (2 * 0.10) = 0.08 / 0.20 = 0.40

    # 6. Right Eye Aspect Ratio (EAR)
    synthetic_mesh[362] = {"x": 0.55, "y": 0.35, "z": 0.0}
    synthetic_mesh[263] = {"x": 0.65, "y": 0.35, "z": 0.0} # width = 0.10
    synthetic_mesh[385] = {"x": 0.60, "y": 0.33, "z": 0.0}
    synthetic_mesh[380] = {"x": 0.60, "y": 0.37, "z": 0.0} # height1 = 0.04
    synthetic_mesh[387] = {"x": 0.60, "y": 0.33, "z": 0.0}
    synthetic_mesh[373] = {"x": 0.60, "y": 0.37, "z": 0.0} # height2 = 0.04
    # Expected Right EAR = 0.40

    ratios = extract_facial_expression_features(synthetic_mesh)
    assert len(ratios) == 6
    mar, left_eer, right_eer, mean_eer, left_ear, right_ear = ratios

    assert np.isclose(mar, 0.50, atol=1e-3)
    assert np.isclose(left_eer, 0.10, atol=1e-3)
    assert np.isclose(right_eer, 0.10, atol=1e-3)
    assert np.isclose(mean_eer, 0.10, atol=1e-3)
    assert np.isclose(left_ear, 0.40, atol=1e-3)
    assert np.isclose(right_ear, 0.40, atol=1e-3)


def test_expanded_164_feature_vector_validation():
    """Schema must validate 164-D vectors (158 Holistic + 6 Facial Ratios)."""
    from training.feature_extractor import EXPANDED_FEATURE_DIM

    vec_164 = np.zeros(EXPANDED_FEATURE_DIM, dtype=np.float32)
    vec_164[RIGHT_PRESENCE_INDEX] = 1.0
    vec_164[LEFT_PRESENCE_INDEX] = 0.0
    vec_164[CONTACT_SCORE_INDEX] = 0.5
    # Set facial ratios at slots 158:164
    vec_164[158:164] = [0.25, 0.12, 0.12, 0.12, 0.30, 0.30]

    assert validate_schema_vector(vec_164) is True

    # 2D batch/sequence of 164-D features
    seq_164 = np.zeros((30, EXPANDED_FEATURE_DIM), dtype=np.float32)
    assert validate_schema_vector(seq_164) is True


def test_velocity_features_328_computation():
    """compute_velocity_features must produce exact 328-D vectors across all input shapes."""
    from training.feature_extractor import compute_velocity_features

    # (30, 164) -> (30, 328)
    seq_164 = np.ones((30, 164), dtype=np.float32)
    res_328_from_164 = compute_velocity_features(seq_164, target_dim=328)
    assert res_328_from_164.shape == (30, 328)
    # First frame velocity should be 0.0
    assert np.all(res_328_from_164[0, 164:] == 0.0)

    # (30, 158) -> zero-padded to 164 -> (30, 328)
    seq_158 = np.ones((30, 158), dtype=np.float32)
    res_328_from_158 = compute_velocity_features(seq_158, target_dim=328)
    assert res_328_from_158.shape == (30, 328)

    # (30, 316) -> (30, 328)
    seq_316 = np.ones((30, 316), dtype=np.float32)
    res_328_from_316 = compute_velocity_features(seq_316, target_dim=328)
    assert res_328_from_316.shape == (30, 328)

    # (30, 328) -> (30, 328)
    seq_328 = np.ones((30, 328), dtype=np.float32)
    res_328_from_328 = compute_velocity_features(seq_328, target_dim=328)
    assert res_328_from_328.shape == (30, 328)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
