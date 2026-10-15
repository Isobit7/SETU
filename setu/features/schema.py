"""
SETU - 158-Feature Schema Specification
Immutable slice definitions, constants, and validation rules for the 158-dimensional feature representation.
"""

from typing import Final
import numpy as np

# Total Dimensions
FEATURE_DIM: Final[int] = 158

# Slice Constants
RIGHT_HAND_COORDS_SLICE: Final[slice] = slice(0, 63)
RIGHT_HAND_DISTANCES_SLICE: Final[slice] = slice(63, 68)
RIGHT_HAND_ANGLES_SLICE: Final[slice] = slice(68, 73)
RIGHT_HAND_GEOM_SLICE: Final[slice] = slice(63, 73)
RIGHT_HAND_TOTAL_SLICE: Final[slice] = slice(0, 73)

LEFT_HAND_COORDS_SLICE: Final[slice] = slice(73, 136)
LEFT_HAND_DISTANCES_SLICE: Final[slice] = slice(136, 141)
LEFT_HAND_ANGLES_SLICE: Final[slice] = slice(141, 146)
LEFT_HAND_GEOM_SLICE: Final[slice] = slice(136, 146)
LEFT_HAND_TOTAL_SLICE: Final[slice] = slice(73, 146)

PRESENCE_FLAGS_SLICE: Final[slice] = slice(146, 148)
RIGHT_PRESENCE_INDEX: Final[int] = 146
LEFT_PRESENCE_INDEX: Final[int] = 147

INTER_HAND_TOTAL_SLICE: Final[slice] = slice(148, 158)
INTER_WRIST_VECTOR_SLICE: Final[slice] = slice(148, 151)
INTER_WRIST_DISTANCE_INDEX: Final[int] = 151
INTER_INDEX_VECTOR_SLICE: Final[slice] = slice(152, 155)
INTER_INDEX_DISTANCE_INDEX: Final[int] = 155
INTER_THUMB_DISTANCE_INDEX: Final[int] = 156
CONTACT_SCORE_INDEX: Final[int] = 157

# Landmark Index Constants (MediaPipe 21 Hand Landmarks)
WRIST_INDEX: Final[int] = 0
THUMB_TIP_INDEX: Final[int] = 4
INDEX_MCP_INDEX: Final[int] = 5
INDEX_TIP_INDEX: Final[int] = 8
MIDDLE_MCP_INDEX: Final[int] = 9
MIDDLE_TIP_INDEX: Final[int] = 12
RING_TIP_INDEX: Final[int] = 16
PINKY_TIP_INDEX: Final[int] = 20

def validate_schema_vector(vec: np.ndarray) -> bool:
    """
    Validates that a NumPy vector matches the 158-feature schema contract:
    - 1D array of length 158 (or 2D (T, 158))
    - Correct data type (floating point)
    - No NaN or Inf values
    - Presence flags in {0.0, 1.0}
    - Contact score in [0.0, 1.0]
    """
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(vec)}")
        
    if vec.ndim == 1:
        if vec.shape[0] != FEATURE_DIM:
            raise ValueError(f"Feature vector must have length {FEATURE_DIM}, got {vec.shape[0]}")
    elif vec.ndim == 2:
        if vec.shape[1] != FEATURE_DIM:
            raise ValueError(f"Feature sequence must have feature dimension {FEATURE_DIM}, got {vec.shape[1]}")
    else:
        raise ValueError(f"Vector must be 1D or 2D, got shape {vec.shape}")
        
    if not np.issubdtype(vec.dtype, np.floating):
        raise TypeError(f"Vector must have floating dtype, got {vec.dtype}")
        
    if np.isnan(vec).any():
        raise ValueError("Feature vector contains NaN values")
        
    if np.isinf(vec).any():
        raise ValueError("Feature vector contains Inf values")
        
    # Check bounds on 1D vector
    if vec.ndim == 1:
        rp = vec[RIGHT_PRESENCE_INDEX]
        lp = vec[LEFT_PRESENCE_INDEX]
        if rp not in (0.0, 1.0):
            raise ValueError(f"Right presence flag must be 0.0 or 1.0, got {rp}")
        if lp not in (0.0, 1.0):
            raise ValueError(f"Left presence flag must be 0.0 or 1.0, got {lp}")
            
        c_score = vec[CONTACT_SCORE_INDEX]
        if not (0.0 <= c_score <= 1.0001):
            raise ValueError(f"Contact score must be in [0.0, 1.0], got {c_score}")
            
    return True
