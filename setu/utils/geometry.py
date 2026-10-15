"""
SETU - Geometric Utility Functions
Pure mathematical functions for Euclidean distances, 3D joint angles,
wrist-centering, scale normalization, and smooth inter-hand contact scoring.
"""

import numpy as np

def compute_euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Computes Euclidean distance between two 2D or 3D points."""
    return float(np.linalg.norm(pt1 - pt2))

def compute_joint_angle(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes angle at joint B given three connected points A -> B -> C.
    Returns angle in radians in range [0, pi].
    """
    vec_ba = p_a - p_b
    vec_bc = p_c - p_b
    
    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)
    
    if norm_ba < eps or norm_bc < eps:
        return 0.0
        
    cosine_val = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    # Clip to valid arccos domain [-1.0, 1.0] to prevent numerical NaN
    cosine_val = float(np.clip(cosine_val, -1.0, 1.0))
    return float(np.arccos(cosine_val))

def normalize_hand_landmarks(landmarks_21x3: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    Normalizes a single hand's 21 3D landmarks:
    1. Translates coordinates so wrist (landmark 0) is at origin (0, 0, 0).
    2. Scales coordinates by the distance between wrist (0) and middle MCP (9).
    Returns:
        (normalized_landmarks_21x3, scale_factor)
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros((21, 3), dtype=np.float32), 0.0
        
    wrist = landmarks_21x3[0]
    translated = landmarks_21x3 - wrist
    
    # Palm span: distance from wrist (0) to middle finger base / MCP (9)
    palm_span = float(np.linalg.norm(landmarks_21x3[9] - landmarks_21x3[0]))
    scale = max(palm_span, eps)
    
    normalized = translated / scale
    return normalized.astype(np.float32), scale

def compute_finger_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Computes 5 normalized fingertip-to-wrist distances:
    Thumb tip (4), Index tip (8), Middle tip (12), Ring tip (16), Pinky tip (20).
    Assumes landmarks_21x3 is already wrist-centered!
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(5, dtype=np.float32)
        
    wrist = landmarks_21x3[0]
    tip_indices = [4, 8, 12, 16, 20]
    dists = [np.linalg.norm(landmarks_21x3[idx] - wrist) for idx in tip_indices]
    return np.array(dists, dtype=np.float32)

def compute_finger_angles(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Computes 5 primary flexion angles (PIP / joint bend):
    - Thumb:  MCP(2) -> IP(3) -> Tip(4)
    - Index:  MCP(5) -> PIP(6) -> DIP(7)
    - Middle: MCP(9) -> PIP(10) -> DIP(11)
    - Ring:   MCP(13) -> PIP(14) -> DIP(15)
    - Pinky:  MCP(17) -> PIP(18) -> DIP(19)
    Normalized by pi so return values are in [0.0, 1.0].
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(5, dtype=np.float32)
        
    joints = [
        (2, 3, 4),    # Thumb
        (5, 6, 7),    # Index
        (9, 10, 11),  # Middle
        (13, 14, 15), # Ring
        (17, 18, 19)  # Pinky
    ]
    angles = []
    for a_idx, b_idx, c_idx in joints:
        rad = compute_joint_angle(landmarks_21x3[a_idx], landmarks_21x3[b_idx], landmarks_21x3[c_idx])
        # Normalize to [0, 1] relative to pi
        angles.append(rad / np.pi)
    return np.array(angles, dtype=np.float32)

def compute_contact_score(min_distance: float, sigma: float = 0.05) -> float:
    """
    Computes a smooth Gaussian contact proximity indicator in [0, 1]:
    Score is 1.0 when hands touch (distance ~ 0), decaying smoothly towards 0.
    """
    return float(np.exp(-(min_distance ** 2) / (2.0 * (sigma ** 2))))
