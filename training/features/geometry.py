"""
SETU - Invariant Hand Geometry & Mathematics
Provides scale, translation, and rotation invariant hand landmark normalization,
15 finger joint angles (MCP, PIP, DIP for all 5 fingers), and pairwise fingertip distances.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Computes Euclidean distance between two 2D or 3D points."""
    return float(np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)))


def compute_joint_angle(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes the 3D angle at joint B given three connected points A -> B -> C.
    Returns angle in radians in range [0, pi].
    """
    vec_ba = np.asarray(p_a, dtype=np.float64) - np.asarray(p_b, dtype=np.float64)
    vec_bc = np.asarray(p_c, dtype=np.float64) - np.asarray(p_b, dtype=np.float64)

    norm_ba = float(np.linalg.norm(vec_ba))
    norm_bc = float(np.linalg.norm(vec_bc))

    if norm_ba < eps or norm_bc < eps:
        return 0.0

    cosine_val = float(np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc))
    # Clip to valid arccos domain [-1.0, 1.0] to prevent numerical NaN
    cosine_val = float(np.clip(cosine_val, -1.0, 1.0))
    return float(np.arccos(cosine_val))


def normalize_wrist_coordinates(
    landmarks_21x3: Optional[Union[np.ndarray, List[Any]]],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Normalizes a single hand's 21 3D landmarks for full scale and rotation invariance:
    1. Translates origin to Wrist (landmark 0).
    2. Computes palm reference length d = ||p_middle_mcp - p_wrist||. Scales all 21 points by 1/d.
    3. Rotates points so the palm vector (p_middle_mcp - p_wrist) aligns strictly with the vertical Y-axis.

    Returns:
        (normalized_landmarks_21x3, palm_reference_length)
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros((21, 3), dtype=np.float32), 0.0

    pts = np.asarray(landmarks_21x3, dtype=np.float64)
    if pts.shape == (21, 2):
        pts = np.hstack([pts, np.zeros((21, 1), dtype=np.float64)])
    elif pts.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32), 0.0

    # 1. Translate origin to Wrist (landmark 0)
    wrist = pts[0].copy()
    translated = pts - wrist

    # 2. Compute palm reference length d = ||p_middle_mcp - p_wrist||
    # Landmark 9 is MIDDLE_FINGER_MCP
    palm_vec = translated[9]
    d = float(np.linalg.norm(palm_vec))
    scale = max(d, eps)
    scaled = translated / scale

    if d < eps:
        return np.zeros((21, 3), dtype=np.float32), 0.0

    # 3. Rotate points so palm vector aligns strictly with the vertical Y-axis.
    # In standard camera/image coordinates, fingers point along negative Y (-Y).
    # Step A: In-plane rotation around Z axis so (v_x, v_y) aligns with (0, -norm_xy)
    v_x = float(scaled[9, 0])
    v_y = float(scaled[9, 1])
    v_z = float(scaled[9, 2])
    norm_xy = float(np.hypot(v_x, v_y))

    if norm_xy > eps:
        theta = float(np.arctan2(-v_x, -v_y))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x_rot = scaled[:, 0] * cos_t - scaled[:, 1] * sin_t
        y_rot = scaled[:, 0] * sin_t + scaled[:, 1] * cos_t
        z_rot = scaled[:, 2].copy()
    else:
        x_rot = scaled[:, 0].copy()
        y_rot = scaled[:, 1].copy()
        z_rot = scaled[:, 2].copy()

    # Step B: Out-of-plane rotation around X axis so (y', z') aligns with (-1, 0)
    y9 = float(y_rot[9])
    z9 = float(z_rot[9])
    norm_yz = float(np.hypot(y9, z9))
    if norm_yz > eps and abs(z9) > eps:
        phi = float(np.arctan2(z9, -y9))
        cos_p = float(np.cos(phi))
        sin_p = float(np.sin(phi))
        y_rot2 = y_rot * cos_p - z_rot * sin_p
        z_rot2 = y_rot * sin_p + z_rot * cos_p
        y_rot = y_rot2
        z_rot = z_rot2

    # Enforce exact canonical alignment on origin and middle MCP
    x_rot[0] = 0.0
    y_rot[0] = 0.0
    z_rot[0] = 0.0
    x_rot[9] = 0.0
    y_rot[9] = -1.0
    z_rot[9] = 0.0

    normalized = np.column_stack([x_rot, y_rot, z_rot]).astype(np.float32)
    return normalized, scale


# Alias for backward compatibility across existing pipeline modules
normalize_hand_landmarks = normalize_wrist_coordinates


# Key finger joint triplets: (Proximal, Vertex, Distal)
FINGER_JOINT_TRIPLETS_15 = [
    # Thumb: CMC(1), MCP(2), IP(3)
    (0, 1, 2),    # Thumb CMC
    (1, 2, 3),    # Thumb MCP
    (2, 3, 4),    # Thumb IP / tip joint
    # Index: MCP(5), PIP(6), DIP(7)
    (0, 5, 6),    # Index MCP
    (5, 6, 7),    # Index PIP
    (6, 7, 8),    # Index DIP
    # Middle: MCP(9), PIP(10), DIP(11)
    (0, 9, 10),   # Middle MCP
    (9, 10, 11),  # Middle PIP
    (10, 11, 12), # Middle DIP
    # Ring: MCP(13), PIP(14), DIP(15)
    (0, 13, 14),  # Ring MCP
    (13, 14, 15), # Ring PIP
    (14, 15, 16), # Ring DIP
    # Pinky: MCP(17), PIP(18), DIP(19)
    (0, 17, 18),  # Pinky MCP
    (17, 18, 19), # Pinky PIP
    (18, 19, 20), # Pinky DIP
]

_FINGER_JOINT_TRIPLETS_ARR = np.array(FINGER_JOINT_TRIPLETS_15, dtype=np.int32)
_FINGER_PRIMARY_TRIPLETS_ARR = np.array(
    [(2, 3, 4), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)],
    dtype=np.int32,
)


def compute_joint_angles_15(landmarks_21x3: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes 15 key finger joint angles across all 5 fingers via vectorized array operations:
    - Thumb:  Wrist(0)->CMC(1)->MCP(2), CMC(1)->MCP(2)->IP(3), MCP(2)->IP(3)->Tip(4)
    - Index:  Wrist(0)->MCP(5)->PIP(6), MCP(5)->PIP(6)->DIP(7), PIP(6)->DIP(7)->Tip(8)
    - Middle: Wrist(0)->MCP(9)->PIP(10), MCP(9)->PIP(10)->DIP(11), PIP(10)->DIP(11)->Tip(12)
    - Ring:   Wrist(0)->MCP(13)->PIP(14), MCP(13)->PIP(14)->DIP(15), PIP(14)->DIP(15)->Tip(16)
    - Pinky:  Wrist(0)->MCP(17)->PIP(18), MCP(17)->PIP(18)->DIP(19), PIP(18)->DIP(19)->Tip(20)

    Normalized by pi so return values are in [0.0, 1.0].
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(15, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float64)
    if pts.shape != (21, 3):
        return np.zeros(15, dtype=np.float32)

    triplets = _FINGER_JOINT_TRIPLETS_ARR
    p_a = pts[triplets[:, 0]]
    p_b = pts[triplets[:, 1]]
    p_c = pts[triplets[:, 2]]

    vec_ba = p_a - p_b
    vec_bc = p_c - p_b

    # Vectorized dot products and norms
    dot_prod = np.einsum("ij,ij->i", vec_ba, vec_bc)
    norm_ba = np.linalg.norm(vec_ba, axis=-1)
    norm_bc = np.linalg.norm(vec_bc, axis=-1)

    denom = norm_ba * norm_bc
    valid_mask = (norm_ba >= eps) & (norm_bc >= eps)

    safe_denom = np.where(valid_mask, denom, 1.0)
    cos_vals = np.clip(dot_prod / safe_denom, -1.0, 1.0)
    angles = np.where(valid_mask, np.arccos(cos_vals) / np.pi, 0.0)

    return angles.astype(np.float32)


# Alias
compute_15_joint_angles = compute_joint_angles_15


def compute_finger_angles(landmarks_21x3: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes 5 primary flexion angles (PIP / joint bend) for 158-D schema backward compatibility:
    - Thumb:  MCP(2) -> IP(3) -> Tip(4)
    - Index:  MCP(5) -> PIP(6) -> DIP(7)
    - Middle: MCP(9) -> PIP(10) -> DIP(11)
    - Ring:   MCP(13) -> PIP(14) -> DIP(15)
    - Pinky:  MCP(17) -> PIP(18) -> DIP(19)
    Normalized by pi so return values are in [0.0, 1.0].
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(5, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float64)
    if pts.shape != (21, 3):
        return np.zeros(5, dtype=np.float32)

    triplets = _FINGER_PRIMARY_TRIPLETS_ARR
    p_a = pts[triplets[:, 0]]
    p_b = pts[triplets[:, 1]]
    p_c = pts[triplets[:, 2]]

    vec_ba = p_a - p_b
    vec_bc = p_c - p_b

    dot_prod = np.einsum("ij,ij->i", vec_ba, vec_bc)
    norm_ba = np.linalg.norm(vec_ba, axis=-1)
    norm_bc = np.linalg.norm(vec_bc, axis=-1)

    denom = norm_ba * norm_bc
    valid_mask = (norm_ba >= eps) & (norm_bc >= eps)

    safe_denom = np.where(valid_mask, denom, 1.0)
    cos_vals = np.clip(dot_prod / safe_denom, -1.0, 1.0)
    angles = np.where(valid_mask, np.arccos(cos_vals) / np.pi, 0.0)

    return angles.astype(np.float32)


def compute_fingertip_to_wrist_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Computes 5 normalized fingertip-to-wrist distances:
    Thumb tip (4), Index tip (8), Middle tip (12), Ring tip (16), Pinky tip (20).
    Assumes landmarks_21x3 is already wrist-centered!
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(5, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(5, dtype=np.float32)

    wrist = pts[0]
    tip_indices = [4, 8, 12, 16, 20]
    diff = pts[tip_indices] - wrist
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


compute_finger_distances = compute_fingertip_to_wrist_distances


def compute_fingertip_to_thumb_distances(
    landmarks_21x3: np.ndarray,
    include_self: bool = False,
) -> np.ndarray:
    """
    Computes normalized pairwise distances from fingertips to Thumb Tip (landmark 4):
    - When include_self=False: [Index(8)-Thumb(4), Middle(12)-Thumb(4), Ring(16)-Thumb(4), Pinky(20)-Thumb(4)] (4 distances)
    - When include_self=True:  [Thumb(4)-Thumb(4), Index(8)-Thumb(4), Middle(12)-Thumb(4), Ring(16)-Thumb(4), Pinky(20)-Thumb(4)] (5 distances)
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(5 if include_self else 4, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(5 if include_self else 4, dtype=np.float32)

    thumb_tip = pts[4]
    indices = [4, 8, 12, 16, 20] if include_self else [8, 12, 16, 20]
    diff = pts[indices] - thumb_tip
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def compute_pairwise_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Computes combined normalized fingertip-to-wrist (5) and fingertip-to-thumb (4) distances (9-D).
    """
    d_wrist = compute_fingertip_to_wrist_distances(landmarks_21x3)
    d_thumb = compute_fingertip_to_thumb_distances(landmarks_21x3, include_self=False)
    return np.concatenate([d_wrist, d_thumb]).astype(np.float32)


def extract_invariant_hand_features(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Extracts complete fully-invariant hand feature vector (87-D):
    - 63-D Normalized 3D Coordinates (wrist-centered, scale-normalized, rotated to vertical Y)
    - 15-D Finger Joint Angles (MCP, PIP, DIP for 5 fingers)
    - 5-D Fingertip-to-Wrist distances
    - 4-D Fingertip-to-Thumb distances
    """
    norm_pts, _ = normalize_wrist_coordinates(landmarks_21x3)
    coords_63 = norm_pts.flatten()
    angles_15 = compute_joint_angles_15(norm_pts)
    dist_wrist_5 = compute_fingertip_to_wrist_distances(norm_pts)
    dist_thumb_4 = compute_fingertip_to_thumb_distances(norm_pts, include_self=False)

    return np.concatenate([coords_63, angles_15, dist_wrist_5, dist_thumb_4]).astype(np.float32)


def compute_contact_score(min_distance: float, sigma: float = 0.05) -> float:
    """
    Computes a smooth Gaussian contact proximity indicator in [0, 1]:
    Score is 1.0 when hands touch (distance ~ 0), decaying smoothly towards 0.
    """
    return float(np.exp(-(min_distance ** 2) / (2.0 * (sigma ** 2))))


def get_finger_extension_states(
    pts: Union[np.ndarray, List[Any]],
    thumb_thresh: float = 1.15,
    index_thresh: float = 1.30,
    middle_thresh: float = 1.30,
    ring_thresh: float = 1.25,
    pinky_thresh: float = 1.15,
) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Analyzes 21 hand landmarks to determine physical extension state of each finger:
    1. Computes distances from fingertips:
       - Thumb tip (landmark 4)
       - Index tip (landmark 8)
       - Middle tip (landmark 12)
       - Ring tip (landmark 16)
       - Pinky tip (landmark 20)
       to the Wrist (landmark 0).
    2. Normalizes distances by Palm reference length (distance between Wrist 0 and Middle MCP 9).
    3. Compares against extension thresholds.

    Returns:
        5-boolean tuple (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext).
    """
    if pts is None or len(pts) == 0:
        return (False, False, False, False, False)

    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 63:
        arr = arr.reshape(21, 3)
    elif arr.shape not in ((21, 3), (21, 2)):
        if len(arr) == 21:
            arr = arr.reshape(21, -1)
        else:
            return (False, False, False, False, False)

    wrist = arr[0]
    middle_mcp = arr[9]
    palm_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_len < 1e-6:
        return (False, False, False, False, False)

    d_thumb = float(np.linalg.norm(arr[4] - wrist)) / palm_len
    d_index = float(np.linalg.norm(arr[8] - wrist)) / palm_len
    d_middle = float(np.linalg.norm(arr[12] - wrist)) / palm_len
    d_ring = float(np.linalg.norm(arr[16] - wrist)) / palm_len
    d_pinky = float(np.linalg.norm(arr[20] - wrist)) / palm_len

    thumb_ext = bool(d_thumb > thumb_thresh)
    index_ext = bool(d_index > index_thresh)
    middle_ext = bool(d_middle > middle_thresh)
    ring_ext = bool(d_ring > ring_thresh)
    pinky_ext = bool(d_pinky > pinky_thresh)

    return (thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext)


def evaluate_alphabet_heuristics(pts: Union[np.ndarray, List[Any]]) -> Optional[str]:
    """
    Evaluates physical finger-extension heuristics for static 26-class alphabet:
    - Index + Thumb extended, others curled ➔ Boost 'L'
    - Pinky extended only, others curled ➔ Boost 'I'
    - Index + Middle extended (apart), others curled ➔ Boost 'V'
    - Index + Middle + Ring extended, others curled ➔ Boost 'W'
    - Thumb + Pinky extended, middle three curled ➔ Boost 'Y'
    - All 4 fingers extended together ➔ Boost 'B'
    - All fingers curled into fist ➔ Boost 'A'

    Returns:
        Letter string ('L', 'I', 'V', 'W', 'Y', 'B', 'A') to boost, or None if no rule matches.
    """
    if pts is None or len(pts) == 0:
        return None

    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 63:
        arr = arr.reshape(21, 3)
    elif arr.shape not in ((21, 3), (21, 2)):
        return None

    wrist = arr[0]
    middle_mcp = arr[9]
    palm_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_len < 1e-6:
        return None

    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = get_finger_extension_states(arr)

    # 1. 'A': All fingers curled into fist
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        return "A"

    # 2. 'B': All 4 fingers extended together (thumb folded / curled)
    if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
        return "B"

    # 3. 'L': Index + Thumb extended, others curled
    if thumb_ext and index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "L"

    # 4. 'I': Pinky extended only, others curled
    if pinky_ext and not thumb_ext and not index_ext and not middle_ext and not ring_ext:
        return "I"

    # 5. 'V': Index + Middle extended (apart), others curled
    if index_ext and middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        d_idx_mid = float(np.linalg.norm(arr[8] - arr[12])) / palm_len
        if d_idx_mid >= 0.20:
            return "V"

    # 6. 'W': Index + Middle + Ring extended, others curled
    if index_ext and middle_ext and ring_ext and not pinky_ext and not thumb_ext:
        return "W"

    # 7. 'Y': Thumb + Pinky extended, middle three curled
    if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return "Y"

    return None

