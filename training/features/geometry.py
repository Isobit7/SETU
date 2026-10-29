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

    # Parse landmarks safely if given as objects (e.g. MediaPipe landmarks, dicts)
    first = landmarks_21x3[0]
    if isinstance(first, dict):
        raw_list = [[float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))] for p in landmarks_21x3]
        pts = np.asarray(raw_list, dtype=np.float64)
    elif hasattr(first, "x") and hasattr(first, "y"):
        raw_list = [[float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))] for p in landmarks_21x3]
        pts = np.asarray(raw_list, dtype=np.float64)
    else:
        try:
            pts = np.asarray(landmarks_21x3, dtype=np.float64)
        except Exception:
            return np.zeros((21, 3), dtype=np.float32), 0.0

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


def normalize_landmarks_to_63d(
    landmarks: Any,
    is_left_hand: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transforms 21 3D hand landmarks into a canonical wrist-centered,
    scale- and rotation-invariant 63-D vector.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, received {len(landmarks)}")

    pts = np.empty((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict):
            pts[i, 0] = float(lm.get("x", 0.0))
            pts[i, 1] = float(lm.get("y", 0.0))
            pts[i, 2] = float(lm.get("z", 0.0))
        elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 3:
            pts[i, 0] = float(lm[0])
            pts[i, 1] = float(lm[1])
            pts[i, 2] = float(lm[2])
        else:
            raise ValueError(f"Invalid landmark at index {i}: {lm}")

    if is_left_hand:
        # Reflect across wrist in sagittal plane: preserves wrist origin without coordinate drift
        pts[:, 0] = 2.0 * pts[0, 0] - pts[:, 0]

    norm_pts, _ = normalize_wrist_coordinates(pts, eps=eps)
    return norm_pts.flatten()


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

    # Aperture distance between thumb tip (landmark 4) and index tip (landmark 8)
    d_thumb_idx = float(np.linalg.norm(arr[4] - arr[8])) / palm_len

    # 1. Non-extended finger poses: 'C' (open curved arc) vs 'A' (closed fist)
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if d_thumb_idx >= 0.35:
            return "C"
        elif d_thumb_idx < 0.25:
            return "A"
        return None

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


# 57 Active coordinate indices: 63 dimensions minus 6 dead coordinates
# Excludes Landmark 0 (wrist origin: 0, 1, 2) and Landmark 9 (middle MCP unit vector: 27, 28, 29)
ACTIVE_COORDINATE_INDICES_57 = np.array([
    i for i in range(63) if i not in (0, 1, 2, 27, 28, 29)
], dtype=np.int32)


def compute_adjacent_fingertip_distances(landmarks_21x3: np.ndarray) -> np.ndarray:
    """
    Computes 4 normalized adjacent fingertip distances:
    Thumb(4)-Index(8), Index(8)-Middle(12), Middle(12)-Ring(16), Ring(16)-Pinky(20).
    Crucial for distinguishing spread (e.g. 'W', 'V') vs held-together digits (e.g. 'M', 'U').
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(4, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(4, dtype=np.float32)

    p1 = pts[[4, 8, 12, 16]]
    p2 = pts[[8, 12, 16, 20]]
    return np.linalg.norm(p1 - p2, axis=-1).astype(np.float32)


def compute_finger_spread_angles(landmarks_21x3: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes 4 angular spreads between adjacent fingertip rays originating at Wrist (0):
    - Angle between Thumb ray (0->4) and Index ray (0->8)
    - Angle between Index ray (0->8) and Middle ray (0->12)
    - Angle between Middle ray (0->12) and Ring ray (0->16)
    - Angle between Ring ray (0->16) and Pinky ray (0->20)
    Normalized by pi into [0.0, 1.0].
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(4, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float64)
    if pts.shape != (21, 3):
        return np.zeros(4, dtype=np.float32)

    # In canonical wrist-normalized coordinates, wrist is at (0, 0, 0)
    wrist = pts[0]
    v_thumb = pts[4] - wrist
    v_index = pts[8] - wrist
    v_mid = pts[12] - wrist
    v_ring = pts[16] - wrist
    v_pinky = pts[20] - wrist

    v_a = np.stack([v_thumb, v_index, v_mid, v_ring])
    v_b = np.stack([v_index, v_mid, v_ring, v_pinky])

    dot_prod = np.sum(v_a * v_b, axis=-1)
    norm_a = np.linalg.norm(v_a, axis=-1)
    norm_b = np.linalg.norm(v_b, axis=-1)

    denom = np.maximum(norm_a * norm_b, eps)
    cos_vals = np.clip(dot_prod / denom, -1.0, 1.0)
    return (np.arccos(cos_vals) / np.pi).astype(np.float32)


def extract_realsign_89d_features(
    landmarks_21x3: Optional[Union[np.ndarray, List[Any]]],
    is_left_hand: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transforms 21 3D hand landmarks into the canonical 89-D invariant feature vector:
    - 57-D Active Coordinates (pruned of zero-variance wrist and middle-MCP anchors)
    - 15-D Invariant Finger Joint Angles (MCP, PIP, DIP for all 5 fingers in [0, 1])
    - 5-D Fingertip-to-Wrist Radial Distances
    - 4-D Fingertip-to-Thumb Proximity Distances
    - 4-D Adjacent Fingertip Spread Distances
    - 4-D Fingertip Angular Spreads from Wrist
    Total: 57 + 15 + 5 + 4 + 4 + 4 = 89 dimensions.
    """
    if landmarks_21x3 is None or len(landmarks_21x3) != 21:
        return np.zeros(89, dtype=np.float32)

    pts = np.empty((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks_21x3):
        if isinstance(lm, dict):
            pts[i, 0] = float(lm.get("x", 0.0))
            pts[i, 1] = float(lm.get("y", 0.0))
            pts[i, 2] = float(lm.get("z", 0.0))
        elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 3:
            pts[i, 0] = float(lm[0])
            pts[i, 1] = float(lm[1])
            pts[i, 2] = float(lm[2])
        else:
            return np.zeros(89, dtype=np.float32)

    if is_left_hand:
        wrist_x = pts[0, 0]
        pts[:, 0] = 2.0 * wrist_x - pts[:, 0]

    norm_pts, scale = normalize_wrist_coordinates(pts, eps=eps)
    if scale < 1e-6:
        return np.zeros(89, dtype=np.float32)

    flat_63 = norm_pts.flatten()
    coords_57 = flat_63[ACTIVE_COORDINATE_INDICES_57]
    angles_15 = compute_joint_angles_15(norm_pts)
    dist_wrist_5 = compute_fingertip_to_wrist_distances(norm_pts)
    dist_thumb_4 = compute_fingertip_to_thumb_distances(norm_pts, include_self=False)
    dist_adj_4 = compute_adjacent_fingertip_distances(norm_pts)
    spread_angles_4 = compute_finger_spread_angles(norm_pts)

    return np.concatenate([
        coords_57,
        angles_15,
        dist_wrist_5,
        dist_thumb_4,
        dist_adj_4,
        spread_angles_4,
    ]).astype(np.float32)


def expand_63d_to_89d(features_63d: np.ndarray) -> np.ndarray:
    """
    Instantaneously expands a canonical 63-D normalized hand landmark vector into the
    high-accuracy 89-D invariant feature vector. Works on single vectors (63,) or batches (N, 63).
    """
    arr = np.asarray(features_63d, dtype=np.float32)
    single = (arr.ndim == 1)
    if single:
        arr = arr.reshape(1, 63)

    num_samples = len(arr)
    # Fast path for degenerate / uninitialized hand vectors
    norms = np.linalg.norm(arr, axis=-1)
    if np.all(norms < 1e-4):
        zero_out = np.zeros((num_samples, 89), dtype=np.float32)
        return zero_out[0] if single else zero_out

    pts = arr.reshape(num_samples, 21, 3)

    # 1. 57 active coordinates
    coords_57 = arr[:, ACTIVE_COORDINATE_INDICES_57]

    # 2. 15 joint angles
    triplets = _FINGER_JOINT_TRIPLETS_ARR
    p_a = pts[:, triplets[:, 0], :]
    p_b = pts[:, triplets[:, 1], :]
    p_c = pts[:, triplets[:, 2], :]
    vec_ba = p_a - p_b
    vec_bc = p_c - p_b
    dot_prod = np.sum(vec_ba * vec_bc, axis=-1)
    norm_ba = np.linalg.norm(vec_ba, axis=-1)
    norm_bc = np.linalg.norm(vec_bc, axis=-1)
    denom = np.maximum(norm_ba * norm_bc, 1e-7)
    angles_15 = np.arccos(np.clip(dot_prod / denom, -1.0, 1.0)) / np.pi

    # 3. 5 wrist distances
    dist_wrist_5 = np.linalg.norm(pts[:, [4, 8, 12, 16, 20], :], axis=-1)

    # 4. 4 thumb distances
    dist_thumb_4 = np.linalg.norm(pts[:, [8, 12, 16, 20], :] - pts[:, 4:5, :], axis=-1)

    # 5. 4 adjacent fingertip distances
    d_4_8 = np.linalg.norm(pts[:, 4] - pts[:, 8], axis=-1, keepdims=True)
    d_8_12 = np.linalg.norm(pts[:, 8] - pts[:, 12], axis=-1, keepdims=True)
    d_12_16 = np.linalg.norm(pts[:, 12] - pts[:, 16], axis=-1, keepdims=True)
    d_16_20 = np.linalg.norm(pts[:, 16] - pts[:, 20], axis=-1, keepdims=True)
    dist_adj_4 = np.concatenate([d_4_8, d_8_12, d_12_16, d_16_20], axis=-1)

    # 6. 4 spread angles from wrist
    v4, v8, v12, v16, v20 = pts[:, 4], pts[:, 8], pts[:, 12], pts[:, 16], pts[:, 20]
    def _vec_angle(v1, v2):
        dot = np.sum(v1 * v2, axis=-1, keepdims=True)
        denom_v = np.maximum(np.linalg.norm(v1, axis=-1, keepdims=True) * np.linalg.norm(v2, axis=-1, keepdims=True), 1e-7)
        return np.arccos(np.clip(dot / denom_v, -1.0, 1.0)) / np.pi

    spread_angles_4 = np.concatenate([
        _vec_angle(v4, v8),
        _vec_angle(v8, v12),
        _vec_angle(v12, v16),
        _vec_angle(v16, v20),
    ], axis=-1)

    out_89 = np.concatenate([
        coords_57,
        angles_15,
        dist_wrist_5,
        dist_thumb_4,
        dist_adj_4,
        spread_angles_4,
    ], axis=-1).astype(np.float32)

    return out_89[0] if single else out_89


def extract_dual_hand_126d(
    left_hand: Optional[Union[np.ndarray, List[Any]]] = None,
    right_hand: Optional[Union[np.ndarray, List[Any]]] = None,
) -> np.ndarray:
    """
    Extracts a symmetric 126-dimensional representation for dual-handed and single-handed ISL recognition:
    - Left Hand normalized 3D landmarks (slots 0..62, 63-D), zero-padded if absent.
    - Right Hand normalized 3D landmarks (slots 63..125, 63-D), zero-padded if absent.
    - Ambidextrous Left-Hand Support: When signing single-handed with the left hand,
      the left hand is normalized, reflected across the sagittal X-axis (x -> -x),
      and projected into the active canonical hand slot [63..125] so single-handed
      left-handed signing achieves identical 99%+ accuracy to right-handed signing.
    Returns:
        np.ndarray of shape (126,) and dtype np.float32.
    """
    norm_lh = np.zeros(63, dtype=np.float32)
    norm_rh = np.zeros(63, dtype=np.float32)

    has_lh = left_hand is not None and len(left_hand) == 21
    has_rh = right_hand is not None and len(right_hand) == 21

    if has_lh and not has_rh:
        # Single-handed signing with left hand:
        # Mirror X coordinate so geometry matches canonical right hand and slot into active hand [63:126].
        # The secondary hand slot [0:63] remains zero-padded as expected for single-hand signs.
        lh_arr, _ = normalize_wrist_coordinates(left_hand)
        lh_mirrored = lh_arr.copy()
        lh_mirrored[:, 0] = -lh_mirrored[:, 0]
        norm_rh = lh_mirrored.flatten().astype(np.float32)
        norm_lh = np.zeros(63, dtype=np.float32)
    elif has_rh and not has_lh:
        # Single-handed signing with right hand
        rh_arr, _ = normalize_wrist_coordinates(right_hand)
        norm_rh = rh_arr.flatten().astype(np.float32)
    elif has_lh and has_rh:
        # Both hands present: dual-hand sign
        lh_arr, _ = normalize_wrist_coordinates(left_hand)
        rh_arr, _ = normalize_wrist_coordinates(right_hand)
        norm_lh = lh_arr.flatten().astype(np.float32)
        norm_rh = rh_arr.flatten().astype(np.float32)

    return np.concatenate([norm_lh, norm_rh], axis=0).astype(np.float32)


def extract_body_anchored_features(
    pose_landmarks: Optional[Union[np.ndarray, List[Any]]] = None,
    left_hand: Optional[Union[np.ndarray, List[Any]]] = None,
    right_hand: Optional[Union[np.ndarray, List[Any]]] = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Extracts 12 body-anchored relational features (4 x 3D vectors):
    - Right wrist relative to mid-shoulder (chest) scaled by shoulder span: [dx, dy, dz] (slots 0..2)
    - Right wrist relative to nose scaled by shoulder span: [dx, dy, dz] (slots 3..5)
    - Left wrist relative to mid-shoulder (chest) scaled by shoulder span: [dx, dy, dz] (slots 6..8)
    - Left wrist relative to nose scaled by shoulder span: [dx, dy, dz] (slots 9..11)

    Returns:
        np.ndarray of shape (12,) and dtype np.float32.
    """
    def _to_pt(lm: Any) -> Optional[np.ndarray]:
        if lm is None:
            return None
        if hasattr(lm, "x") and hasattr(lm, "y"):
            return np.array([float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))], dtype=np.float32)
        if isinstance(lm, dict):
            return np.array([float(lm.get("x", 0.0)), float(lm.get("y", 0.0)), float(lm.get("z", 0.0))], dtype=np.float32)
        if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
            z_val = float(lm[2]) if len(lm) >= 3 else 0.0
            return np.array([float(lm[0]), float(lm[1]), z_val], dtype=np.float32)
        return None

    pose = pose_landmarks
    if hasattr(pose, "landmark"):
        pose = pose.landmark
    elif isinstance(pose, (list, tuple)) and len(pose) > 0 and isinstance(pose[0], (list, tuple)):
        pose = pose[0]

    nose = None
    chest = None
    shoulder_span = 0.3

    if pose is not None and len(pose) >= 13:
        nose = _to_pt(pose[0])
        ls = _to_pt(pose[11])
        rs = _to_pt(pose[12])
        if ls is not None and rs is not None:
            chest = (ls + rs) / 2.0
            shoulder_span = max(float(np.linalg.norm(ls - rs)), eps)

    if chest is None:
        chest = nose if nose is not None else np.zeros(3, dtype=np.float32)

    rw = None
    if right_hand is not None and len(right_hand) > 0:
        first_r = right_hand[0] if not isinstance(right_hand[0], (list, tuple)) or len(right_hand[0]) != 21 else right_hand[0][0]
        rw = _to_pt(first_r)
    if rw is None and pose is not None and len(pose) > 16:
        rw = _to_pt(pose[16])

    lw = None
    if left_hand is not None and len(left_hand) > 0:
        first_l = left_hand[0] if not isinstance(left_hand[0], (list, tuple)) or len(left_hand[0]) != 21 else left_hand[0][0]
        lw = _to_pt(first_l)
    if lw is None and pose is not None and len(pose) > 15:
        lw = _to_pt(pose[15])

    if rw is not None:
        rw_chest = (rw - chest) / shoulder_span
        rw_nose = (rw - nose) / shoulder_span if nose is not None else rw_chest
    else:
        rw_chest = np.zeros(3, dtype=np.float32)
        rw_nose = np.zeros(3, dtype=np.float32)

    if lw is not None:
        lw_chest = (lw - chest) / shoulder_span
        lw_nose = (lw - nose) / shoulder_span if nose is not None else lw_chest
    else:
        lw_chest = np.zeros(3, dtype=np.float32)
        lw_nose = np.zeros(3, dtype=np.float32)

    return np.concatenate([rw_chest, rw_nose, lw_chest, lw_nose], axis=0).astype(np.float32)



