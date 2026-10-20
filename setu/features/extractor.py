"""
SETU - 158-Feature Extractor
Extracts wrist-centered, scale-invariant 3D coordinates, finger geometry,
and inter-hand relational features from MediaPipe hand landmarks.
"""

from typing import Optional, Union, List, Any
import numpy as np

from setu.exceptions import LandmarkExtractionError
from setu.utils.geometry import (
    normalize_hand_landmarks,
    compute_finger_distances,
    compute_finger_angles,
    compute_euclidean_distance,
    compute_contact_score
)
from setu.features.schema import (
    FEATURE_DIM,
    RIGHT_HAND_COORDS_SLICE,
    RIGHT_HAND_DISTANCES_SLICE,
    RIGHT_HAND_ANGLES_SLICE,
    LEFT_HAND_COORDS_SLICE,
    LEFT_HAND_DISTANCES_SLICE,
    LEFT_HAND_ANGLES_SLICE,
    RIGHT_PRESENCE_INDEX,
    LEFT_PRESENCE_INDEX,
    INTER_WRIST_VECTOR_SLICE,
    INTER_WRIST_DISTANCE_INDEX,
    INTER_INDEX_VECTOR_SLICE,
    INTER_INDEX_DISTANCE_INDEX,
    INTER_THUMB_DISTANCE_INDEX,
    CONTACT_SCORE_INDEX,
    WRIST_INDEX,
    THUMB_TIP_INDEX,
    INDEX_TIP_INDEX,
    validate_schema_vector,
    FACIAL_RATIO_DIM,
    EXPANDED_FEATURE_DIM,
    FACIAL_RATIOS_SLICE,
    VELOCITY_FEATURE_DIM_316,
    VELOCITY_FEATURE_DIM_328,
)

def _to_numpy_21x3(landmarks: Optional[Union[np.ndarray, List[Any]]]) -> Optional[np.ndarray]:
    """Converts a list of landmark objects or array into a standard (21, 3) float32 array."""
    if landmarks is None or len(landmarks) == 0:
        return None
        
    if hasattr(landmarks, "landmark"):
        landmarks = landmarks.landmark

    if isinstance(landmarks, np.ndarray):
        if landmarks.shape == (21, 3):
            return landmarks.astype(np.float32)
        elif landmarks.shape == (21, 2):
            # Pad z with zeros if only 2D provided
            z = np.zeros((21, 1), dtype=np.float32)
            return np.hstack([landmarks, z]).astype(np.float32)
        else:
            raise LandmarkExtractionError(f"Expected array of shape (21, 3) or (21, 2), got {landmarks.shape}")
            
    # List of landmark objects with .x, .y, .z attributes or dicts
    coords = []
    for lm in landmarks:
        if isinstance(lm, dict):
            coords.append([lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)])
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            z_val = lm[2] if len(lm) >= 3 else 0.0
            coords.append([lm[0], lm[1], z_val])
        else:
            x = getattr(lm, 'x', 0.0)
            y = getattr(lm, 'y', 0.0)
            z = getattr(lm, 'z', 0.0)
            coords.append([x, y, z])
    coords_arr = np.array(coords, dtype=np.float32)
    if coords_arr.shape != (21, 3):
        raise LandmarkExtractionError(f"Expected 21 landmarks, extracted shape {coords_arr.shape}")
    return coords_arr

class FeatureExtractor158:
    """
    State-free, thread-safe extractor producing the immutable 158-feature representation.
    """
    
    @staticmethod
    def extract_from_hands(
        right_hand: Optional[Union[np.ndarray, List[Any]]] = None,
        left_hand: Optional[Union[np.ndarray, List[Any]]] = None
    ) -> np.ndarray:
        """
        Extracts the 158-dimensional feature vector from two hand landmark sets.
        Args:
            right_hand: 21 landmarks for Right Hand (user's right hand), or None if undetected.
            left_hand: 21 landmarks for Left Hand (user's left hand), or None if undetected.
        Returns:
            np.ndarray of shape (158,) and dtype np.float32.
        """
        feature_vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        raw_rh = _to_numpy_21x3(right_hand)
        raw_lh = _to_numpy_21x3(left_hand)
        
        is_rh_present = raw_rh is not None
        is_lh_present = raw_lh is not None
        
        # 1. Right Hand Features (Slots [0:73])
        if is_rh_present:
            norm_rh, _ = normalize_hand_landmarks(raw_rh)
            rh_distances = compute_finger_distances(norm_rh)
            rh_angles = compute_finger_angles(norm_rh)
            
            feature_vector[RIGHT_HAND_COORDS_SLICE] = norm_rh.flatten()
            feature_vector[RIGHT_HAND_DISTANCES_SLICE] = rh_distances
            feature_vector[RIGHT_HAND_ANGLES_SLICE] = rh_angles
            feature_vector[RIGHT_PRESENCE_INDEX] = 1.0
        else:
            feature_vector[RIGHT_PRESENCE_INDEX] = 0.0
            
        # 2. Left Hand Features (Slots [73:146])
        if is_lh_present:
            norm_lh, _ = normalize_hand_landmarks(raw_lh)
            lh_distances = compute_finger_distances(norm_lh)
            lh_angles = compute_finger_angles(norm_lh)
            
            feature_vector[LEFT_HAND_COORDS_SLICE] = norm_lh.flatten()
            feature_vector[LEFT_HAND_DISTANCES_SLICE] = lh_distances
            feature_vector[LEFT_HAND_ANGLES_SLICE] = lh_angles
            feature_vector[LEFT_PRESENCE_INDEX] = 1.0
        else:
            feature_vector[LEFT_PRESENCE_INDEX] = 0.0
            
        # 3. Inter-Hand Relational Features (Slots [148:158])
        if is_rh_present and is_lh_present:
            # Use raw uncentered landmarks in common camera space for inter-hand vectors
            wrist_r = raw_rh[WRIST_INDEX]
            wrist_l = raw_lh[WRIST_INDEX]
            delta_wrist = wrist_r - wrist_l
            dist_wrist = compute_euclidean_distance(wrist_r, wrist_l)
            
            index_r = raw_rh[INDEX_TIP_INDEX]
            index_l = raw_lh[INDEX_TIP_INDEX]
            delta_index = index_r - index_l
            dist_index = compute_euclidean_distance(index_r, index_l)
            
            thumb_r = raw_rh[THUMB_TIP_INDEX]
            thumb_l = raw_lh[THUMB_TIP_INDEX]
            dist_thumb = compute_euclidean_distance(thumb_r, thumb_l)
            
            # Smooth contact score
            min_dist = min(dist_index, dist_wrist, dist_thumb)
            contact_score = compute_contact_score(min_dist, sigma=0.05)
            
            feature_vector[INTER_WRIST_VECTOR_SLICE] = delta_wrist
            feature_vector[INTER_WRIST_DISTANCE_INDEX] = dist_wrist
            feature_vector[INTER_INDEX_VECTOR_SLICE] = delta_index
            feature_vector[INTER_INDEX_DISTANCE_INDEX] = dist_index
            feature_vector[INTER_THUMB_DISTANCE_INDEX] = dist_thumb
            feature_vector[CONTACT_SCORE_INDEX] = contact_score
            
        # Validate against schema invariants
        validate_schema_vector(feature_vector)
        return feature_vector

    @classmethod
    def extract_from_mediapipe_result(
        cls,
        mp_result: Any,
        include_facial: bool = False,
    ) -> np.ndarray:
        """
        Direct adapter for MediaPipe Tasks HolisticLandmarkerResult or HandLandmarkerResult.
        Maps detected hands to Right and Left slots correctly based on handedness labels.
        When include_facial=False (default): returns standard (158,) vector.
        When include_facial=True: appends 6-D facial ratios forming a (164,) vector.
        """
        rh = None
        lh = None

        # Case 1: HolisticLandmarkerResult (has left_hand_landmarks and right_hand_landmarks)
        if hasattr(mp_result, 'left_hand_landmarks') or hasattr(mp_result, 'right_hand_landmarks'):
            lh = getattr(mp_result, 'left_hand_landmarks', None)
            rh = getattr(mp_result, 'right_hand_landmarks', None)

        # Case 2: HandLandmarkerResult (has hand_landmarks list and handedness list)
        elif hasattr(mp_result, 'hand_landmarks') and hasattr(mp_result, 'handedness'):
            for idx, hand_cat in enumerate(mp_result.handedness):
                label = hand_cat[0].category_name if hand_cat else 'Right'
                lms = mp_result.hand_landmarks[idx]
                if label.lower() == 'right':
                    rh = lms
                elif label.lower() == 'left':
                    lh = lms
            # Fallback if handedness list was empty or couldn't categorize
            if rh is None and lh is None and len(mp_result.hand_landmarks) > 0:
                rh = mp_result.hand_landmarks[0]
                if len(mp_result.hand_landmarks) > 1:
                    lh = mp_result.hand_landmarks[1]

        # Case 3: Serialized or direct dictionary format
        elif isinstance(mp_result, dict):
            lh = mp_result.get('left_hand')
            rh = mp_result.get('right_hand')
            if rh is None and lh is None and 'hand_landmarks' in mp_result:
                h_lms = mp_result['hand_landmarks']
                if h_lms and len(h_lms) > 0:
                    rh = h_lms[0]
                if h_lms and len(h_lms) > 1:
                    lh = h_lms[1]

        # Case 4: Generic result with hand_landmarks attribute only
        elif hasattr(mp_result, 'hand_landmarks') and getattr(mp_result, 'hand_landmarks'):
            hands = getattr(mp_result, 'hand_landmarks')
            if len(hands) > 0:
                rh = hands[0]
            if len(hands) > 1:
                lh = hands[1]

        base_vec = cls.extract_from_hands(right_hand=rh, left_hand=lh)

        if not include_facial:
            return base_vec

        face_lms = getattr(mp_result, 'face_landmarks', None)
        facial_ratios = extract_facial_expression_features(face_lms)
        expanded_vec = np.concatenate([base_vec, np.array(facial_ratios, dtype=np.float32)])
        validate_schema_vector(expanded_vec)
        return expanded_vec

    @classmethod
    def extract_holistic_with_facial(cls, mp_result: Any) -> np.ndarray:
        """Extracts 164-D holistic representation including 6-D facial expression ratios."""
        return cls.extract_from_mediapipe_result(mp_result, include_facial=True)


def extract_facial_expression_features(face_landmarks: Any) -> List[float]:
    """
    Computes 6 scale-invariant facial expression and Non-Manual Marker (NMM) ratios
    from MediaPipe Face Mesh keypoints:
    1. Mouth Aspect Ratio (MAR): ||P_13 - P_14|| / max(||P_61 - P_291||, 1e-6)
    2. Left Eyebrow Elevation Ratio: ||P_105 - P_159|| / max(||P_10 - P_152||, 1e-6)
    3. Right Eyebrow Elevation Ratio: ||P_334 - P_386|| / max(||P_10 - P_152||, 1e-6)
    4. Mean Eyebrow Elevation Ratio (EER): mean(left_eer, right_eer)
    5. Left Eye Aspect Ratio (EAR): (||P_160 - P_144|| + ||P_158 - P_153||) / (2 * max(||P_33 - P_133||, 1e-6))
    6. Right Eye Aspect Ratio (EAR): (||P_385 - P_380|| + ||P_387 - P_373||) / (2 * max(||P_362 - P_263||, 1e-6))

    Returns:
        List of 6 floats. If face_landmarks is empty/missing, returns [0.0] * 6.
    """
    if face_landmarks is None:
        return [0.0] * 6

    # In case face_landmarks is wrapped in an outer list [[lm, ...]]
    lms = face_landmarks
    if isinstance(lms, (list, tuple)) and len(lms) > 0 and isinstance(lms[0], (list, tuple)):
        lms = lms[0]

    if not hasattr(lms, "__len__") or len(lms) < 388:
        return [0.0] * 6

    def get_pt(idx: int) -> np.ndarray:
        lm = lms[idx]
        if hasattr(lm, "x"):
            return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=np.float32)
        elif isinstance(lm, dict):
            return np.array([lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)], dtype=np.float32)
        return np.array(lm[:3], dtype=np.float32)

    try:
        def dist(i: int, j: int) -> float:
            return float(np.linalg.norm(get_pt(i) - get_pt(j)))

        # 1. Mouth Aspect Ratio (MAR)
        mouth_width = max(dist(61, 291), 1e-6)
        mar = dist(13, 14) / mouth_width

        # Face height for eyebrow normalization (Forehead top P_10 to Chin P_152)
        face_height = max(dist(10, 152), 1e-6)

        # 2. Left Eyebrow Elevation Ratio
        left_eer = dist(105, 159) / face_height

        # 3. Right Eyebrow Elevation Ratio
        right_eer = dist(334, 386) / face_height

        # 4. Mean Eyebrow Elevation Ratio (EER)
        mean_eer = (left_eer + right_eer) / 2.0

        # 5. Left Eye Aspect Ratio (EAR)
        left_ear = (dist(160, 144) + dist(158, 153)) / (2.0 * max(dist(33, 133), 1e-6))

        # 6. Right Eye Aspect Ratio (EAR)
        right_ear = (dist(385, 380) + dist(387, 373)) / (2.0 * max(dist(362, 263), 1e-6))

        ratios = [float(mar), float(left_eer), float(right_eer), float(mean_eer), float(left_ear), float(right_ear)]
        return [0.0 if (np.isnan(r) or np.isinf(r)) else r for r in ratios]
    except Exception:
        return [0.0] * 6


FeatureExtractor158.extract_facial_expression_features = staticmethod(extract_facial_expression_features)


def compute_velocity_features(seq: np.ndarray, target_dim: int = 328) -> np.ndarray:
    """
    Computes frame-to-frame velocity vectors (x_t - x_{t-1}) across the sequence window.
    Supports:
    - (T, 164) -> compute velocities -> (T, 328) [164 coords/ratios + 164 velocities]
    - (T, 158) -> pad 6 zero facial ratios to (T, 164) -> compute velocities -> (T, 328)
    - (T, 158) -> (T, 316) if target_dim == 316
    - (T, 316) -> padded to (T, 328) if target_dim == 328
    - (T, 328) -> returns as float32
    """
    if not isinstance(seq, np.ndarray) or seq.ndim != 2:
        raise ValueError(f"Expected 2D numpy array, got {getattr(seq, 'shape', type(seq))}")

    T, D = seq.shape
    if D == 328:
        return seq.astype(np.float32)

    if D == 164:
        vel = np.zeros_like(seq, dtype=np.float32)
        if T > 1:
            vel[1:] = seq[1:] - seq[:-1]
        return np.concatenate([seq.astype(np.float32), vel], axis=-1)

    if D == 158:
        if target_dim == 316:
            vel = np.zeros_like(seq, dtype=np.float32)
            if T > 1:
                vel[1:] = seq[1:] - seq[:-1]
            return np.concatenate([seq.astype(np.float32), vel], axis=-1)
        else:
            facial_pad = np.zeros((T, 6), dtype=np.float32)
            seq_164 = np.concatenate([seq.astype(np.float32), facial_pad], axis=-1)
            vel = np.zeros_like(seq_164, dtype=np.float32)
            if T > 1:
                vel[1:] = seq_164[1:] - seq_164[:-1]
            return np.concatenate([seq_164, vel], axis=-1)

    if D == 316 and target_dim == 328:
        coords = seq[:, :158]
        vel = seq[:, 158:316]
        pad = np.zeros((T, 6), dtype=np.float32)
        return np.concatenate([coords, pad, vel, pad], axis=-1)

    raise ValueError(f"Cannot compute velocity features for unexpected shape {seq.shape}")


def extract_realsign_hand_features(
    hand_landmarks: Optional[Union[np.ndarray, List[Any]]]
) -> Optional[np.ndarray]:
    """
    Dedicated wrist-centered feature extraction for RealSign 26-class alphabet model:
    1. Checks if hand landmarks (21 points) are present. If missing, returns None.
    2. Retrieves wrist position: wrist = hand_landmarks[0].
    3. Calculates palm scale: palm_size = max(distance(wrist, hand_landmarks[9]), 1e-6).
    4. Normalizes each landmark point: p_norm = (p - wrist) / palm_size.
    5. Flattens all 21 normalized 3D points into a 63-D float32 array (21 x 3).
    """
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return None

    # Handle wrapped hand list if a list of hands was provided
    target = hand_landmarks
    if hasattr(target, "landmark"):
        target = target.landmark

    if isinstance(target, (list, tuple)) and len(target) > 0:
        first_elem = target[0]
        if hasattr(first_elem, "landmark"):
            target = first_elem.landmark
        # If target[0] is itself a list/tuple/array of 21 landmarks, unwrap the first hand
        elif isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) == 21:
            target = first_elem

    try:
        pts = _to_numpy_21x3(target)
    except Exception:
        return None

    if pts is None or pts.shape != (21, 3):
        return None

    if not np.all(np.isfinite(pts)):
        return None

    wrist = pts[0]
    palm_size = float(np.linalg.norm(pts[9] - wrist))
    palm_size = max(palm_size, 1e-6)

    normalized = (pts - wrist) / palm_size
    return normalized.flatten().astype(np.float32)


# Attach to FeatureExtractor158 for convenience
FeatureExtractor158.extract_realsign_hand_features = staticmethod(extract_realsign_hand_features)

