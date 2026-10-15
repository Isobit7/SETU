"""
SETU - 158-Feature Extractor
Extracts wrist-centered, scale-invariant 3D coordinates, finger geometry,
and inter-hand relational features from MediaPipe hand landmarks.
"""

from typing import Optional, Union, List, Any
import numpy as np

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
    validate_schema_vector
)

def _to_numpy_21x3(landmarks: Optional[Union[np.ndarray, List[Any]]]) -> Optional[np.ndarray]:
    """Converts a list of landmark objects or array into a standard (21, 3) float32 array."""
    if landmarks is None or len(landmarks) == 0:
        return None
        
    if isinstance(landmarks, np.ndarray):
        if landmarks.shape == (21, 3):
            return landmarks.astype(np.float32)
        elif landmarks.shape == (21, 2):
            # Pad z with zeros if only 2D provided
            z = np.zeros((21, 1), dtype=np.float32)
            return np.hstack([landmarks, z]).astype(np.float32)
        else:
            raise ValueError(f"Expected array of shape (21, 3) or (21, 2), got {landmarks.shape}")
            
    # List of landmark objects with .x, .y, .z attributes
    coords = []
    for lm in landmarks:
        x = getattr(lm, 'x', 0.0)
        y = getattr(lm, 'y', 0.0)
        z = getattr(lm, 'z', 0.0)
        coords.append([x, y, z])
    coords_arr = np.array(coords, dtype=np.float32)
    if coords_arr.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks, extracted shape {coords_arr.shape}")
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
    def extract_from_mediapipe_result(cls, mp_result: Any) -> np.ndarray:
        """
        Direct adapter for MediaPipe Tasks HolisticLandmarkerResult or HandLandmarkerResult.
        Maps detected hands to Right and Left slots correctly based on handedness labels.
        """
        # Case 1: HolisticLandmarkerResult (has left_hand_landmarks and right_hand_landmarks)
        if hasattr(mp_result, 'left_hand_landmarks') or hasattr(mp_result, 'right_hand_landmarks'):
            lh = getattr(mp_result, 'left_hand_landmarks', None)
            rh = getattr(mp_result, 'right_hand_landmarks', None)
            return cls.extract_from_hands(right_hand=rh, left_hand=lh)
            
        # Case 2: HandLandmarkerResult (has hand_landmarks list and handedness list)
        if hasattr(mp_result, 'hand_landmarks') and hasattr(mp_result, 'handedness'):
            rh = None
            lh = None
            for idx, hand_cat in enumerate(mp_result.handedness):
                # hand_cat is a list of Category objects; first is highest score
                label = hand_cat[0].category_name if hand_cat else 'Right'
                lms = mp_result.hand_landmarks[idx]
                if label.lower() == 'right':
                    rh = lms
                elif label.lower() == 'left':
                    lh = lms
            return cls.extract_from_hands(right_hand=rh, left_hand=lh)
            
        return np.zeros(FEATURE_DIM, dtype=np.float32)
