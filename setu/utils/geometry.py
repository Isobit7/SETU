"""
SETU - Geometric Utility Functions
Pure mathematical functions for Euclidean distances, 3D joint angles,
wrist-centering, scale normalization, and smooth inter-hand contact scoring.
Re-exports canonical implementations from setu.features.geometry.
"""

from setu.features.geometry import (
    compute_euclidean_distance,
    compute_joint_angle,
    normalize_wrist_coordinates,
    normalize_hand_landmarks,
    compute_joint_angles_15,
    compute_finger_angles,
    compute_fingertip_to_wrist_distances,
    compute_fingertip_to_thumb_distances,
    compute_finger_distances,
    compute_pairwise_distances,
    extract_invariant_hand_features,
    compute_contact_score,
    get_finger_extension_states,
    evaluate_alphabet_heuristics,
)

__all__ = [
    "compute_euclidean_distance",
    "compute_joint_angle",
    "normalize_wrist_coordinates",
    "normalize_hand_landmarks",
    "compute_joint_angles_15",
    "compute_finger_angles",
    "compute_fingertip_to_wrist_distances",
    "compute_fingertip_to_thumb_distances",
    "compute_finger_distances",
    "compute_pairwise_distances",
    "extract_invariant_hand_features",
    "compute_contact_score",
    "get_finger_extension_states",
    "evaluate_alphabet_heuristics",
]
