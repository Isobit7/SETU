"""
SETU — ML Feature Extraction & Normalization
Extracts wrist-centered, scale- and rotation-invariant 63-D hand landmark vectors
from MediaPipe hand landmark results or raw point lists.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

# Total feature dimension for single-hand 21 3D landmarks
FEATURE_DIM_63: int = 63


def normalize_landmarks_to_63d(
    landmarks: Sequence[Union[Dict[str, float], Sequence[float]]],
    is_left_hand: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transforms 21 3D hand landmarks into a canonical wrist-centered,
    scale- and rotation-invariant 63-D vector.

    Steps:
      1. Origin translation to Wrist (landmark 0).
      2. Scale normalization by Euclidean distance from Wrist (0) to Middle MCP (9).
      3. In-plane rotation around Z axis aligning Middle MCP with (0, -1).
      4. Out-of-plane rotation around X axis aligning depth vector.
      5. Canonical anchoring: 0 = (0, 0, 0), 9 = (0, -1, 0).
      6. Flattens to (63,) float32 vector.
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
        pts[:, 0] = 1.0 - pts[:, 0]

    # 1. Translate so wrist (0) is at origin (0, 0, 0)
    wrist = pts[0].copy()
    translated = pts - wrist

    # 2. Scale by Euclidean distance from wrist (0) to middle finger MCP (9)
    v_wrist_middle = translated[9]
    scale = float(np.linalg.norm(v_wrist_middle))
    if scale < eps:
        scale = 1.0
    scaled = translated / scale

    # 3. In-plane rotation around Z axis so (x, y) vector of middle MCP aligns with (0, -1)
    v_x = float(scaled[9, 0])
    v_y = float(scaled[9, 1])
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

    # 4. Out-of-plane rotation around X axis so (y', z') aligns with (-1, 0)
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

    # Canonical exact alignment
    x_rot[0] = 0.0
    y_rot[0] = 0.0
    z_rot[0] = 0.0
    x_rot[9] = 0.0
    y_rot[9] = -1.0
    z_rot[9] = 0.0

    normalized = np.column_stack([x_rot, y_rot, z_rot]).astype(np.float32)
    return normalized.flatten()


def extract_realsign_hand_features(
    hand_landmarks_or_result: Optional[Union[np.ndarray, List[Any], Any]],
    dominant_hand: str = "right",
    out: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Dedicated wrist-centered scale and rotation invariant feature extraction
    for RealSign 26-class alphabet model with dominant hand isolation.
    """
    if hand_landmarks_or_result is None:
        return None

    if isinstance(dominant_hand, np.ndarray) and out is None:
        out = dominant_hand
        dominant_hand = "right"

    target = hand_landmarks_or_result
    is_left_hand = False
    dom_target = (dominant_hand or "right").lower()

    # Case 1: MediaPipe HandLandmarkerResult (has hand_landmarks and handedness)
    if hasattr(target, "hand_landmarks") and hasattr(target, "handedness"):
        hands = target.hand_landmarks
        cats = target.handedness
        if not hands or len(hands) == 0:
            return None

        hand_entries = []
        for i, h in enumerate(hands):
            label = "right"
            if cats and i < len(cats) and len(cats[i]) > 0:
                c = cats[i][0]
                label = getattr(c, "category_name", getattr(c, "displayName", "right")).lower()
            elif isinstance(cats, list) and i < len(cats) and isinstance(cats[i], dict):
                label = cats[i].get("category_name", cats[i].get("displayName", "right")).lower()
            hand_entries.append((label, h))

        dom_match = next((h for lbl, h in hand_entries if lbl == dom_target), None)
        if dom_match is not None:
            target = dom_match
            is_left_hand = (dom_target == "left")
        else:
            first_lbl, first_h = hand_entries[0]
            target = first_h
            is_left_hand = (first_lbl == "left")

    # Case 2: Multi-hand landmarks list
    elif isinstance(target, list) and len(target) > 0 and isinstance(target[0], list):
        target = target[0]

    # Convert landmarks into (21, 3) points
    pts = np.empty((21, 3), dtype=np.float32)
    if hasattr(target, "__iter__"):
        items = list(target)
        if len(items) < 21:
            return None
        for i in range(21):
            lm = items[i]
            if hasattr(lm, "x"):
                pts[i, 0] = float(lm.x)
                pts[i, 1] = float(lm.y)
                pts[i, 2] = float(getattr(lm, "z", 0.0))
            elif isinstance(lm, dict):
                pts[i, 0] = float(lm.get("x", 0.0))
                pts[i, 1] = float(lm.get("y", 0.0))
                pts[i, 2] = float(lm.get("z", 0.0))
            elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 3:
                pts[i, 0] = float(lm[0])
                pts[i, 1] = float(lm[1])
                pts[i, 2] = float(lm[2])
            else:
                return None
    elif isinstance(target, np.ndarray):
        if target.size == 63:
            pts = target.reshape(21, 3).astype(np.float32)
        elif target.ndim == 2 and target.shape[0] == 21 and target.shape[1] >= 3:
            pts = target[:, :3].astype(np.float32)
        else:
            return None
    else:
        return None

    features = normalize_landmarks_to_63d(pts, is_left_hand=is_left_hand)
    if out is not None and out.shape == (63,):
        out[:] = features
        return out
    return features
