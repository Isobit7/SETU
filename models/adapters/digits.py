"""
SETU — Indian Sign Language Digit (0–9) & Currency Classifier Adapter (PROB-18).

Extracts anatomical finger extension counts, joint angles, and relative tip contacts
to classify ISL numerical digits 0 through 9 from 21 3D hand landmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from training.features.geometry import (
    compute_joint_angles_15,
    compute_finger_spread_angles,
    normalize_wrist_coordinates,
)


@dataclass(frozen=True)
class DigitPrediction:
    """Represents an ISL numerical digit inference result."""
    digit: str
    confidence: float
    is_digit: bool
    probabilities: Dict[str, float]
    top_3: List[Tuple[str, float]]
    finger_count: int


class ISLDigitClassifier:
    """Classifies ISL numerical digits 0–9 from 21 canonical 3D hand landmarks."""

    DIGITS = [str(i) for i in range(10)]

    def predict_landmarks(
        self,
        landmarks_21x3: Union[np.ndarray, List[Dict[str, float]]],
    ) -> DigitPrediction:
        """Predicts digit 0–9 based on anatomical joint angles and finger extension states."""
        if landmarks_21x3 is None or len(landmarks_21x3) != 21:
            return self._empty_prediction()

        # Convert to numpy array
        if isinstance(landmarks_21x3, list):
            pts = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(landmarks_21x3):
                if isinstance(lm, dict):
                    pts[i, 0] = float(lm.get("x", 0.0))
                    pts[i, 1] = float(lm.get("y", 0.0))
                    pts[i, 2] = float(lm.get("z", 0.0))
                elif isinstance(lm, (list, tuple, np.ndarray)):
                    pts[i, :len(lm)] = lm[:3]
        else:
            pts = np.array(landmarks_21x3, dtype=np.float32)

        # Canonical normalization
        norm_pts, scale = normalize_wrist_coordinates(pts)
        if scale < 1e-4:
            return self._empty_prediction()

        # Compute 15 joint angles (normalized in [0.0, 1.0], where 1.0 = straight ~180 deg)
        angles_15 = compute_joint_angles_15(norm_pts)
        spread_angles = compute_finger_spread_angles(norm_pts)

        # Detect extension state for each of the 5 fingers (1.0 = extended, 0.0 = folded)
        # Thumb: angles 0, 1, 2
        # Index: angles 3, 4, 5 (MCP, PIP, DIP)
        # Middle: angles 6, 7, 8
        # Ring: angles 9, 10, 11
        # Pinky: angles 12, 13, 14
        thumb_ext = bool(angles_15[1] > 0.78 and angles_15[2] > 0.78)
        index_ext = bool(angles_15[4] > 0.82 and angles_15[5] > 0.82)
        middle_ext = bool(angles_15[7] > 0.82 and angles_15[8] > 0.82)
        ring_ext = bool(angles_15[10] > 0.82 and angles_15[11] > 0.82)
        pinky_ext = bool(angles_15[13] > 0.82 and angles_15[14] > 0.82)

        extended_fingers = [index_ext, middle_ext, ring_ext, pinky_ext]
        ext_count = sum(extended_fingers)

        # Tip to thumb distances
        thumb_tip = norm_pts[4]
        index_tip = norm_pts[8]
        middle_tip = norm_pts[12]

        d_thumb_index = float(np.linalg.norm(thumb_tip - index_tip))
        d_thumb_middle = float(np.linalg.norm(thumb_tip - middle_tip))

        probs: Dict[str, float] = {d: 0.02 for d in self.DIGITS}

        predicted_digit: str = "0"
        conf: float = 0.50

        # Anatomical ISL digit classification rules:
        if ext_count == 0:
            if d_thumb_index < 0.15:
                predicted_digit = "0"
                conf = 0.94
            else:
                # Closed fist
                predicted_digit = "0"
                conf = 0.88
        elif ext_count == 1:
            if index_ext:
                predicted_digit = "1"
                conf = 0.95
            elif pinky_ext and thumb_ext:
                predicted_digit = "6"
                conf = 0.90
            else:
                predicted_digit = "1"
                conf = 0.70
        elif ext_count == 2:
            if index_ext and middle_ext:
                predicted_digit = "2"
                conf = 0.96
            elif index_ext and thumb_ext:
                predicted_digit = "2"  # Variation
                conf = 0.85
            elif pinky_ext and thumb_ext:
                predicted_digit = "6"
                conf = 0.88
            else:
                predicted_digit = "2"
                conf = 0.75
        elif ext_count == 3:
            if index_ext and middle_ext and ring_ext:
                predicted_digit = "3"
                conf = 0.95
            elif thumb_ext and index_ext and middle_ext:
                predicted_digit = "3"
                conf = 0.93
            elif middle_ext and ring_ext and pinky_ext and d_thumb_index < 0.15:
                # 9 in ISL (F-shape: thumb + index touch, other 3 extended)
                predicted_digit = "9"
                conf = 0.94
            elif index_ext and ring_ext and pinky_ext and d_thumb_middle < 0.15:
                # 8 in ISL (thumb + middle touch)
                predicted_digit = "8"
                conf = 0.92
            else:
                predicted_digit = "3"
                conf = 0.80
        elif ext_count == 4:
            if not thumb_ext:
                predicted_digit = "4"
                conf = 0.96
            else:
                # 4 extended + thumb = 5
                predicted_digit = "5"
                conf = 0.92
        elif ext_count >= 4 and thumb_ext:
            predicted_digit = "5"
            conf = 0.98

        probs[predicted_digit] = conf
        # Normalize probabilities
        tot = sum(probs.values())
        norm_probs = {k: round(v / tot, 4) for k, v in probs.items()}

        sorted_top = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        return DigitPrediction(
            digit=predicted_digit,
            confidence=round(conf, 4),
            is_digit=True,
            probabilities=norm_probs,
            top_3=sorted_top,
            finger_count=ext_count + (1 if thumb_ext else 0),
        )

    def _empty_prediction(self) -> DigitPrediction:
        probs = {d: 0.10 for d in self.DIGITS}
        return DigitPrediction(
            digit="0",
            confidence=0.0,
            is_digit=False,
            probabilities=probs,
            top_3=[("0", 0.0), ("1", 0.0), ("2", 0.0)],
            finger_count=0,
        )


# Singleton
digit_classifier = ISLDigitClassifier()
