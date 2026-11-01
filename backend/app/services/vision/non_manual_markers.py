"""
SETU — Non-Manual Markers (NMMs) Facial & Head Gesture Detector (PROB-20).

Extracts grammatical linguistic cues from face landmarks and head kinematics:
1. Eyebrow raise (delta_y > threshold): Interrogative marker ('?').
2. Head shake (yaw angular oscillation): Grammatical negation ('NOT').
3. Head nod (pitch vertical oscillation): Affirmative emphasis ('YES').
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class NonManualGrammar:
    """Grammatical operators inferred from facial expression and head movement."""
    is_question: bool
    is_negation: bool
    is_affirmation: bool
    eyebrow_elevation: float
    head_yaw_oscillation: float
    head_pitch_oscillation: float
    confidence: float


class NonManualMarkerDetector:
    """Detects grammatical non-manual markers from 468/478 MediaPipe face landmarks."""

    # Key landmark indices in MediaPipe Face Mesh:
    # Left eye top: 159, Left eyebrow mid: 105
    # Right eye top: 386, Right eyebrow mid: 334
    # Nose tip: 1, Chin: 152
    LEFT_EYE_TOP = 159
    LEFT_EYEBROW = 105
    RIGHT_EYE_TOP = 386
    RIGHT_EYEBROW = 334
    NOSE_TIP = 1
    CHIN = 152

    def __init__(self, history_len: int = 10) -> None:
        self.history_len = history_len
        self.nose_x_history: List[float] = []
        self.nose_y_history: List[float] = []

    def reset(self) -> None:
        self.nose_x_history.clear()
        self.nose_y_history.clear()

    def detect(
        self,
        face_landmarks: Optional[Union[List[Any], np.ndarray]],
        baseline_eyebrow_ratio: float = 0.055,
    ) -> NonManualGrammar:
        """Evaluates facial blendshapes and head kinematics for grammatical cues."""
        if face_landmarks is None or len(face_landmarks) < 400:
            return NonManualGrammar(
                is_question=False,
                is_negation=False,
                is_affirmation=False,
                eyebrow_elevation=0.0,
                head_yaw_oscillation=0.0,
                head_pitch_oscillation=0.0,
                confidence=0.0,
            )

        def _get_xy(idx: int) -> Tuple[float, float]:
            lm = face_landmarks[idx]
            if hasattr(lm, "x") and hasattr(lm, "y"):
                return float(lm.x), float(lm.y)
            elif isinstance(lm, dict):
                return float(lm.get("x", 0.0)), float(lm.get("y", 0.0))
            elif isinstance(lm, (list, tuple, np.ndarray)):
                return float(lm[0]), float(lm[1])
            return 0.0, 0.0

        lx_eye, ly_eye = _get_xy(self.LEFT_EYE_TOP)
        lx_brow, ly_brow = _get_xy(self.LEFT_EYEBROW)
        rx_eye, ry_eye = _get_xy(self.RIGHT_EYE_TOP)
        rx_brow, ry_brow = _get_xy(self.RIGHT_EYEBROW)
        nx, ny = _get_xy(self.NOSE_TIP)

        # Update nose displacement history for head oscillations
        self.nose_x_history.append(nx)
        self.nose_y_history.append(ny)
        if len(self.nose_x_history) > self.history_len:
            self.nose_x_history.pop(0)
            self.nose_y_history.pop(0)

        # 1. Eyebrow Elevation (negative Y is upward in normalized screen coordinates)
        # Distance from eye to eyebrow: (eye_y - brow_y)
        left_dist = max(0.0, ly_eye - ly_brow)
        right_dist = max(0.0, ry_eye - ry_brow)
        avg_brow_elevation = (left_dist + right_dist) / 2.0

        # Eyebrows significantly raised above neutral baseline -> Question marker
        is_question = bool(avg_brow_elevation > (baseline_eyebrow_ratio * 1.25))

        # 2. Head Yaw Oscillation (horizontal head shake -> Negation)
        yaw_oscillation = 0.0
        if len(self.nose_x_history) >= 6:
            # Measure directional reversals in X
            diffs = np.diff(self.nose_x_history)
            sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
            range_x = float(np.ptp(self.nose_x_history))
            if sign_changes >= 2 and range_x > 0.03:
                yaw_oscillation = range_x

        is_negation = bool(yaw_oscillation > 0.035)

        # 3. Head Pitch Oscillation (vertical nod -> Affirmation)
        pitch_oscillation = 0.0
        if len(self.nose_y_history) >= 6:
            diffs_y = np.diff(self.nose_y_history)
            sign_changes_y = np.sum(diffs_y[:-1] * diffs_y[1:] < 0)
            range_y = float(np.ptp(self.nose_y_history))
            if sign_changes_y >= 2 and range_y > 0.03:
                pitch_oscillation = range_y

        is_affirmation = bool(pitch_oscillation > 0.035)

        return NonManualGrammar(
            is_question=is_question,
            is_negation=is_negation,
            is_affirmation=is_affirmation,
            eyebrow_elevation=round(float(avg_brow_elevation), 4),
            head_yaw_oscillation=round(float(yaw_oscillation), 4),
            head_pitch_oscillation=round(float(pitch_oscillation), 4),
            confidence=0.90 if (is_question or is_negation or is_affirmation) else 0.50,
        )


# Singleton
non_manual_detector = NonManualMarkerDetector()
