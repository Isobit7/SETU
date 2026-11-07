"""
SETU — Non-Manual Markers (NMMs) Facial & Head Gesture Detector (PROB-20).

Extracts grammatical linguistic cues from face landmarks and head kinematics:
1. Eyebrow raise (delta_y > threshold): Interrogative marker ('?').
2. Head shake (yaw angular oscillation): Grammatical negation ('NOT').
3. Head nod (pitch vertical oscillation): Affirmative emphasis ('YES').
"""

from __future__ import annotations

import math
import time
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
    head_gesture: str = "neutral"  # "nod" (affirmation), "shake" (negation), or "neutral"


class NonManualMarkerDetector:
    """Detects grammatical non-manual markers and user intent gestures (nod/shake) from MediaPipe face landmarks."""

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

    def __init__(self, history_len: int = 18) -> None:
        self.history_len = history_len
        self.nose_x_history: List[float] = []
        self.nose_y_history: List[float] = []
        self.last_nod_time: float = 0.0
        self.last_shake_time: float = 0.0
        self.pitch_threshold: float = 0.035
        self.yaw_threshold: float = 0.035

    def reset(self) -> None:
        self.nose_x_history.clear()
        self.nose_y_history.clear()
        self.last_nod_time = 0.0
        self.last_shake_time = 0.0

    def detect(
        self,
        face_landmarks: Optional[Union[List[Any], np.ndarray]],
        baseline_eyebrow_ratio: float = 0.055,
    ) -> NonManualGrammar:
        """Evaluates facial blendshapes and head kinematics for grammatical cues and intent gestures."""
        if face_landmarks is None or len(face_landmarks) < 400:
            return NonManualGrammar(
                is_question=False,
                is_negation=False,
                is_affirmation=False,
                eyebrow_elevation=0.0,
                head_yaw_oscillation=0.0,
                head_pitch_oscillation=0.0,
                confidence=0.0,
                head_gesture="neutral",
            )

        now = time.time()

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
        left_dist = max(0.0, ly_eye - ly_brow)
        right_dist = max(0.0, ry_eye - ry_brow)
        avg_brow_elevation = (left_dist + right_dist) / 2.0
        is_question = bool(avg_brow_elevation > (baseline_eyebrow_ratio * 1.25))

        # Continuous live oscillation values for HUD and calibration feedback
        range_x = float(np.ptp(self.nose_x_history)) if len(self.nose_x_history) >= 4 else 0.0
        range_y = float(np.ptp(self.nose_y_history)) if len(self.nose_y_history) >= 4 else 0.0
        yaw_oscillation = range_x
        pitch_oscillation = range_y

        # 2. Head Yaw Oscillation (horizontal head shake -> Negation / Cancel)
        raw_negation = False
        if len(self.nose_x_history) >= 6:
            diffs_x = np.diff(self.nose_x_history)
            sign_changes_x = np.sum(diffs_x[:-1] * diffs_x[1:] < 0)
            if sign_changes_x >= 1 and range_x >= self.yaw_threshold:
                raw_negation = True

        # Debounced negation
        is_negation = False
        if raw_negation and (now - self.last_shake_time > 0.60):
            is_negation = True
            self.last_shake_time = now

        # 3. Head Pitch Oscillation (vertical nod -> Affirmation / Commit)
        raw_affirmation = False
        if len(self.nose_y_history) >= 6:
            diffs_y = np.diff(self.nose_y_history)
            sign_changes_y = np.sum(diffs_y[:-1] * diffs_y[1:] < 0)
            if sign_changes_y >= 1 and range_y >= self.pitch_threshold:
                raw_affirmation = True

        # Debounced affirmation (nod)
        is_affirmation = False
        if raw_affirmation and (now - self.last_nod_time > 0.60):
            is_affirmation = True
            self.last_nod_time = now

        # Resolve primary head gesture
        head_gesture = "neutral"
        if is_affirmation:
            head_gesture = "nod"
        elif is_negation:
            head_gesture = "shake"

        return NonManualGrammar(
            is_question=is_question,
            is_negation=is_negation,
            is_affirmation=is_affirmation,
            eyebrow_elevation=round(float(avg_brow_elevation), 4),
            head_yaw_oscillation=round(float(yaw_oscillation), 4),
            head_pitch_oscillation=round(float(pitch_oscillation), 4),
            confidence=0.90 if (is_question or is_negation or is_affirmation) else 0.50,
            head_gesture=head_gesture,
        )


# Singleton
non_manual_detector = NonManualMarkerDetector()

