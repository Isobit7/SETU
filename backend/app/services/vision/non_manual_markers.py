"""
SETU — Non-Manual Markers (NMMs) Facial & Head Gesture Detector (PROB-20).

Extracts grammatical linguistic cues from face landmarks and head kinematics:
1. Eyebrow raise (delta_y > threshold): Interrogative marker ('?').
2. Head shake (yaw angular oscillation): Grammatical negation ('NOT').
3. Head nod (pitch vertical oscillation): Affirmative emphasis ('YES').
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("setu.services.nmm")


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
    head_gesture: str = "none"  # "nod" | "shake" | "none"


class NonManualMarkerDetector:
    """Detects grammatical non-manual markers from 468/478 MediaPipe face landmarks."""

    # Key landmark indices in MediaPipe Face Mesh:
    # Forehead top: 10, Chin bottom: 152
    # Left cheek / ear: 234, Right cheek / ear: 454
    # Left eye top: 159, Left eyebrow mid: 105
    # Right eye top: 386, Right eyebrow mid: 334
    # Nose tip: 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_FACE = 234
    RIGHT_FACE = 454
    LEFT_EYE_TOP = 159
    LEFT_EYEBROW = 105
    RIGHT_EYE_TOP = 386
    RIGHT_EYEBROW = 334
    NOSE_TIP = 1

    def __init__(self, history_len: int = 18) -> None:
        self.history_len = history_len
        self.nose_x_history: List[float] = []
        self.nose_y_history: List[float] = []
        self.last_gesture_time: float = 0.0

    def reset(self) -> None:
        self.nose_x_history.clear()
        self.nose_y_history.clear()
        self.last_gesture_time = 0.0

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
                head_gesture="none",
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

        fh_x, fh_y = _get_xy(self.FOREHEAD)
        ch_x, ch_y = _get_xy(self.CHIN)
        lf_x, lf_y = _get_xy(self.LEFT_FACE)
        rf_x, rf_y = _get_xy(self.RIGHT_FACE)
        lx_eye, ly_eye = _get_xy(self.LEFT_EYE_TOP)
        lx_brow, ly_brow = _get_xy(self.LEFT_EYEBROW)
        rx_eye, ry_eye = _get_xy(self.RIGHT_EYE_TOP)
        rx_brow, ry_brow = _get_xy(self.RIGHT_EYEBROW)
        nx, ny = _get_xy(self.NOSE_TIP)

        # Scale-invariant normalization using face bounding geometry
        face_h = abs(ch_y - fh_y)
        face_w = abs(rf_x - lf_x)

        # In face-normalized coordinates, natural body drift/breathing is canceled out!
        rel_y = (ny - fh_y) / face_h if face_h >= 0.05 else ny
        rel_x = (nx - lf_x) / face_w if face_w >= 0.05 else nx

        # Update nose displacement history for head oscillations
        self.nose_x_history.append(rel_x)
        self.nose_y_history.append(rel_y)
        if len(self.nose_x_history) > self.history_len:
            self.nose_x_history.pop(0)
        if len(self.nose_y_history) > self.history_len:
            self.nose_y_history.pop(0)

        # 1. Eyebrow Elevation (negative Y is upward in normalized screen coordinates)
        left_dist = max(0.0, ly_eye - ly_brow)
        right_dist = max(0.0, ry_eye - ry_brow)
        avg_brow_elevation = (left_dist + right_dist) / 2.0
        is_question = bool(
            (avg_brow_elevation / face_h > 0.18) if face_h >= 0.05
            else (avg_brow_elevation > (baseline_eyebrow_ratio * 1.25))
        )

        # Kinematic Oscillation Analysis (requires at least 6 frames)
        pitch_oscillation = 0.0
        yaw_oscillation = 0.0
        is_affirmation = False
        is_negation = False

        if len(self.nose_y_history) >= 6 and len(self.nose_x_history) >= 6:
            ptp_y = float(np.ptp(self.nose_y_history))
            ptp_x = float(np.ptp(self.nose_x_history))

            diffs_y = np.diff(self.nose_y_history)
            sign_changes_y = int(np.sum(diffs_y[:-1] * diffs_y[1:] < 0))

            diffs_x = np.diff(self.nose_x_history)
            sign_changes_x = int(np.sum(diffs_x[:-1] * diffs_x[1:] < 0))

            # Calibrated thresholds: requires deliberate, larger intentional head movements
            min_ptp_y = 0.065 if face_h >= 0.05 else 0.035
            min_ptp_x = 0.070 if face_w >= 0.05 else 0.035

            # 2. Head Pitch Oscillation (vertical nod up-down -> Affirmation / Commit)
            if ptp_y >= min_ptp_y and ptp_y > (ptp_x * 1.25):
                if sign_changes_y >= 1:
                    pitch_oscillation = ptp_y
                    is_affirmation = True

            # 3. Head Yaw Oscillation (horizontal head shake -> Negation / Undo)
            if ptp_x >= min_ptp_x and ptp_x > (ptp_y * 1.25):
                if sign_changes_x >= 1:
                    yaw_oscillation = ptp_x
                    is_negation = True

        now = time.time()
        head_gesture = "none"
        if now - self.last_gesture_time >= 0.90:
            if is_affirmation and pitch_oscillation >= (yaw_oscillation * 1.15):
                head_gesture = "nod"
                self.last_gesture_time = now
                self.nose_y_history.clear()
                self.nose_x_history.clear()
                logger.info("HEAD GESTURE: DELIBERATE NOD DETECTED (pitch=%.4f, yaw=%.4f)", pitch_oscillation, yaw_oscillation)
            elif is_negation and yaw_oscillation >= (pitch_oscillation * 1.15):
                head_gesture = "shake"
                self.last_gesture_time = now
                self.nose_x_history.clear()
                self.nose_y_history.clear()
                logger.info("HEAD GESTURE: DELIBERATE SHAKE DETECTED (pitch=%.4f, yaw=%.4f)", pitch_oscillation, yaw_oscillation)

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

