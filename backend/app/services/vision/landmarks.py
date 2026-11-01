"""
SETU - MediaPipe Holistic Landmarks Wrapper
Provides an explicit, validated wrapper around Google MediaPipe Tasks HolisticLandmarker
with comprehensive artifact checks and clear exception handling.
"""

import math
from pathlib import Path
import time
from typing import Optional, Union, Any
import cv2
import numpy as np
import mediapipe as mp

from config.exceptions import ModelArtifactError, LandmarkExtractionError
from config.settings import config
from backend.app.services.vision.one_euro_filter import OneEuroFilter

class NormalizedLandmarkPoint:
    """Lightweight mutable landmark point compatible with MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"NormalizedLandmark(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


class HolisticLandmarkerResult:
    """Standardized result container compatible with MediaPipe Tasks HolisticLandmarkerResult."""
    __slots__ = (
        "pose_landmarks",
        "face_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",
        "pose_world_landmarks",
        "left_hand_world_landmarks",
        "right_hand_world_landmarks",
        "face_blendshapes",
        "segmentation_mask",
    )

    def __init__(
        self,
        pose_landmarks: Optional[list[NormalizedLandmarkPoint]] = None,
        face_landmarks: Optional[list[NormalizedLandmarkPoint]] = None,
        left_hand_landmarks: Optional[list[NormalizedLandmarkPoint]] = None,
        right_hand_landmarks: Optional[list[NormalizedLandmarkPoint]] = None,
    ) -> None:
        self.pose_landmarks = pose_landmarks or []
        self.face_landmarks = face_landmarks or []
        self.left_hand_landmarks = left_hand_landmarks or []
        self.right_hand_landmarks = right_hand_landmarks or []
        self.pose_world_landmarks = []
        self.left_hand_world_landmarks = []
        self.right_hand_world_landmarks = []
        self.face_blendshapes = None
        self.segmentation_mask = None


class HolisticLandmarkerWrapper:
    """
    Wraps MediaPipe Holistic for robust 3D keypoint detection across face, pose, and hands.
    Validates model bundle existence, handles BGR to RGB frame transformation, and isolates
    underlying C++ runtime resources safely.
    """
    def __init__(
        self,
        task_model_path: Optional[Union[str, Path]] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
    ) -> None:
        raw_path = task_model_path if task_model_path is not None else config.vision.task_model_path
        self.task_model_path = Path(raw_path)
        self.min_detection_confidence = (
            min_detection_confidence if min_detection_confidence is not None
            else config.vision.min_detection_confidence
        )
        self.min_tracking_confidence = (
            min_tracking_confidence if min_tracking_confidence is not None
            else config.vision.min_tracking_confidence
        )
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self._landmarker: Optional[Any] = None
        self._prev_landmarks: dict[str, Optional[list[tuple[float, float, float]]]] = {
            "pose": None,
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }
        self._filters: dict[str, OneEuroFilter] = {
            "pose": OneEuroFilter(min_cutoff=0.8, beta=0.01),
            "face": OneEuroFilter(min_cutoff=0.8, beta=0.01),
            "left_hand": OneEuroFilter(min_cutoff=0.8, beta=0.01),
            "right_hand": OneEuroFilter(min_cutoff=0.8, beta=0.01),
        }

    def initialize(self) -> None:
        """Initializes the MediaPipe Holistic landmarker after verifying the model asset path."""
        if self._landmarker is not None:
            return

        if not self.task_model_path.exists():
            raise ModelArtifactError(
                f"MediaPipe task bundle not found at {self.task_model_path}. "
                "Ensure 'holistic_landmarker.task' is downloaded and located in 'models/tasks/' "
                "or specify a valid path via configuration."
            )

        try:
            self._landmarker = mp.solutions.holistic.Holistic(
                min_detection_confidence=float(self.min_detection_confidence),
                min_tracking_confidence=float(self.min_tracking_confidence),
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
            )
        except Exception as e:
            raise ModelArtifactError(
                f"Failed to initialize MediaPipe Holistic landmarker: {e}"
            ) from e

    def _smooth_landmark_list(
        self,
        key: str,
        landmarks: Optional[list[Any]],
        alpha: float = 0.70,
    ) -> None:
        """Applies adaptive speed-based One-Euro low-pass filtering to eliminate stationary jitter while preventing motion lag."""
        if not landmarks:
            self._prev_landmarks[key] = None
            if key in self._filters:
                self._filters[key].reset()
            return

        filt = self._filters.get(key)
        if filt is None:
            filt = OneEuroFilter(min_cutoff=0.8, beta=0.01)
            self._filters[key] = filt

        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        filtered = filt.filter(coords)
        for i, lm in enumerate(landmarks):
            lm.x = float(filtered[i, 0])
            lm.y = float(filtered[i, 1])
            lm.z = float(filtered[i, 2])
        self._prev_landmarks[key] = [(lm.x, lm.y, lm.z) for lm in landmarks]

    def reset_smoothing(self) -> None:
        """Resets smoothing history to clear temporal state."""
        self._prev_landmarks = {
            "pose": None,
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }
        for f in self._filters.values():
            f.reset()

    def detect(self, bgr_image: np.ndarray) -> HolisticLandmarkerResult:
        """
        Executes landmark detection on a BGR image array.
        Returns:
            HolisticLandmarkerResult containing left_hand_landmarks, right_hand_landmarks,
            pose_landmarks, and face_landmarks as NormalizedLandmarkPoint lists.
        """
        if self._landmarker is None:
            self.initialize()

        if not isinstance(bgr_image, np.ndarray) or bgr_image.ndim != 3:
            raise LandmarkExtractionError(
                f"Expected 3-channel image array (H, W, C), got {type(bgr_image)} with shape {getattr(bgr_image, 'shape', None)}"
            )

        try:
            rgb_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            raw = self._landmarker.process(rgb_frame)

            def _to_points(lms_proto) -> list[NormalizedLandmarkPoint]:
                if not lms_proto or not hasattr(lms_proto, "landmark"):
                    return []
                return [
                    NormalizedLandmarkPoint(lm.x, lm.y, getattr(lm, "z", 0.0))
                    for lm in lms_proto.landmark
                ]

            pose = _to_points(raw.pose_landmarks)
            face = _to_points(raw.face_landmarks)
            left_hand = _to_points(raw.left_hand_landmarks)
            right_hand = _to_points(raw.right_hand_landmarks)

            if self.smooth_landmarks:
                self._smooth_landmark_list("pose", pose)
                self._smooth_landmark_list("face", face)
                self._smooth_landmark_list("left_hand", left_hand)
                self._smooth_landmark_list("right_hand", right_hand)

            return HolisticLandmarkerResult(
                pose_landmarks=pose,
                face_landmarks=face,
                left_hand_landmarks=left_hand,
                right_hand_landmarks=right_hand,
            )
        except Exception as e:
            raise LandmarkExtractionError(f"Error during MediaPipe landmark detection: {e}") from e

    def close(self) -> None:
        """Closes landmarker and frees underlying runtime resources."""
        self.reset_smoothing()
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None

    def __enter__(self) -> "HolisticLandmarkerWrapper":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class HandLandmarkerWrapper:
    """
    Wraps MediaPipe Tasks HandLandmarker for robust 21-joint 3D hand keypoint detection.
    Optimized for isolated hand signing and 26-class alphabet gestures.
    """

    def __init__(
        self,
        task_model_path: Optional[Union[str, Path]] = None,
        num_hands: int = 2,
        min_detection_confidence: float = 0.3,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
    ) -> None:
        self.task_model_path = Path(task_model_path or "models/tasks/hand_landmarker.task")
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self._landmarker: Optional[Any] = None
        self._prev_hands: list[Optional[list[tuple[float, float, float]]]] = []
        self._hand_filters: list[OneEuroFilter] = [
            OneEuroFilter(min_cutoff=0.8, beta=0.01) for _ in range(self.num_hands)
        ]

    def initialize(self) -> None:
        """Initializes the MediaPipe Tasks HandLandmarker."""
        if self._landmarker is not None:
            return

        if not self.task_model_path.exists():
            raise ModelArtifactError(
                f"MediaPipe hand landmarker task bundle not found at {self.task_model_path}. "
                "Ensure 'hand_landmarker.task' is located in 'models/tasks/'."
            )

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_model_bytes = self.task_model_path.read_bytes()
            base_opts = python.BaseOptions(model_asset_buffer=task_model_bytes)
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(opts)
        except Exception as e:
            raise ModelArtifactError(
                f"Failed to initialize MediaPipe HandLandmarker from {self.task_model_path}: {e}"
            ) from e

    def reset_smoothing(self) -> None:
        """Resets smoothing history to clear temporal state."""
        self._prev_hands.clear()
        for f in self._hand_filters:
            f.reset()

    def detect(self, bgr_image: np.ndarray) -> Any:
        """
        Executes hand landmark detection on a BGR image array.
        Returns HandLandmarkerResult containing hand_landmarks and handedness.
        """
        if self._landmarker is None:
            self.initialize()

        if not isinstance(bgr_image, np.ndarray) or bgr_image.ndim != 3:
            raise LandmarkExtractionError(
                f"Expected 3-channel image array (H, W, C), got {type(bgr_image)} with shape {getattr(bgr_image, 'shape', None)}"
            )

        try:
            rgb_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect(mp_image)
            if self.smooth_landmarks and result is not None and getattr(result, "hand_landmarks", None):
                hands = result.hand_landmarks
                while len(self._hand_filters) < len(hands):
                    self._hand_filters.append(OneEuroFilter(min_cutoff=0.8, beta=0.01))
                for h_idx, hand in enumerate(hands):
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
                    filtered = self._hand_filters[h_idx].filter(coords)
                    for i, lm in enumerate(hand):
                        lm.x = float(filtered[i, 0])
                        lm.y = float(filtered[i, 1])
                        lm.z = float(filtered[i, 2])
            elif not (result and getattr(result, "hand_landmarks", None)):
                for f in self._hand_filters:
                    f.reset()
            return result
        except Exception as e:
            raise LandmarkExtractionError(f"Error during HandLandmarker detection: {e}") from e

    def close(self) -> None:
        """Closes HandLandmarker and releases underlying C++ runtime resources."""
        self.reset_smoothing()
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None

    def __enter__(self) -> "HandLandmarkerWrapper":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()



class HandednessTracker:
    """
    Spatial-temporal handedness tracker with hysteresis for live video streams.
    Prevents single-frame label flapping (Left <-> Right) from causing sudden
    sagittal reflection and prediction flickering when signing near the center.
    """

    def __init__(self, hysteresis_frames: int = 4, max_distance: float = 0.25) -> None:
        self.hysteresis_frames = hysteresis_frames
        self.max_distance = max_distance
        self.stable_label: Optional[str] = None
        self.last_wrist: Optional[tuple[float, float]] = None
        self.flip_streak: int = 0
        self.last_timestamp: float = 0.0

    def reset(self) -> None:
        """Resets tracking memory on buffer clear or long idle."""
        self.stable_label = None
        self.last_wrist = None
        self.flip_streak = 0
        self.last_timestamp = 0.0

    def stabilize_single_hand(
        self,
        reported_label: str,
        landmarks: Any,
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Stabilizes the handedness label ('left' or 'right') for a single detected hand.
        If the hand is spatially contiguous with the previously tracked hand,
        requires `hysteresis_frames` consecutive opposite frames to change label.
        """
        now = time.time() if timestamp is None else timestamp
        if not landmarks or len(landmarks) == 0:
            if now - self.last_timestamp > 0.40:
                self.reset()
            return reported_label

        # Wrist is landmark 0
        lm0 = landmarks[0]
        if hasattr(lm0, "x") and hasattr(lm0, "y"):
            wx, wy = float(lm0.x), float(lm0.y)
        elif isinstance(lm0, dict):
            wx, wy = float(lm0.get("x", 0.0)), float(lm0.get("y", 0.0))
        elif isinstance(lm0, (list, tuple, np.ndarray)) and len(lm0) >= 2:
            wx, wy = float(lm0[0]), float(lm0[1])
        else:
            wx, wy = 0.0, 0.0

        time_gap = now - self.last_timestamp if self.last_timestamp > 0.0 else 0.0
        self.last_timestamp = now
        norm_label = reported_label.lower().strip()
        if norm_label not in ("left", "right"):
            norm_label = "right"

        # Cold start or hand absent for > 0.40s
        if self.stable_label is None or self.last_wrist is None or time_gap > 0.40:
            self.stable_label = norm_label
            self.last_wrist = (wx, wy)
            self.flip_streak = 0
            return norm_label

        # Spatial displacement from last tracked wrist
        pw_x, pw_y = self.last_wrist
        dist = math.hypot(wx - pw_x, wy - pw_y)

        # Hand jumped across screen (user switched physical hands or entered from other side)
        if dist > self.max_distance:
            self.stable_label = norm_label
            self.last_wrist = (wx, wy)
            self.flip_streak = 0
            return norm_label

        # Hand is in the same spatial vicinity (same physical hand)
        self.last_wrist = (wx, wy)
        if norm_label == self.stable_label:
            self.flip_streak = 0
            return self.stable_label

        # MediaPipe gave opposite label: increment flip streak
        self.flip_streak += 1
        if self.flip_streak >= self.hysteresis_frames:
            # Confirmed deliberate switch
            self.stable_label = norm_label
            self.flip_streak = 0
            return self.stable_label
        else:
            # Suppress single-frame flapping! Retain previous stable label
            return self.stable_label
