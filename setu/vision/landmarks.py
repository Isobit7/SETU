"""
SETU - MediaPipe Holistic Landmarks Wrapper
Provides an explicit, validated wrapper around Google MediaPipe Tasks HolisticLandmarker
with comprehensive artifact checks and clear exception handling.
"""

from pathlib import Path
from typing import Optional, Union, Any
import cv2
import numpy as np
import mediapipe as mp

from setu.exceptions import ModelArtifactError, LandmarkExtractionError
from configs.settings import config

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
        """Applies exponential moving average coordinate smoothing to minimize frame-to-frame joint jitter."""
        if not landmarks:
            self._prev_landmarks[key] = None
            return

        prev = self._prev_landmarks.get(key)
        if prev is not None and len(prev) == len(landmarks):
            for i, lm in enumerate(landmarks):
                lm.x = float(alpha * lm.x + (1.0 - alpha) * prev[i][0])
                lm.y = float(alpha * lm.y + (1.0 - alpha) * prev[i][1])
                lm.z = float(alpha * lm.z + (1.0 - alpha) * prev[i][2])

        self._prev_landmarks[key] = [(lm.x, lm.y, lm.z) for lm in landmarks]

    def reset_smoothing(self) -> None:
        """Resets smoothing history to clear temporal state."""
        self._prev_landmarks = {
            "pose": None,
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }

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

            import os
            try:
                asset_path_str = os.path.relpath(self.task_model_path)
            except ValueError:
                asset_path_str = str(self.task_model_path)
            base_opts = python.BaseOptions(model_asset_path=asset_path_str)
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
                while len(self._prev_hands) < len(hands):
                    self._prev_hands.append(None)
                alpha = 0.70
                for h_idx, hand in enumerate(hands):
                    prev = self._prev_hands[h_idx]
                    if prev is not None and len(prev) == len(hand):
                        for i, lm in enumerate(hand):
                            lm.x = float(alpha * lm.x + (1.0 - alpha) * prev[i][0])
                            lm.y = float(alpha * lm.y + (1.0 - alpha) * prev[i][1])
                            lm.z = float(alpha * lm.z + (1.0 - alpha) * prev[i][2])
                    self._prev_hands[h_idx] = [(lm.x, lm.y, lm.z) for lm in hand]
            elif not (result and getattr(result, "hand_landmarks", None)):
                self._prev_hands.clear()
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

