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

class HolisticLandmarkerWrapper:
    """
    Wraps MediaPipe Tasks HolisticLandmarker for robust 3D keypoint detection.
    Validates model bundle existence and handles BGR to RGB frame transformation.
    """
    def __init__(
        self,
        task_model_path: Optional[Union[str, Path]] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None
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
        self._landmarker: Optional[mp.tasks.vision.HolisticLandmarker] = None

    def initialize(self) -> None:
        """Initializes the MediaPipe Tasks landmarker after verifying the model asset path."""
        if self._landmarker is not None:
            return

        if not self.task_model_path.exists():
            raise ModelArtifactError(
                f"MediaPipe task bundle not found at {self.task_model_path}. "
                "Ensure 'holistic_landmarker.task' is downloaded and located in 'models/tasks/' "
                "or specify a valid path via configuration."
            )

        try:
            # Create landmarker using MediaPipe Tasks API
            self._landmarker = mp.tasks.vision.HolisticLandmarker.create_from_model_path(
                str(self.task_model_path)
            )
        except Exception as e:
            raise ModelArtifactError(
                f"Failed to initialize MediaPipe HolisticLandmarker from {self.task_model_path}: {e}"
            ) from e

    def detect(self, bgr_image: np.ndarray) -> Any:
        """
        Executes landmark detection on a BGR image array.
        Returns:
            HolisticLandmarkerResult containing left_hand_landmarks, right_hand_landmarks,
            pose_landmarks, and face_landmarks.
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
            return self._landmarker.detect(mp_image)
        except Exception as e:
            raise LandmarkExtractionError(f"Error during MediaPipe landmark detection: {e}") from e

    def close(self) -> None:
        """Closes landmarker and frees underlying C++ runtime resources."""
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
