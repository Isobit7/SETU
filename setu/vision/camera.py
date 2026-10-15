"""
SETU - Camera Capture Module
Provides robust webcam frame acquisition with hardware checks, preview mirroring,
and explicit error handling via CameraError.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from setu.exceptions import CameraError
from configs.settings import config

class CameraReader:
    """
    Manages OpenCV video capture device lifecycle, error handling,
    and preview frame transformations.
    """
    def __init__(
        self,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mirror_preview: Optional[bool] = None
    ) -> None:
        self.camera_index = camera_index if camera_index is not None else config.vision.camera_index
        self.width = width if width is not None else config.vision.frame_width
        self.height = height if height is not None else config.vision.frame_height
        self.mirror_preview = mirror_preview if mirror_preview is not None else config.vision.mirror_user_preview
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Opens camera capture device and sets resolution properties."""
        if self._cap is not None and self._cap.isOpened():
            return

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Failed to open camera index {self.camera_index}. "
                "Ensure a webcam is connected, not in use by another application, "
                "or set SETU_CAMERA_INDEX environment variable."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads a frame from the camera.
        Returns:
            (success, raw_bgr_frame, preview_bgr_frame)
            raw_bgr_frame is un-mirrored for ML inference.
            preview_bgr_frame is horizontally flipped for natural user interaction.
        """
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Cannot read frame: Camera is not opened. Call open() first.")

        success, frame = self._cap.read()
        if not success or frame is None:
            return False, None, None

        raw_frame = frame
        if self.mirror_preview:
            preview_frame = cv2.flip(frame, 1)
        else:
            preview_frame = frame.copy()

        return True, raw_frame, preview_frame

    def release(self) -> None:
        """Safely releases the camera device."""
        if self._cap is not None:
            if self._cap.isOpened():
                self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
