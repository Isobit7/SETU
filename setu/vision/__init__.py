"""
SETU - Vision Processing Subsystem
Provides camera video capture and MediaPipe Tasks keypoint detection wrappers.
"""

from setu.vision.camera import CameraReader
from setu.vision.landmarks import HolisticLandmarkerWrapper

__all__ = ["CameraReader", "HolisticLandmarkerWrapper"]
