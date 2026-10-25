"""
Tests for SETU Vision Subsystem
Tests CameraReader error handling and HolisticLandmarkerWrapper integration.
"""
from pathlib import Path
import numpy as np
import pytest

from backend.app.services.vision import CameraReader
from backend.app.services.vision import HolisticLandmarkerWrapper
from training.feature_extractor import FeatureExtractor158
from config.exceptions import CameraError, ModelArtifactError, LandmarkExtractionError

def test_camera_reader_not_opened():
    """Verify CameraReader raises CameraError when read_frame is called before open()."""
    cam = CameraReader(camera_index=99)
    with pytest.raises(CameraError, match="Camera is not opened"):
        cam.read_frame()

def test_holistic_landmarker_missing_model():
    """Verify HolisticLandmarkerWrapper raises ModelArtifactError if task bundle is missing."""
    wrapper = HolisticLandmarkerWrapper(task_model_path="non_existent_path.task")
    with pytest.raises(ModelArtifactError, match="MediaPipe task bundle not found"):
        wrapper.initialize()

def test_holistic_landmarker_detect_invalid_image():
    """Verify detect raises LandmarkExtractionError when passed non-3D array."""
    wrapper = HolisticLandmarkerWrapper()
    # Mocking initialization or using dummy array
    with pytest.raises(LandmarkExtractionError):
        wrapper.detect(np.zeros((100, 100), dtype=np.uint8))  # 2D instead of 3D

def test_holistic_landmarker_pipeline_with_local_task():
    """
    If models/tasks/holistic_landmarker.task is present, verify full pipeline:
    Synthetic frame -> MediaPipe HolisticLandmarker -> FeatureExtractor158 -> valid 158-D vector.
    """
    task_path = Path("models/tasks/holistic_landmarker.task")
    if not task_path.exists():
        pytest.skip("Local holistic_landmarker.task not available in test environment")

    with HolisticLandmarkerWrapper(task_model_path=task_path) as landmarker:
        # Create a synthetic black frame (480, 640, 3)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_result = landmarker.detect(blank_frame)
        
        # Pass result directly to FeatureExtractor158
        feat_vector = FeatureExtractor158.extract_from_mediapipe_result(mp_result)
        
        assert feat_vector.shape == (158,)
        assert feat_vector.dtype == np.float32
        # On a blank frame with no hands, all features should be zero
        assert np.all(feat_vector == 0.0)
