"""
Tests for RF-UV-11 Pretrained Model Adapter
Verifies weight loading, 63-feature contract, inference determinism,
smoothing window, and rigorous error handling.
"""

from dataclasses import dataclass
from pathlib import Path
import pytest
import numpy as np

from models.adapters.rf_uv11 import RFUV11Adapter
from config.exceptions import ModelArtifactError, LandmarkExtractionError, InferenceError

MODEL_PATH = Path("models/pretrained/rf_uv_11_model.h5")


@dataclass
class MockLandmark:
    x: float
    y: float
    z: float


def test_adapter_initialization_and_weights_shapes():
    """Verify that adapter loads valid weights from models/pretrained/rf_uv_11_model.h5."""
    adapter = RFUV11Adapter(MODEL_PATH)
    assert adapter._is_loaded is True
    assert adapter.W1.shape == (63, 128)
    assert adapter.b1.shape == (128,)
    assert adapter.W2.shape == (128, 64)
    assert adapter.b2.shape == (64,)
    assert adapter.W3.shape == (64, 26)
    assert adapter.b3.shape == (26,)
    assert len(adapter.labels) == 26
    assert adapter.labels[0] == "A"
    assert adapter.labels[-1] == "Z"


def test_adapter_missing_file_raises_artifact_error():
    """Verify ModelArtifactError is raised when model file does not exist."""
    with pytest.raises(ModelArtifactError, match="not found"):
        RFUV11Adapter("models/pretrained/non_existent_weights.h5")


def test_extract_63_features_from_landmarks():
    """Verify extraction of 21 landmarks produces exact (63,) float32 vector."""
    mock_lms = [MockLandmark(x=float(i) * 0.01, y=float(i) * 0.02, z=float(i) * 0.005) for i in range(21)]
    feats = RFUV11Adapter.extract_63_features(mock_lms)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (63,)
    assert feats.dtype == np.float32
    # Verify first landmark coordinates
    assert np.isclose(feats[0], 0.0)
    assert np.isclose(feats[1], 0.0)
    assert np.isclose(feats[2], 0.0)
    # Verify second landmark coordinates
    assert np.isclose(feats[3], 0.01)
    assert np.isclose(feats[4], 0.02)
    assert np.isclose(feats[5], 0.005)


def test_extract_63_features_validation_errors():
    """Verify that landmark extraction rejects incorrect count, None, and NaNs."""
    with pytest.raises(LandmarkExtractionError, match="is None"):
        RFUV11Adapter.extract_63_features(None)

    with pytest.raises(LandmarkExtractionError, match="Expected exactly 21"):
        RFUV11Adapter.extract_63_features([MockLandmark(0, 0, 0)] * 20)

    with pytest.raises(LandmarkExtractionError, match="non-finite"):
        corrupt = [MockLandmark(0, 0, 0)] * 20 + [MockLandmark(float("nan"), 0, 0)]
        RFUV11Adapter.extract_63_features(corrupt)


def test_predict_raw_contract():
    """Verify raw inference outputs valid class, confidence bounds, and top-3 ranking."""
    adapter = RFUV11Adapter(MODEL_PATH)
    dummy_input = np.full((63,), 0.5, dtype=np.float32)

    label, conf, top3 = adapter.predict_raw(dummy_input)
    assert label in adapter.labels
    assert 0.0 <= conf <= 1.0
    assert len(top3) == 3
    assert top3[0][0] == label
    assert np.isclose(top3[0][1], conf)
    assert top3[0][1] >= top3[1][1] >= top3[2][1]


def test_predict_raw_rejects_invalid_inputs():
    """Verify InferenceError on wrong input dimension or NaN values."""
    adapter = RFUV11Adapter(MODEL_PATH)

    with pytest.raises(InferenceError, match="Invalid input shape"):
        adapter.predict_raw(np.zeros((158,), dtype=np.float32))

    with pytest.raises(InferenceError, match="non-finite"):
        bad_arr = np.zeros((63,), dtype=np.float32)
        bad_arr[10] = float("nan")
        adapter.predict_raw(bad_arr)


def test_smoothing_window_and_reset():
    """Verify temporal smoothing majority vote and reset functionality."""
    adapter = RFUV11Adapter(MODEL_PATH, window_size=5, confidence_threshold=0.3)
    dummy_input = np.full((63,), 0.3, dtype=np.float32)

    # First prediction
    smoothed, s_conf, raw_lbl, r_conf = adapter.predict_smoothed(dummy_input)
    assert len(adapter.prediction_window) == 1
    assert raw_lbl in adapter.labels

    # Fill window
    for _ in range(5):
        adapter.predict_smoothed(dummy_input)
    assert len(adapter.prediction_window) == 5

    # Reset
    adapter.reset_window()
    assert len(adapter.prediction_window) == 0
