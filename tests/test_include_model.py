"""Tests for AI4Bharat INCLUDE ISL pretrained model, feature extractor, and API endpoints."""

from pathlib import Path
import json
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pipeline_manager import pipeline
from models.adapters.include import (
    INCLUDEInferenceService,
    INCLUDELstmConfig,
    INCLUDELstmModel,
    INCLUDEResult,
    extract_include_134_features,
    interp_channel,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_include_model_instantiation_and_forward():
    config = INCLUDELstmConfig(input_size=134, hidden_size=256, num_layers=5, bidirectional=True)
    model = INCLUDELstmModel(config, n_classes=263)
    model.eval()

    dummy_input = torch.randn(2, 50, 134)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 263)
    assert not torch.isnan(output).any()


def test_checkpoint_loading_and_state_dict():
    ckpt_path = Path("models/pretrained/include_no_cnn_lstm.pth")
    assert ckpt_path.exists(), "Official INCLUDE checkpoint file is missing."

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "score" in ckpt
    assert abs(ckpt["score"] - 0.5624) < 0.01

    state_dict = ckpt["model"]
    assert len(state_dict) == 42

    model = INCLUDELstmModel(INCLUDELstmConfig(), n_classes=263)
    # Must load with strict=True
    model.load_state_dict(state_dict, strict=True)


def test_label_map_integrity_263_classes():
    label_map_path = Path("models/pretrained/label_map_include.json")
    assert label_map_path.exists(), "Official INCLUDE label map is missing."

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    assert len(label_map) == 263
    indices = sorted(label_map.values())
    assert indices == list(range(263)), "Label map must cover indices 0 to 262 contiguously."

    # Verify key benchmark classes
    expected_classes = ["thankyou", "hello", "goodmorning", "i", "you"]
    for cls_name in expected_classes:
        assert cls_name in label_map, f"Expected class {cls_name} not found in INCLUDE label map."


def test_interp_channel_linear_interpolation():
    # Test interpolation of NaNs with scaling
    arr = np.array([
        [np.nan, 10.0],
        [2.0, np.nan],
        [4.0, 30.0],
        [np.nan, np.nan],
    ], dtype=np.float32)

    res = interp_channel(arr, scale=2.0)
    assert res.shape == (4, 2)
    assert not np.isnan(res).any()
    # First column: [2.0, 2.0, 4.0, 4.0] * 2 = [4.0, 4.0, 8.0, 8.0]
    np.testing.assert_allclose(res[:, 0], [4.0, 4.0, 8.0, 8.0])


def test_feature_dimension_validation_rejects_158d():
    service = INCLUDEInferenceService.get_instance()
    service.load()

    # Pass incompatible 158-D array (SETU MVP pipeline format)
    incompatible_158 = np.zeros((30, 158), dtype=np.float32)
    with pytest.raises(ValueError, match="expects 134 features"):
        service.predict_features(incompatible_158)

    # Pass compatible 134-D array
    compatible_134 = np.zeros((30, 134), dtype=np.float32)
    res = service.predict_features(compatible_134)
    assert isinstance(res, INCLUDEResult)
    assert res.feature_dimension == 134
    assert len(res.top5_candidates) == 5


def test_genuine_inference_on_sample_video():
    video_path = Path("data/raw/include/thank_you_s04_mvi_0006.mov")
    if not video_path.exists():
        pytest.skip(f"Test video {video_path} not found.")

    pipeline.load()
    service = INCLUDEInferenceService.get_instance()
    service.load()

    res = service.predict_video(video_path, pipeline.landmarker)

    assert res.label == "thankyou"
    assert res.display_name == "THANK YOU"
    assert res.confidence > 0.20
    assert res.feature_dimension == 134
    top5_labels = [c.label for c in res.top5_candidates]
    assert "thankyou" in top5_labels


def test_api_include_status_endpoint(client):
    res = client.get("/api/v1/models/include/status")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "AI4Bharat-INCLUDE-263-LSTM"
    assert data["is_ready"] is True
    assert data["classes_count"] == 263
    assert data["input_features"] == 134
    assert data["sequence_length"] == 200
    assert data["checkpoint_exists"] is True
    assert data["label_map_exists"] is True


def test_api_include_classes_endpoint(client):
    res = client.get("/api/v1/models/include/classes")
    assert res.status_code == 200
    data = res.json()
    assert data["total_classes"] == 263
    assert len(data["classes"]) == 263
    assert data["classes"][0]["index"] == 0


def test_api_recognize_video_endpoint(client):
    video_path = Path("data/raw/include/thank_you_s04_mvi_0006.mov")
    if not video_path.exists():
        pytest.skip(f"Test video {video_path} not found.")

    with open(video_path, "rb") as f:
        res = client.post(
            "/api/v1/recognize/video",
            files={"file": ("thank_you.mov", f, "video/quicktime")},
        )

    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["label"] == "thankyou"
    assert data["display_name"] == "THANK YOU"
    assert data["confidence"] > 0.20
    assert data["feature_dimension"] == 134
    assert len(data["top5_candidates"]) == 5


def test_api_recognize_video_rejects_invalid_extension(client):
    fake_file = b"not a real video content"
    res = client.post(
        "/api/v1/recognize/video",
        files={"file": ("malicious.exe", fake_file, "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Unsupported video extension" in res.json()["detail"]


def test_streaming_include_adapter_sliding_window_buffer():
    from models.adapters.include import StreamingIncludeAdapter

    adapter = StreamingIncludeAdapter(window_size=30, eval_every_n_frames=3, min_frames=15)
    adapter.load()

    # Stream 45 synthetic frames to ensure buffer caps at 30
    rng = np.random.RandomState(42)
    for i in range(45):
        dummy_feat = rng.randn(134).astype(np.float32)
        adapter.append_vector(dummy_feat)

    # 1. Verify buffer bound (zero memory leak under continuous streaming)
    assert len(adapter.frame_buffer) == 30
    assert adapter.frame_buffer.maxlen == 30
    assert adapter.frame_counter == 45

    # 2. Verify window prediction & entropy evaluation
    res = adapter.predict_window(confidence_threshold=0.01)
    assert res.feature_dimension == 134
    assert res.entropy is not None
    assert isinstance(res.entropy, float)
    assert len(res.top5_candidates) == 5

    # 3. Test reset
    adapter.reset()
    assert len(adapter.frame_buffer) == 0
    assert adapter.frame_counter == 0

