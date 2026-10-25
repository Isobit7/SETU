import json as py_json
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pipeline_manager import pipeline
from backend.app.websocket.recognize import serialize_landmarks, check_subtitle_collision, get_positioning_guidance


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["pipeline_ready"] is True
    assert "hello" in data["supported_signs"]
    assert "thank_you" in data["supported_signs"]
    assert "isl_sequence_lstm" in data["active_classifiers"]
    assert "sequence_lstm_ready" in data


def test_models_catalog_endpoint(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_ids = [m["id"] for m in data["models"]]
    assert "realsign" in model_ids
    assert "isl_mvp" in model_ids
    assert "isl_sequence_lstm" in model_ids


def test_sequence_lstm_status_endpoint(client):
    res = client.get("/api/v1/models/sequence_lstm/status")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "ISL-Sequence-2Layer-LSTM"
    assert data["classes_count"] == len(data["classes"])
    assert data["input_features"] == 328
    assert "hello" in data["classes"]
    assert "namaste" in data["classes"]


def test_vocabulary_endpoint(client):
    res = client.get("/api/v1/vocabulary")
    assert res.status_code == 200
    data = res.json()
    assert len(data["vocabulary"]) == 5
    assert data["negative_class"] == "no_sign"


def test_subtitle_collision():
    # Hands outside subtitle box
    outside = [{"x": 0.5, "y": 0.3, "z": 0.0}]
    assert check_subtitle_collision(outside, []) is False

    # Hands inside subtitle box (y >= 0.72, x in [0.2, 0.8])
    inside = [{"x": 0.5, "y": 0.85, "z": 0.0}]
    assert check_subtitle_collision(inside, []) is True


def test_positioning_guidance():
    # Empty pose
    assert "Step into frame" in get_positioning_guidance([], False)

    # Mock pose with close shoulders (narrow width = far away)
    pose_far = [{"x": 0.5, "y": 0.2, "z": 0.0} for _ in range(33)]
    pose_far[11] = {"x": 0.45, "y": 0.3, "z": 0.0}
    pose_far[12] = {"x": 0.52, "y": 0.3, "z": 0.0}
    assert "Too far" in get_positioning_guidance(pose_far, False)


def test_websocket_recognize_mock_frame(client):
    # Create test JPEG image
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    jpeg_bytes = encoded.tobytes()

    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Send raw bytes frame
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        assert resp["type"] == "prediction"
        assert "label" in resp
        assert "confidence" in resp
        assert "landmarks" in resp
        assert "latency_ms" in resp
        assert resp["label"] in ["no_sign", "unknown", "Awaiting Sign Input"]

        # Send text action to switch model to isl_sequence_lstm
        ws.send_text('{"action": "set_model", "model": "isl_sequence_lstm"}')
        resp_model = ws.receive_json()
        assert resp_model["type"] == "info"
        assert "isl_sequence_lstm" in resp_model["message"]

        # Send frame under isl_sequence_lstm model
        ws.send_bytes(jpeg_bytes)
        resp_seq = ws.receive_json()
        assert resp_seq["type"] == "prediction"
        assert resp_seq["active_model"] == "isl_sequence_lstm"
        assert "sequence_lstm_prediction" in resp_seq

        # Send set_threshold action
        ws.send_text('{"action": "set_threshold", "threshold": 0.45}')
        resp_thresh = ws.receive_json()
        assert resp_thresh["type"] == "info"
        assert resp_thresh["threshold"] == 0.45

        # Send reset action
        ws.send_text('{"action": "reset"}')
        resp_action = ws.receive_json()
        assert resp_action["type"] == "info"
        assert "Buffer reset" in resp_action["message"]


def test_record_sample_endpoint_valid_and_list(client):
    """Verifies POST /api/v1/dataset/record_sample saves 30x164 npz and GET lists it."""
    test_label = "test_custom_hello"
    valid_seq = [[0.05] * 164 for _ in range(30)]

    response = client.post(
        "/api/v1/dataset/record_sample",
        json={"label": test_label, "sequence_164d": valid_seq},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["label"] == test_label
    assert data["frames"] == 30
    assert data["feature_dim"] == 164
    created_file = Path(data["file_path"])
    assert created_file.exists()

    try:
        # Verify file content
        with np.load(created_file) as npz:
            assert "features" in npz
            assert npz["features"].shape == (30, 164)
            assert str(npz["label"]) == test_label

        # Verify GET /api/v1/dataset/custom_samples
        list_res = client.get("/api/v1/dataset/custom_samples")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total_samples"] >= 1
        assert test_label in list_data["labels"]
        sample_names = [s["filename"] for s in list_data["samples"]]
        assert created_file.name in sample_names
    finally:
        # Cleanup test artifact to preserve pristine test environment
        if created_file.exists():
            created_file.unlink()


def test_record_sample_validation_errors(client):
    """Verifies POST /api/v1/dataset/record_sample validation gates."""
    # 1. Empty label
    res = client.post(
        "/api/v1/dataset/record_sample",
        json={"label": "   ", "sequence_164d": [[0.0] * 164 for _ in range(30)]},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()

    # 2. Sequence length != 30
    res = client.post(
        "/api/v1/dataset/record_sample",
        json={"label": "invalid_len", "sequence_164d": [[0.0] * 164 for _ in range(15)]},
    )
    assert res.status_code == 400
    assert "30 frames" in res.json()["detail"]

    # 3. Feature dimension != 164
    invalid_dim_seq = [[0.0] * 158 for _ in range(30)]
    res = client.post(
        "/api/v1/dataset/record_sample",
        json={"label": "invalid_dim", "sequence_164d": invalid_dim_seq},
    )
    assert res.status_code == 400
    assert "164" in res.json()["detail"]

    # 4. NaN / Inf values
    nan_seq = [[0.0] * 164 for _ in range(30)]
    nan_seq[5][10] = float("nan")
    raw_payload = py_json.dumps({"label": "nan_test", "sequence_164d": nan_seq}, allow_nan=True)
    res = client.post(
        "/api/v1/dataset/record_sample",
        content=raw_payload,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 400
    assert "nan" in res.json()["detail"].lower()


def test_api_predict_features(client):
    res = client.post("/api/predict", json={"features": [0.05] * 63})
    assert res.status_code == 200
    data = res.json()
    assert "letter" in data
    assert "confidence" in data
    assert "top_3" in data
    assert len(data["top_3"]) == 3
    for candidate in data["top_3"]:
        assert "letter" in candidate
        assert "confidence" in candidate
        assert isinstance(candidate["letter"], str)
        assert isinstance(candidate["confidence"], float)


def test_api_predict_landmarks(client):
    mock_landmarks = [{"x": 0.5, "y": 0.5 + i * 0.01, "z": 0.0} for i in range(21)]
    res = client.post("/api/predict", json={"landmarks": mock_landmarks, "dominant_hand": "right"})
    assert res.status_code == 200
    data = res.json()
    assert "letter" in data
    assert "confidence" in data
    assert "top_3" in data
    assert len(data["top_3"]) == 3


def test_api_predict_validation_errors(client):
    # Bad features dimension
    res = client.post("/api/predict", json={"features": [0.1] * 10})
    assert res.status_code == 400

    # Bad landmarks count
    res = client.post("/api/predict", json={"landmarks": [{"x": 0.1, "y": 0.2}] * 10})
    assert res.status_code == 400

    # Empty payload
    res = client.post("/api/predict", json={})
    assert res.status_code == 400


def test_backend_main_standalone_app():
    from backend.main import app as standalone_app
    with TestClient(standalone_app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        pred_res = client.post("/api/predict", json={"features": [0.1] * 63})
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert "letter" in pred_data
        assert "confidence" in pred_data
        assert "top_3" in pred_data

