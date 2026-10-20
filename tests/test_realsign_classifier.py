"""
Tests for SETU RealSign 26-Class PyTorch Alphabet Classifier and API Integration.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from setu.api.server import app, pipeline
from setu.models.realsign_classifier import RealSignClassifier, RealSignPrediction, RealSignMLP
from setu.features.extractor import extract_realsign_hand_features


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_realsign_classifier_initialization():
    """Verify RealSignClassifier loads the 26-class PyTorch model correctly."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)
    assert clf.num_classes == 26
    assert clf.class_names == [chr(ord("A") + i) for i in range(26)]
    assert isinstance(clf.model, RealSignMLP)
    assert clf.input_dim == 63
    if clf.test_accuracy is not None:
        assert clf.test_accuracy >= 0.85


def test_extract_realsign_hand_features():
    """Verify dedicated wrist-centered 63-D RealSign feature extraction."""
    # 1. Missing / None handling
    assert extract_realsign_hand_features(None) is None
    assert extract_realsign_hand_features([]) is None
    assert extract_realsign_hand_features(np.zeros((10, 3))) is None

    # 2. Valid 21x3 landmarks
    raw_pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.5, 0.0)
    raw_pts[0] = [0.5, 0.5, 0.0]
    # MCP middle finger (index 9) at (0.5, 0.3, 0.0) -> distance = 0.2
    raw_pts[9] = [0.5, 0.3, 0.0]
    # Other points
    for i in range(1, 21):
        if i != 9:
            raw_pts[i] = [0.5 + 0.01 * i, 0.5 - 0.01 * i, 0.0]

    feat = extract_realsign_hand_features(raw_pts)
    assert feat is not None
    assert feat.shape == (63,)
    assert feat.dtype == np.float32
    # Wrist point in normalized coords must be at (0, 0, 0)
    assert np.allclose(feat[:3], [0.0, 0.0, 0.0], atol=1e-5)
    # Distance of MCP 9 from wrist should be exactly 1.0 (since normalized by palm_size)
    assert np.isclose(np.linalg.norm(feat[27:30]), 1.0, atol=1e-5)


def test_realsign_classifier_predict_direct():
    """Verify clf.predict accepts 63-D normalized features and populates logits & probs."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)
    dummy_vec = np.zeros(63, dtype=np.float32)

    pred = clf.predict(dummy_vec, confidence_threshold=0.0)
    assert isinstance(pred, RealSignPrediction)
    assert pred.letter in clf.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 26
    assert pred.logits is not None
    assert len(pred.logits) == 26
    assert len(pred.top_3) == 3


def test_realsign_classifier_predict_vector():
    """Verify inference on a 63-D normalized feature vector."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)
    dummy_vec = np.zeros(63, dtype=np.float32)

    pred = clf.predict_vector(dummy_vec, confidence_threshold=0.0)
    assert isinstance(pred, RealSignPrediction)
    assert pred.letter in clf.class_names
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 26
    assert len(pred.top_3) == 3


def test_realsign_classifier_predict_landmarks():
    """Verify inference from 21x3 raw landmark points."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)
    # Generate 21 dummy 3D points
    pts = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)

    pred = clf.predict_landmarks(pts, confidence_threshold=0.0)
    assert isinstance(pred, RealSignPrediction)
    assert pred.letter in clf.class_names
    assert 0.0 <= pred.confidence <= 1.0


def test_realsign_api_endpoints(client):
    """Verify RealSign status, classes, and models catalog REST endpoints."""
    # 1. Models catalog
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    model_ids = [m["id"] for m in models_data["models"]]
    assert "realsign" in model_ids
    assert "isl_mvp" in model_ids
    assert "include" in model_ids

    # 2. RealSign status
    res_status = client.get("/api/v1/models/realsign/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["classes_count"] == 26
    assert status_data["input_features"] == 63
    assert status_data["test_accuracy"] >= 0.85

    # 3. RealSign classes
    res_classes = client.get("/api/v1/models/realsign/classes")
    assert res_classes.status_code == 200
    classes_data = res_classes.json()
    assert classes_data["total_classes"] == 26
    letters = [c["letter"] for c in classes_data["classes"]]
    assert letters == [chr(ord("A") + i) for i in range(26)]


def test_websocket_realsign_mode(client):
    """Verify WebSocket recognize endpoint serves RealSign 26-class alphabet model."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    jpeg_bytes = encoded.tobytes()

    with client.websocket_connect("/api/v1/ws/recognize?model=realsign") as ws:
        # Send a frame
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        assert resp["type"] == "prediction"
        assert resp["active_model"] == "realsign"
        assert "label" in resp
        assert "confidence" in resp
        assert "landmarks" in resp
        assert "realsign_prediction" in resp

        # Switch model via text action
        ws.send_text('{"action": "set_model", "model": "isl_mvp"}')
        switch_resp = ws.receive_json()
        assert switch_resp["type"] == "info"
        assert switch_resp["active_model"] == "isl_mvp"

        # Switch back to realsign
        ws.send_text('{"action": "set_model", "model": "realsign"}')
        switch_resp2 = ws.receive_json()
        assert switch_resp2["type"] == "info"
        assert switch_resp2["active_model"] == "realsign"
