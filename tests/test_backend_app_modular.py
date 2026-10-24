"""Unit and integration tests for modular backend and ML modules:
- backend.app.main
- backend.app.schemas.predict
- backend.app.services.classifier
- backend.app.api.endpoints.predict
- ml.feature_extractor
- ml.evaluate
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app as modular_app
from backend.app.schemas.predict import LandmarkRequest, PredictionResponse, TopCandidate
from backend.app.services.classifier import RealSignMLP, RealSignInferenceService, classifier_service
from ml.feature_extractor import normalize_landmarks_to_63d, extract_realsign_hand_features
from ml.evaluate import evaluate_model


@pytest.fixture(scope="module")
def client():
    with TestClient(modular_app) as c:
        yield c


def test_modular_app_health(client):
    """Verify /health endpoint on backend.app.main."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 26


def test_modular_app_predict_features(client):
    """Verify POST /api/predict with 63-D feature array."""
    # MCP middle finger extended, other fingers zero
    feats = [0.0] * 63
    feats[28] = -1.0  # Middle MCP y = -1.0
    res = client.post("/api/predict", json={"features": feats})
    assert res.status_code == 200
    data = res.json()
    assert "letter" in data
    assert "confidence" in data
    assert "top_3" in data
    assert len(data["top_3"]) == 3
    assert 0.0 <= data["confidence"] <= 1.0


def test_modular_app_predict_landmarks(client):
    """Verify POST /api/predict with 21 3D landmarks."""
    landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    # Middle MCP
    landmarks[9] = {"x": 0.5, "y": 0.3, "z": 0.0}
    res = client.post("/api/predict", json={"landmarks": landmarks})
    assert res.status_code == 200
    data = res.json()
    assert len(data["letter"]) >= 1
    assert len(data["top_3"]) == 3


def test_modular_app_predict_validation_errors(client):
    """Verify error codes when payload is invalid."""
    # Empty payload (raises 400 Bad Request)
    res = client.post("/api/predict", json={})
    assert res.status_code == 400

    # Invalid feature count (raises 400 Bad Request with explanatory detail)
    res = client.post("/api/predict", json={"features": [0.1] * 10})
    assert res.status_code == 400
    assert "63" in res.json()["detail"]

    # Invalid landmark count (raises 400 Bad Request with explanatory detail)
    res = client.post("/api/predict", json={"landmarks": [{"x": 0.1, "y": 0.2}] * 5})
    assert res.status_code == 400
    assert "21" in res.json()["detail"]

    # Malformed data type (string instead of list) -> 422 Unprocessable Entity
    res = client.post("/api/predict", json={"features": "not_a_list"})
    assert res.status_code == 422


def test_realsign_mlp_architecture():
    """Verify PyTorch model shapes and forward pass."""
    import torch
    model = RealSignMLP(input_dim=63, hidden1=128, hidden2=64, num_classes=26)
    x = torch.randn(4, 63)
    out = model(x)
    assert out.shape == (4, 26)


def test_ml_feature_extractor_standalone():
    """Verify ml.feature_extractor transforms properly."""
    raw_pts = np.zeros((21, 3), dtype=np.float32)
    raw_pts[0] = [0.5, 0.5, 0.0]
    raw_pts[9] = [0.5, 0.3, 0.0]
    for i in range(1, 21):
        if i != 9:
            raw_pts[i] = [0.5 + 0.01 * i, 0.5 - 0.01 * i, 0.0]

    feats = extract_realsign_hand_features(raw_pts)
    assert feats is not None
    assert feats.shape == (63,)
    assert np.allclose(feats[:3], [0.0, 0.0, 0.0], atol=1e-5)
