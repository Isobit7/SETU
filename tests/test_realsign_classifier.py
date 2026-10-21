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
        assert "candidates" in resp
        assert isinstance(resp["candidates"], list)

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


def _make_hand_pose(
    thumb: bool,
    index: bool,
    middle: bool,
    ring: bool,
    pinky: bool,
    fingers_spread: bool = False,
) -> np.ndarray:
    """Creates a canonical 21-landmark hand array for specific finger extension states."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]        # Wrist
    pts[5] = [-0.2, -0.7, 0.0]       # Index MCP
    pts[9] = [0.0, -1.0, 0.0]        # Middle MCP (defines palm length = 1.0)
    pts[13] = [0.2, -0.7, 0.0]       # Ring MCP
    pts[17] = [0.35, -0.6, 0.0]      # Pinky MCP
    pts[1] = [-0.15, -0.2, 0.0]      # Thumb CMC
    pts[2] = [-0.25, -0.4, 0.0]      # Thumb MCP

    # Thumb tip (4)
    pts[4] = [-0.8, -0.9, 0.0] if thumb else [-0.3, -0.6, 0.0]
    # Index tip (8)
    idx_x = -0.4 if fingers_spread else -0.2
    pts[8] = [idx_x, -1.8, 0.0] if index else [-0.2, -0.75, 0.0]
    # Middle tip (12)
    mid_x = 0.2 if fingers_spread else 0.0
    pts[12] = [mid_x, -2.0, 0.0] if middle else [0.0, -0.85, 0.0]
    # Ring tip (16)
    pts[16] = [0.2, -1.8, 0.0] if ring else [0.2, -0.75, 0.0]
    # Pinky tip (20)
    pts[20] = [0.35, -1.5, 0.0] if pinky else [0.35, -0.65, 0.0]

    return pts


def test_finger_extension_states_and_heuristics():
    """Verify get_finger_extension_states and evaluate_alphabet_heuristics across 7 key signs."""
    from setu.features.geometry import get_finger_extension_states, evaluate_alphabet_heuristics

    # 1. 'L': Index + Thumb extended, others curled
    pose_l = _make_hand_pose(thumb=True, index=True, middle=False, ring=False, pinky=False)
    assert get_finger_extension_states(pose_l) == (True, True, False, False, False)
    assert evaluate_alphabet_heuristics(pose_l) == "L"

    # 2. 'I': Pinky extended only, others curled
    pose_i = _make_hand_pose(thumb=False, index=False, middle=False, ring=False, pinky=True)
    assert get_finger_extension_states(pose_i) == (False, False, False, False, True)
    assert evaluate_alphabet_heuristics(pose_i) == "I"

    # 3. 'V': Index + Middle extended (apart), others curled
    pose_v = _make_hand_pose(thumb=False, index=True, middle=True, ring=False, pinky=False, fingers_spread=True)
    assert get_finger_extension_states(pose_v) == (False, True, True, False, False)
    assert evaluate_alphabet_heuristics(pose_v) == "V"

    # 4. 'W': Index + Middle + Ring extended, others curled
    pose_w = _make_hand_pose(thumb=False, index=True, middle=True, ring=True, pinky=False)
    assert get_finger_extension_states(pose_w) == (False, True, True, True, False)
    assert evaluate_alphabet_heuristics(pose_w) == "W"

    # 5. 'Y': Thumb + Pinky extended, middle three curled
    pose_y = _make_hand_pose(thumb=True, index=False, middle=False, ring=False, pinky=True)
    assert get_finger_extension_states(pose_y) == (True, False, False, False, True)
    assert evaluate_alphabet_heuristics(pose_y) == "Y"

    # 6. 'B': All 4 fingers extended together, thumb tucked
    pose_b = _make_hand_pose(thumb=False, index=True, middle=True, ring=True, pinky=True)
    assert get_finger_extension_states(pose_b) == (False, True, True, True, True)
    assert evaluate_alphabet_heuristics(pose_b) == "B"

    # 7. 'A': All fingers curled into fist
    pose_a = _make_hand_pose(thumb=False, index=False, middle=False, ring=False, pinky=False)
    assert get_finger_extension_states(pose_a) == (False, False, False, False, False)
    assert evaluate_alphabet_heuristics(pose_a) == "A"


def test_realsign_classifier_hybrid_heuristics_boost():
    """Verify RealSignClassifier.predict applies geometric boosts to achieve high confidence on clear poses."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)

    cases = [
        ("L", _make_hand_pose(thumb=True, index=True, middle=False, ring=False, pinky=False)),
        ("I", _make_hand_pose(thumb=False, index=False, middle=False, ring=False, pinky=True)),
        ("V", _make_hand_pose(thumb=False, index=True, middle=True, ring=False, pinky=False, fingers_spread=True)),
        ("W", _make_hand_pose(thumb=False, index=True, middle=True, ring=True, pinky=False)),
        ("Y", _make_hand_pose(thumb=True, index=False, middle=False, ring=False, pinky=True)),
        ("B", _make_hand_pose(thumb=False, index=True, middle=True, ring=True, pinky=True)),
        ("A", _make_hand_pose(thumb=False, index=False, middle=False, ring=False, pinky=False)),
    ]

    for expected_letter, pose_21x3 in cases:
        pred = clf.predict_landmarks(pose_21x3, confidence_threshold=0.50)
        assert pred.letter == expected_letter
        assert pred.is_known_sign is True
        assert pred.confidence >= 0.80
        assert pred.top_3[0][0] == expected_letter


def test_instant_retrain_realsign_mlp(tmp_path):
    """Verify scripts.train_az_realsign.retrain_realsign_mlp completes in < 5 seconds with high accuracy."""
    from scripts.train_az_realsign import retrain_realsign_mlp, generate_synthetic_samples_for_letter

    # Create small calibration dataset of 10 samples each for letters A and B
    samples_a = generate_synthetic_samples_for_letter("A", num_samples=10)
    samples_b = generate_synthetic_samples_for_letter("B", num_samples=10)
    X_new = np.vstack([samples_a, samples_b])
    y_new = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    output_model = tmp_path / "test_realsign_mlp.pth"
    base_model = Path("models/realsign_26class_mlp.pth")

    elapsed, acc = retrain_realsign_mlp(
        X_new=X_new,
        y_new=y_new,
        base_model_path=base_model,
        output_model_path=output_model,
        epochs=10,
        lr=0.001,
        batch_size=16,
    )

    assert elapsed < 5.0, f"Retraining took {elapsed:.2f}s, expected < 5.0s"
    assert acc >= 0.85
    assert output_model.exists()
    assert output_model.stat().st_size > 10000


def test_realsign_classifier_ema_smoothing():
    """Verify EMA probability smoothing S_t = alpha * P_t + (1 - alpha) * S_{t-1} with alpha=0.40."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)
    assert clf.ema_alpha == 0.40

    pose_l = _make_hand_pose(thumb=True, index=True, middle=False, ring=False, pinky=False)
    pose_i = _make_hand_pose(thumb=False, index=False, middle=False, ring=False, pinky=True)

    # Frame 1: Without smoothing vs with smoothing initialization
    clf.reset_smoothing()
    pred_1 = clf.predict_landmarks(pose_l, apply_smoothing=True)
    assert pred_1.letter == "L"
    assert clf._ema_probs is not None
    s0 = clf._ema_probs.copy()

    # Frame 2: Switch instantly to I with smoothing
    # Raw I probability would jump, but EMA filters the transition
    pred_2_raw = clf.predict_landmarks(pose_i, apply_smoothing=False)
    p1 = np.array([pred_2_raw.probabilities[c] for c in clf.class_names], dtype=np.float32)

    pred_2_smooth = clf.predict_landmarks(pose_i, apply_smoothing=True)
    expected_s1 = 0.40 * p1 + 0.60 * s0
    actual_s1 = np.array([pred_2_smooth.probabilities[c] for c in clf.class_names], dtype=np.float32)

    assert np.allclose(actual_s1, expected_s1, atol=1e-4)

    # Reset smoothing clears the state
    clf.reset_smoothing()
    assert clf._ema_probs is None


def test_realsign_confidence_and_margin_gate():
    """Verify that predict_vector returns 'Awaiting Sign Input' when top1 < 0.50 or margin < 0.15."""
    weights_path = Path("models/realsign_26class_mlp.pth")
    if not weights_path.exists():
        pytest.skip("models/realsign_26class_mlp.pth not found")

    clf = RealSignClassifier(weights_path)

    # 1. Ambiguous / low-confidence feature vector (e.g., zero vector)
    dummy_vec = np.zeros(63, dtype=np.float32)
    pred_gated = clf.predict_vector(dummy_vec)
    assert pred_gated.letter == "Awaiting Sign Input"
    assert pred_gated.label == "Awaiting Sign Input"
    assert pred_gated.confidence == 0.0
    assert pred_gated.is_known_sign is False

    # 2. Clear sign pose (e.g., 'L')
    pose_l = _make_hand_pose(thumb=True, index=True, middle=False, ring=False, pinky=False)
    pred_clear = clf.predict_landmarks(pose_l)
    assert pred_clear.letter == "L"
    assert pred_clear.label == "L"
    assert pred_clear.is_known_sign is True
    assert pred_clear.confidence >= 0.50
    margin = pred_clear.top_3[0][1] - pred_clear.top_3[1][1]
    assert margin >= 0.15


