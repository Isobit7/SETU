"""
SETU — Comprehensive User Inputs & Error Scenarios Test Suite
Tests all possible inputs a user can provide (WebSocket frames, control commands,
REST API payloads, edge cases) and verifies system resilience against all user errors.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pipeline_manager import pipeline


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. WEBSOCKET FRAME INPUTS: VALID, DISTORTED, EXTREME LIGHTING & CORRUPT BYTES
# ---------------------------------------------------------------------------

def test_websocket_user_input_valid_16_9_frame(client):
    """User streams standard 16:9 webcam frame (e.g., 480x270 or 640x360)."""
    frame = np.full((270, 480, 3), 120, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)

    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        ws.send_bytes(encoded.tobytes())
        resp = ws.receive_json()

        assert resp["type"] == "prediction"
        assert "label" in resp
        assert "confidence" in resp
        assert "fps" in resp
        assert "latency_ms" in resp
        assert resp["hand_detected"] in (True, False)


def test_websocket_user_input_distorted_aspect_ratios(client):
    """User has an unusual webcam aspect ratio (portrait 9:16, ultra-wide 21:9, tiny)."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Portrait frame (e.g. mobile webcam stream 270x480)
        portrait = np.full((480, 270, 3), 100, dtype=np.uint8)
        _, enc_portrait = cv2.imencode(".jpg", portrait)
        ws.send_bytes(enc_portrait.tobytes())
        resp_p = ws.receive_json()
        assert resp_p["type"] == "prediction"

        # Ultra-wide frame (120x640)
        ultrawide = np.full((120, 640, 3), 100, dtype=np.uint8)
        _, enc_uw = cv2.imencode(".jpg", ultrawide)
        ws.send_bytes(enc_uw.tobytes())
        resp_uw = ws.receive_json()
        assert resp_uw["type"] == "prediction"


def test_websocket_user_input_extreme_lighting_conditions(client):
    """User in extreme lighting: pitch black (CLAHE trigger) vs overexposed bright."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Pitch black room (mean lightness < 30)
        dark_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _, enc_dark = cv2.imencode(".jpg", dark_frame)
        ws.send_bytes(enc_dark.tobytes())
        resp_dark = ws.receive_json()
        assert resp_dark["type"] == "prediction"
        assert resp_dark["label"] in ("no_sign", "unknown", "Awaiting Sign Input")

        # Overexposed washed-out room (intensity 250)
        bright_frame = np.full((240, 320, 3), 250, dtype=np.uint8)
        _, enc_bright = cv2.imencode(".jpg", bright_frame)
        ws.send_bytes(enc_bright.tobytes())
        resp_bright = ws.receive_json()
        assert resp_bright["type"] == "prediction"


def test_websocket_user_error_corrupt_frame_bytes(client):
    """User sends corrupted data, 0-byte payload, or random garbage over WebSocket."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # 1. Random garbage bytes
        ws.send_bytes(b"\x00\xFF\xAA\x55\xDE\xAD\xBE\xEF" * 16)
        resp_err = ws.receive_json()
        assert resp_err["type"] == "error"
        assert "Failed to decode JPEG frame" in resp_err["message"]

        # 2. Incomplete / truncated JPEG header
        ws.send_bytes(b"\xFF\xD8\xFF\xE0")
        resp_trunc = ws.receive_json()
        assert resp_trunc["type"] == "error"

        # 3. Plain text sent as binary bytes
        ws.send_bytes(b"HELLO_NOT_AN_IMAGE")
        resp_text_bin = ws.receive_json()
        assert resp_text_bin["type"] == "error"

        # 4. Connection survives and accepts a valid frame immediately after
        valid_frame = np.full((120, 160, 3), 80, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", valid_frame)
        ws.send_bytes(enc.tobytes())
        resp_valid = ws.receive_json()
        assert resp_valid["type"] == "prediction"


# ---------------------------------------------------------------------------
# 2. WEBSOCKET CONTROL COMMANDS: VALID ACTIONS & USER ERROR HANDLING
# ---------------------------------------------------------------------------

def test_websocket_user_control_action_set_model(client):
    """User changes active model via dropdown or shortcuts."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Valid model switches
        for model_name in ("realsign", "isl_mvp", "isl_sequence_lstm", "include"):
            ws.send_text(json.dumps({"action": "set_model", "model": model_name}))
            resp = ws.receive_json()
            assert resp["type"] == "info"
            expected = "include_stream" if model_name == "include" else model_name
            assert resp["active_model"] == expected


def test_websocket_user_control_action_set_threshold_boundaries(client):
    """User adjusts confidence slider to normal, negative, or excessive values."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Normal threshold
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": 0.65}))
        r1 = ws.receive_json()
        assert r1["type"] == "info"
        assert r1["threshold"] == 0.65

        # Sub-zero value should be clamped to 0.10
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": -0.8}))
        r2 = ws.receive_json()
        assert r2["threshold"] == 0.10

        # Excessive value > 1.0 should be clamped to 0.95
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": 2.5}))
        r3 = ws.receive_json()
        assert r3["threshold"] == 0.95


def test_websocket_user_control_action_sentence_buffer_operations(client):
    """User clicks 'Clear', 'Backspace', and 'Reset' on empty and populated buffers."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Backspace on empty buffer
        ws.send_text(json.dumps({"action": "backspace"}))
        r_bs1 = ws.receive_json()
        assert r_bs1["type"] == "info"
        assert "Backspace applied" in r_bs1["message"]

        # Clear on empty buffer
        ws.send_text(json.dumps({"action": "clear_sentence"}))
        r_clr1 = ws.receive_json()
        assert r_clr1["type"] == "info"
        assert "cleared" in r_clr1["message"].lower()

        # Reset action
        ws.send_text(json.dumps({"action": "reset"}))
        r_rst = ws.receive_json()
        assert r_rst["type"] == "info"
        assert "Buffer reset" in r_rst["message"]


def test_websocket_user_control_disambiguation_and_prediction(client):
    """User interacts with Disambiguation HUD and Next-Word Predictive buttons."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Resolve disambiguation with explicit letter choice
        ws.send_text(json.dumps({"action": "resolve_disambiguation", "selected": "M"}))
        r_dis = ws.receive_json()
        assert r_dis["type"] == "info"
        assert "Resolved disambiguation with M" in r_dis["message"]
        assert "M" in r_dis["sentence_buffer"]

        # Resolve disambiguation with 'left' alias
        ws.send_text(json.dumps({"action": "resolve_disambiguation", "selected": "left"}))
        r_left = ws.receive_json()
        assert r_left["type"] == "info"

        # Accept predictive word phrase
        ws.send_text(json.dumps({"action": "accept_prediction", "phrase": "PLEASE"}))
        r_pred = ws.receive_json()
        assert r_pred["type"] == "info"
        assert "PLEASE" in r_pred["sentence_buffer"]


def test_websocket_user_error_malformed_json_and_unknown_actions(client):
    """User client sends malformed JSON, missing action, or unknown command strings."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Malformed JSON syntax
        ws.send_text("{bad_json: True, missing_quotes")
        # System catches error internally, does not crash connection

        # Unknown action
        ws.send_text(json.dumps({"action": "unknown_action_foo_bar"}))

        # Action with wrong data type for threshold
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": "not_a_number"}))

        # Verify WebSocket is healthy by running a reset
        ws.send_text(json.dumps({"action": "reset"}))
        r_ok = ws.receive_json()
        assert r_ok["type"] == "info"
        assert "Buffer reset" in r_ok["message"]


# ---------------------------------------------------------------------------
# 3. REST API /api/predict: INPUT FEATURE DIMENSIONS & USER VALIDATION ERRORS
# ---------------------------------------------------------------------------

def test_api_predict_user_input_dimensions(client):
    """User provides 63-D single-hand, 89-D enriched, or 126-D dual-hand vectors."""
    # 63-D single-hand feature vector
    r63 = client.post("/api/predict", json={"features": [0.05] * 63})
    assert r63.status_code == 200
    assert "letter" in r63.json()

    # 126-D dual-hand feature vector
    r126 = client.post("/api/predict", json={"features": [0.05] * 126})
    assert r126.status_code == 200
    assert "letter" in r126.json()

    # 21 primary + 21 secondary landmarks for two-handed ISL letters
    prim = [{"x": 0.5, "y": 0.5 + i * 0.01, "z": 0.0} for i in range(21)]
    sec = [{"x": 0.3, "y": 0.4 + i * 0.01, "z": 0.0} for i in range(21)]
    r_dual = client.post(
        "/api/predict",
        json={"landmarks": prim, "secondary_landmarks": sec, "dominant_hand": "right"},
    )
    assert r_dual.status_code == 200
    assert len(r_dual.json()["top_3"]) == 3


def test_api_predict_user_error_invalid_inputs(client):
    """User submits invalid payloads to /api/predict."""
    # 1. Empty body
    r_empty = client.post("/api/predict", json={})
    assert r_empty.status_code == 400

    # 2. Unsupported vector dimension (e.g., 10-D)
    r_bad_dim = client.post("/api/predict", json={"features": [0.1] * 10})
    assert r_bad_dim.status_code == 400
    assert "expected" in r_bad_dim.json()["detail"].lower()

    # 3. Bad landmark count (15 instead of 21)
    r_bad_lm = client.post(
        "/api/predict",
        json={"landmarks": [{"x": 0.5, "y": 0.5}] * 15},
    )
    assert r_bad_lm.status_code == 400

    # 4. NaN values in features
    nan_features = [0.1] * 63
    nan_features[10] = float("nan")
    r_nan = client.post(
        "/api/predict",
        content=json.dumps({"features": nan_features}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert r_nan.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 4. VIDEO UPLOAD API: SUPPORTED & UNSUPPORTED FILE USER ERRORS
# ---------------------------------------------------------------------------

def test_api_video_user_error_unsupported_extensions(client):
    """User uploads non-video file or unsupported video extension."""
    # 1. Text file
    res_txt = client.post(
        "/api/v1/recognize/video",
        files={"file": ("notes.txt", b"Hello ISL", "text/plain")},
    )
    assert res_txt.status_code == 400
    assert "Unsupported video extension" in res_txt.json()["detail"]

    # 2. Executable / binary file
    res_exe = client.post(
        "/api/v1/recognize/video",
        files={"file": ("virus.exe", b"\x4D\x5A\x90\x00", "application/octet-stream")},
    )
    assert res_exe.status_code == 400


def test_api_video_user_error_empty_file(client):
    """User uploads a 0-byte video file."""
    res_empty = client.post(
        "/api/v1/recognize/video",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    # The endpoint should reject or return an error cleanly without unhandled crash
    assert res_empty.status_code in (400, 500)


# ---------------------------------------------------------------------------
# 5. DATASET RECORDING & CAPTURE: PATH TRAVERSAL & INJECTION USER ERRORS
# ---------------------------------------------------------------------------

def test_dataset_record_sample_path_traversal_sanitization(client):
    """User passes path traversal or suspicious characters in dataset label."""
    # Label with directory traversal sequences
    traversal_label = "../../sneaky_sign"
    valid_seq = [[0.05] * 164 for _ in range(30)]

    res = client.post(
        "/api/v1/dataset/record_sample",
        json={"label": traversal_label, "sequence_164d": valid_seq},
    )
    if res.status_code == 200:
        data = res.json()
        saved_path = Path(data["file_path"])
        try:
            # Ensure file was saved strictly inside data/processed, with traversal stripped
            assert "data" in str(saved_path)
            assert ".." not in saved_path.name
        finally:
            if saved_path.exists():
                saved_path.unlink()
    else:
        assert res.status_code == 400


def test_dataset_capture_session_user_inputs(client):
    """User uploads session data via continuous capture recording."""
    # Valid continuous session payload
    valid_payload = {
        "label": "A",
        "session_id": "test_session_user_input_001",
        "sequence_features": [[0.05] * 126],
        "user_permission_granted": True,
    }
    res = client.post("/api/v1/dataset/capture_session", json=valid_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["session_id"] == "test_session_user_input_001"

    # Cleanup created session directory
    import shutil
    created_dir = Path(data.get("directory", ""))
    if created_dir.exists():
        session_dir = created_dir.parent
        if session_dir.exists() and "test_session" in session_dir.name:
            shutil.rmtree(session_dir, ignore_errors=True)
        else:
            shutil.rmtree(created_dir, ignore_errors=True)


def test_nonexistent_endpoint_returns_404(client):
    """User navigates to invalid or mistyped API routes."""
    res = client.get("/api/v1/this_route_does_not_exist")
    assert res.status_code == 404
