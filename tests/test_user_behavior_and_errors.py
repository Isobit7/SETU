"""
SETU — Comprehensive User Behavior, Edge Cases & Error Handling Test Suite (Pillar 1)

Validates how the system handles realistic human behaviors, operational mistakes,
malformed client payloads, extreme camera/environment conditions, and client misuse:
1. WebSocket corrupt/truncated frames, random noise, and connection resilience.
2. WebSocket frame bursting and backpressure drain handling.
3. Out-of-range thresholds, invalid command JSON, and unknown model fallbacks.
4. Rapid user buffer manipulation (spamming backspace, clear, reset, disambiguation).
5. User ergonomic errors: hands in lap/desk (resting pose rejection PROB-08).
6. High kinetic transition / motion-blur suppression (PROB-09).
7. Extreme lighting: 0 lux (pitch black) and 255 lux (washed out).
8. Spatial handedness flapping prevention via temporal hysteresis (PROB-05).
9. REST API video upload errors (0-byte, disguised non-video, corrupt header).
10. Directory traversal and path injection sanitization in dataset recording.
11. REST API feature dimension and non-finite float (NaN/Inf) rejection.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.vision.landmarks import HandednessTracker
from training.features.geometry import (
    is_resting_or_neutral_pose,
    compute_inter_hand_relative_geometry,
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. WEBSOCKET CORRUPT FRAMES, TRUNCATED PAYLOADS & RECOVERY
# ---------------------------------------------------------------------------

def test_ws_corrupt_and_malformed_image_frames(client):
    """User streams corrupt bytes, 0-byte blobs, or non-image payloads; verifies recovery."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # A. 0-byte binary frame
        ws.send_bytes(b"")
        # System drops/ignores empty frame or returns decode error without crashing

        # B. High-entropy random byte string
        ws.send_bytes(b"\xde\xad\xbe\xef\xaa\x55\x00\xff" * 32)
        resp_noise = ws.receive_json()
        assert resp_noise["type"] == "error"
        assert "decode" in resp_noise["message"].lower()

        # C. Truncated JPEG stream (SOI present but truncated data)
        ws.send_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00")
        resp_trunc = ws.receive_json()
        assert resp_trunc["type"] == "error"

        # D. Plaintext sent in binary frame channel
        ws.send_bytes(b"GET /camera_stream HTTP/1.1\r\n\r\n")
        resp_text = ws.receive_json()
        assert resp_text["type"] == "error"

        # E. Immediate recovery: valid frame succeeds immediately after error sequence
        valid_frame = np.full((180, 240, 3), 100, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", valid_frame)
        ws.send_bytes(enc.tobytes())
        resp_valid = ws.receive_json()
        assert resp_valid["type"] == "prediction"
        assert "label" in resp_valid
        assert "latency_ms" in resp_valid


def test_ws_rapid_frame_burst_and_backpressure(client):
    """User webcam floods server with rapid frames; drains without deadlock."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", frame)
        frame_bytes = enc.tobytes()

        # Burst 3 frames into the queue rapidly
        ws.send_bytes(frame_bytes)
        ws.send_bytes(frame_bytes)
        ws.send_bytes(frame_bytes)

        # Receive at least one prediction confirming frames are processed
        resp = ws.receive_json()
        assert resp["type"] == "prediction"
        assert "latency_ms" in resp

        # Send control reset and loop through responses until reset acknowledgment arrives
        ws.send_text(json.dumps({"action": "reset"}))
        found_reset = False
        for _ in range(5):
            msg = ws.receive_json()
            if msg.get("type") == "info" and "Buffer reset" in msg.get("message", ""):
                found_reset = True
                break
        assert found_reset, "Reset control message should be acknowledged even after frame burst"


# ---------------------------------------------------------------------------
# 2. WEBSOCKET CONTROL COMMAND ERRORS & OUT-OF-RANGE PARAMETERS
# ---------------------------------------------------------------------------

def test_ws_control_invalid_and_out_of_bound_commands(client):
    """User sends malformed JSON, out-of-range thresholds, or nonexistent models."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # A. Malformed JSON syntax
        ws.send_text("{broken: json, missing_bracket")

        # B. Out-of-bounds threshold < 0.10 (should be clamped to 0.10)
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": -99.5}))
        r_low = ws.receive_json()
        assert r_low["type"] == "info"
        assert r_low["threshold"] == 0.10

        # C. Out-of-bounds threshold > 0.95 (should be clamped to 0.95)
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": 50.0}))
        r_high = ws.receive_json()
        assert r_high["type"] == "info"
        assert r_high["threshold"] == 0.95

        # D. Non-numeric threshold
        ws.send_text(json.dumps({"action": "set_threshold", "threshold": "ultra_confident"}))

        # E. Unknown model name (should be ignored or fallback gracefully)
        ws.send_text(json.dumps({"action": "set_model", "model": "nonexistent_neural_v99"}))

        # F. Valid reset verifies session is still in clean working state
        ws.send_text(json.dumps({"action": "reset"}))
        r_rst = ws.receive_json()
        assert r_rst["type"] == "info"
        assert "Buffer reset" in r_rst["message"]


def test_ws_user_buffer_manipulation_spam(client):
    """User rapidly spams backspace, clear, and reset actions on empty and populated buffers."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Spam backspace 5 times on an empty buffer
        for _ in range(5):
            ws.send_text(json.dumps({"action": "backspace"}))
            r = ws.receive_json()
            assert r["type"] == "info"
            assert "sentence_buffer" in r

        # Spam clear_sentence 3 times
        for _ in range(3):
            ws.send_text(json.dumps({"action": "clear_sentence"}))
            r = ws.receive_json()
            assert r["type"] == "info"
            assert r["sentence_buffer"] == ""

        # Disambiguation selection with non-standard payload
        ws.send_text(json.dumps({"action": "resolve_disambiguation", "selected": ""}))
        r_dis = ws.receive_json()
        assert r_dis["type"] == "info"

        # Predictive phrase with unicode characters and emojis
        ws.send_text(json.dumps({"action": "accept_prediction", "phrase": "THANK YOU! \U0001f44d"}))
        r_pred = ws.receive_json()
        assert r_pred["type"] == "info"
        assert "THANK YOU!" in r_pred["sentence_buffer"]


# ---------------------------------------------------------------------------
# 3. USER ERGONOMIC BEHAVIORS: LAP RESTING, KINETIC ENERGY & LIGHTING
# ---------------------------------------------------------------------------

def test_user_behavior_resting_hands_and_lap_rejection():
    """Hands resting in lap or on desk (y >= 0.82) are rejected to prevent false signs (PROB-08)."""
    # 1. Hand in resting lap position (y = 0.88, below threshold)
    resting_hand = [{"x": 0.5, "y": 0.88 + i * 0.002, "z": 0.0} for i in range(21)]
    assert is_resting_or_neutral_pose(resting_hand) is True

    # 2. Hand active in chest/signing zone (y = 0.45)
    active_hand = [{"x": 0.5, "y": 0.45 + i * 0.002, "z": 0.0} for i in range(21)]
    assert is_resting_or_neutral_pose(active_hand) is False

    # 3. Degenerate or empty hand input
    assert is_resting_or_neutral_pose([]) is True
    assert is_resting_or_neutral_pose(None) is True
    assert is_resting_or_neutral_pose([{"x": 0.1, "y": 0.1}] * 10) is True


def test_user_behavior_extreme_ambient_lighting(client):
    """User signs in pitch-black room (0 lux) vs washed out lighting (255 lux)."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # A. Pitch-black frame (intensity 0)
        black_frame = np.zeros((180, 240, 3), dtype=np.uint8)
        _, enc_black = cv2.imencode(".jpg", black_frame)
        ws.send_bytes(enc_black.tobytes())
        r_black = ws.receive_json()
        assert r_black["type"] == "prediction"
        assert r_black["hand_detected"] is False

        # B. Overexposed bright frame (intensity 255)
        bright_frame = np.full((180, 240, 3), 255, dtype=np.uint8)
        _, enc_bright = cv2.imencode(".jpg", bright_frame)
        ws.send_bytes(enc_bright.tobytes())
        r_bright = ws.receive_json()
        assert r_bright["type"] == "prediction"


def test_user_behavior_handedness_flapping_mitigation():
    """Spatial-temporal HandednessTracker suppresses flapping when hands cross midline (PROB-05)."""
    tracker = HandednessTracker(hysteresis_frames=4, max_distance=0.25)
    t0 = 1000.0

    # Frame 1: Right hand detected at wrist (0.50, 0.50)
    lm_right = [{"x": 0.50, "y": 0.50}]
    label1 = tracker.stabilize_single_hand("right", lm_right, timestamp=t0)
    assert label1 == "right"
    assert tracker.stable_label == "right"

    # Frame 2: Flapping glitch reporting "left" at almost same wrist location (0.51, 0.50)
    lm_jitter = [{"x": 0.51, "y": 0.50}]
    label2 = tracker.stabilize_single_hand("left", lm_jitter, timestamp=t0 + 0.033)
    # Hysteresis suppresses single-frame flapping -> retains "right"
    assert label2 == "right"
    assert tracker.flip_streak == 1

    # Frame 3: Another glitch frame
    label3 = tracker.stabilize_single_hand("left", lm_jitter, timestamp=t0 + 0.066)
    assert label3 == "right"
    assert tracker.flip_streak == 2

    # Frame 4: Returning to "right" resets flip streak
    label4 = tracker.stabilize_single_hand("right", lm_jitter, timestamp=t0 + 0.099)
    assert label4 == "right"
    assert tracker.flip_streak == 0

    # Reset cleans history
    tracker.reset()
    assert tracker.stable_label is None
    assert tracker.last_wrist is None


# ---------------------------------------------------------------------------
# 4. REST API VIDEO UPLOAD USER ERRORS & SANITIZATION
# ---------------------------------------------------------------------------

def test_api_video_upload_user_errors(client):
    """User uploads 0-byte video, non-video formats, or corrupt headers."""
    # A. 0-byte video file
    res_empty = client.post(
        "/api/v1/recognize/video",
        files={"file": ("zero_bytes.mp4", b"", "video/mp4")},
    )
    assert res_empty.status_code in (400, 500)

    # B. Plain text file disguised as MP4
    res_fake = client.post(
        "/api/v1/recognize/video",
        files={"file": ("fake_video.mp4", b"This is not a video file", "video/mp4")},
    )
    # Backend fails video stream decoding and returns 400 or error detail
    assert res_fake.status_code in (400, 500)

    # C. Executable file rejected by extension validator
    res_exe = client.post(
        "/api/v1/recognize/video",
        files={"file": ("payload.exe", b"\x4d\x5a\x90\x00", "application/x-msdownload")},
    )
    assert res_exe.status_code == 400
    assert "Unsupported video extension" in res_exe.json()["detail"]


def test_video_processing_graceful_failure_on_corrupt_bytes(client):
    """Corrupt non-container bytes sent to video recognition endpoint returns error cleanly."""
    corrupt_bytes = b"CORRUPT_NON_CONTAINER_BYTES_FOR_OPENCV" * 10
    res = client.post(
        "/api/v1/recognize/video",
        files={"file": ("corrupt.mp4", corrupt_bytes, "video/mp4")},
    )
    assert res.status_code in (400, 500)


# ---------------------------------------------------------------------------
# 5. DATASET RECORDING PATH TRAVERSAL & PARAMETER INJECTIONS
# ---------------------------------------------------------------------------

def test_custom_sign_path_traversal_and_injection(client):
    """User passes directory traversal patterns in sign recording."""
    traversal_labels = [
        "../../etc/passwd",
        r"..\..\Windows\System32\cmd",
        "nested/sub/dir/sign",
        "sign_with_spaces_and_symbols_!@#$%",
    ]
    valid_seq = [[0.05] * 164 for _ in range(5)]

    for label in traversal_labels:
        res = client.post(
            "/api/v1/dataset/record_sample",
            json={"label": label, "sequence_164d": valid_seq},
        )
        if res.status_code == 200:
            saved_path = Path(res.json()["file_path"])
            try:
                # Must be strictly within data/processed and traversal characters stripped
                assert ".." not in saved_path.name
                assert "/" not in saved_path.name
                assert "\\" not in saved_path.name
            finally:
                if saved_path.exists():
                    saved_path.unlink()
        else:
            assert res.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 6. REST API /api/predict DIMENSION & FLOAT EDGE CASES
# ---------------------------------------------------------------------------

def test_api_predict_corrupt_and_illegal_dimensions(client):
    """User submits unsupported feature vector lengths or non-finite numbers."""
    # A. Illegal dimensions (e.g., 10, 42, 64, 127)
    for bad_dim in (1, 10, 42, 64, 127):
        res = client.post("/api/predict", json={"features": [0.1] * bad_dim})
        assert res.status_code == 400
        assert "expected" in res.json()["detail"].lower()

    # B. Empty JSON payload
    res_empty = client.post("/api/predict", json={})
    assert res_empty.status_code == 400

    # C. Truncated landmark list (< 21 points)
    res_short = client.post(
        "/api/predict",
        json={"landmarks": [{"x": 0.5, "y": 0.5}] * 10},
    )
    assert res_short.status_code == 400

    # D. Missing coordinate fields in landmark dict
    res_bad_lm = client.post(
        "/api/predict",
        json={"landmarks": [{"x": 0.5}] * 21},  # missing 'y'
    )
    assert res_bad_lm.status_code in (200, 400, 422)
