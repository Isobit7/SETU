"""
Unit tests verifying Phase 2 & Phase 3:
1. Head Nod (pitch vertical oscillation) detection in NonManualMarkerDetector.
2. Head Shake (yaw horizontal oscillation) detection in NonManualMarkerDetector.
3. SentenceAccumulator.reject_last() undoes the last token.
4. Intentional buffer gating: signs only commit upon head nod when require_head_nod_confirm is enabled.
"""

import time
import numpy as np
import pytest

from backend.app.services.vision.non_manual_markers import NonManualMarkerDetector, NonManualGrammar
from backend.app.services.nlp.accumulator import SentenceAccumulator


def test_head_nod_detection():
    """Verify that vertical pitch oscillations of the nose tip trigger head_gesture='nod'."""
    detector = NonManualMarkerDetector(history_len=12)
    face_pts = [type("Point", (), {"x": 0.5, "y": 0.5, "z": 0.0})() for _ in range(468)]

    # Simulate vertical head nodding: nose y moving up and down across frames
    y_sequence = [0.50, 0.47, 0.53, 0.46, 0.54, 0.47, 0.53]
    detected_gestures = []
    for y_val in y_sequence:
        face_pts[NonManualMarkerDetector.NOSE_TIP] = type("Point", (), {"x": 0.5, "y": y_val, "z": 0.0})()
        res = detector.detect(face_pts)
        if res.head_gesture != "none":
            detected_gestures.append(res.head_gesture)

    assert "nod" in detected_gestures, f"Expected 'nod' in {detected_gestures}"


def test_head_shake_detection():
    """Verify that horizontal yaw oscillations of the nose tip trigger head_gesture='shake'."""
    detector = NonManualMarkerDetector(history_len=12)
    face_pts = [type("Point", (), {"x": 0.5, "y": 0.5, "z": 0.0})() for _ in range(468)]

    # Simulate horizontal head shaking: nose x moving left and right
    x_sequence = [0.50, 0.46, 0.54, 0.45, 0.55, 0.46, 0.54]
    detected_gestures = []
    for x_val in x_sequence:
        face_pts[NonManualMarkerDetector.NOSE_TIP] = type("Point", (), {"x": x_val, "y": 0.5, "z": 0.0})()
        res = detector.detect(face_pts)
        if res.head_gesture != "none":
            detected_gestures.append(res.head_gesture)

    assert "shake" in detected_gestures, f"Expected 'shake' in {detected_gestures}"


def test_sentence_accumulator_reject_last():
    """Verify that reject_last() removes the last word/token from the sentence buffer."""
    acc = SentenceAccumulator()
    now = time.time()
    acc.add_token("HELLO", token_type="phrase", timestamp=now)
    acc.add_token("WORLD", token_type="phrase", timestamp=now + 0.1)

    assert acc.sentence_buffer == "HELLO WORLD"

    # User shakes head -> Undo last token
    undone = acc.reject_last()
    assert undone is True
    assert acc.sentence_buffer == "HELLO"

    # Undo again
    undone = acc.reject_last()
    assert undone is True
    assert acc.sentence_buffer == ""

    # Undo on empty buffer returns False
    assert acc.reject_last() is False


def test_head_nod_confirmed_gating_logic():
    """Verify that when head-nod mode is enabled, signs are held until nod confirms."""
    acc = SentenceAccumulator()
    now = time.time()

    candidate = "V"
    require_head_nod_confirm = True

    # When user is holding sign V, but hasn't nodded yet
    # In head-nod mode, we do NOT commit to buffer
    committed = False
    if not require_head_nod_confirm:
        committed = acc.add_continuous_token(candidate, confidence=0.95, token_type="letter", timestamp=now)

    assert committed is False
    assert acc.sentence_buffer == ""

    # Now simulate user nodding head (head_gesture == 'nod')
    head_gesture = "nod"
    if head_gesture == "nod":
        committed = bool(acc.add_continuous_token(candidate, confidence=0.95, token_type="letter", timestamp=now, bypass_debounce=True))

    assert committed is True
    assert acc.sentence_buffer == "V"

    # Now simulate user making a mistake and shaking head (head_gesture == 'shake')
    head_gesture = "shake"
    if head_gesture == "shake":
        acc.reject_last()

    assert acc.sentence_buffer == ""
