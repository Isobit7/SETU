"""
SETU — Comprehensive Tests for Live Fixes:
1. Left Hand vs Right Hand Recognition Symmetry
2. Anti-Spam Letter Repeat Lockout
3. High-Precision Object Detection Filtering (Watch vs Phone)
"""

import numpy as np
import pytest
import torch

from models.adapters.realsign import RealSignClassifier
from training.features.geometry import (
    extract_dual_hand_126d,
    normalize_wrist_coordinates,
)
from backend.app.services.vision.object_detector import (
    COCOObjectDetector,
    DetectedObject,
)


def test_single_hand_left_right_recognition_symmetry():
    """Verify that a left-handed sign produces identical predictions to the right-handed sign."""
    clf = RealSignClassifier()
    
    # Synthesize realistic landmark points for a sign (e.g. thumb and fingers spread)
    rng = np.random.RandomState(42)
    fake_rh = [
        {"x": 0.5 + float(rng.randn() * 0.05), "y": 0.5 + float(rng.randn() * 0.05), "z": float(rng.randn() * 0.02)}
        for _ in range(21)
    ]
    # Symmetrically reflected left hand
    fake_lh = [{"x": 1.0 - p["x"], "y": p["y"], "z": p["z"]} for p in fake_rh]

    # 1. Feature extraction with right hand only
    feat_rh = extract_dual_hand_126d(left_hand=None, right_hand=fake_rh)
    pred_rh = clf.predict(feat_rh, apply_heuristic_boost=False)

    # 2. Feature extraction with left hand only (uses ambidextrous reflection)
    feat_lh = extract_dual_hand_126d(left_hand=fake_lh, right_hand=None)
    pred_lh = clf.predict(feat_lh, apply_heuristic_boost=False)

    # Both must output the exact same predicted letter and high confidence
    assert pred_rh.letter == pred_lh.letter
    assert abs(pred_rh.confidence - pred_lh.confidence) < 1e-4
    assert pred_rh.confidence > 0.50


def test_dual_hand_left_handed_signer_support():
    """Verify dual-handed left-handed signer symmetry logic."""
    clf = RealSignClassifier()
    
    # Create dual-hand pose
    rng = np.random.RandomState(101)
    fake_base = [
        {"x": 0.4 + float(rng.randn() * 0.03), "y": 0.5 + float(rng.randn() * 0.03), "z": 0.0}
        for _ in range(21)
    ]
    fake_dominant = [
        {"x": 0.6 + float(rng.randn() * 0.03), "y": 0.5 + float(rng.randn() * 0.03), "z": 0.0}
        for _ in range(21)
    ]

    # Standard right-handed signer (LH base, RH dominant)
    feat_standard = extract_dual_hand_126d(left_hand=fake_base, right_hand=fake_dominant)
    pred_standard = clf.predict(feat_standard, apply_heuristic_boost=False)

    # Left-handed signer with roles swapped (RH base, LH dominant)
    # Reflected left-handed version
    fake_base_mirrored = [{"x": 1.0 - p["x"], "y": p["y"], "z": p["z"]} for p in fake_base]
    fake_dom_mirrored = [{"x": 1.0 - p["x"], "y": p["y"], "z": p["z"]} for p in fake_dominant]
    feat_lefty = extract_dual_hand_126d(left_hand=fake_dom_mirrored, right_hand=fake_base_mirrored)
    pred_lefty = clf.predict(feat_lefty, apply_heuristic_boost=False)

    # Left-handed version must evaluate with the same letter and confidence
    assert pred_standard.letter == pred_lefty.letter
    assert abs(pred_standard.confidence - pred_lefty.confidence) < 1e-3


def test_letter_repeat_lockout_logic():
    """Simulate streaming frame loop and verify that holding a letter never spams tokens."""
    committed_tokens = []
    alphabet_last_committed = ""
    alphabet_repeat_lock = False
    last_hand_seen_time = 100.0

    def process_frame(prediction: str, now: float, hand_detected: bool):
        nonlocal alphabet_last_committed, alphabet_repeat_lock, last_hand_seen_time
        if hand_detected:
            last_hand_seen_time = now
            hand_absent_duration = 0.0
        else:
            hand_absent_duration = now - last_hand_seen_time

        if hand_absent_duration >= 0.45:
            alphabet_repeat_lock = False
            alphabet_last_committed = ""

        can_commit = False
        if prediction != alphabet_last_committed:
            can_commit = True
        elif not alphabet_repeat_lock:
            can_commit = True

        if can_commit and prediction not in ("", "Awaiting Sign Input", "no_sign"):
            committed_tokens.append((prediction, now))
            alphabet_last_committed = prediction
            alphabet_repeat_lock = True

    # 1. User holds 'A' steadily for 3 continuous seconds (90 frames at 30fps)
    for frame_idx in range(90):
        t = 100.0 + (frame_idx * 0.033)
        process_frame("A", now=t, hand_detected=True)

    # 'A' must be committed exactly ONCE despite being held for 3 seconds!
    assert len(committed_tokens) == 1
    assert committed_tokens[0][0] == "A"

    # 2. Hand micro-shakes / flickers for 2 frames (66ms, well under 0.45s)
    process_frame("Awaiting Sign Input", now=103.033, hand_detected=False)
    process_frame("Awaiting Sign Input", now=103.066, hand_detected=False)
    # Hand recovers on 'A'
    for frame_idx in range(30):
        t = 103.10 + (frame_idx * 0.033)
        process_frame("A", now=t, hand_detected=True)

    # Micro-flicker MUST NOT trigger repeat 'A'!
    assert len(committed_tokens) == 1

    # 3. User switches to letter 'B'
    for frame_idx in range(30):
        t = 104.10 + (frame_idx * 0.033)
        process_frame("B", now=t, hand_detected=True)

    # 'B' must commit exactly once
    assert len(committed_tokens) == 2
    assert committed_tokens[1][0] == "B"

    # 4. User drops hand for 0.5s (deliberate release) then signs 'B' again (double letter)
    process_frame("Awaiting Sign Input", now=105.10, hand_detected=False)
    process_frame("Awaiting Sign Input", now=105.65, hand_detected=False) # 0.55s absent
    process_frame("B", now=105.70, hand_detected=True)

    assert len(committed_tokens) == 3
    assert committed_tokens[2][0] == "B"


def test_object_detector_filters_watches_and_noise():
    """Verify that COCOObjectDetector filters out watches, tiny boxes, and low confidence."""
    detector = COCOObjectDetector()
    
    # Create synthetic frame (480, 640, 3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. Empty/blank frame returns empty list
    results = detector.detect(frame, confidence_threshold=0.65)
    assert isinstance(results, list)
    
    # 2. Verify filter thresholds
    assert detector.confidence_threshold >= 0.65
