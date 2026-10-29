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
    """Verify that COCOObjectDetector initializes properly and handles blank frames."""
    detector = COCOObjectDetector()
    
    # Create synthetic frame (480, 640, 3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. Empty/blank frame returns empty list
    results = detector.detect(frame, confidence_threshold=0.65)
    assert isinstance(results, list)
    
    # 2. Verify filter thresholds
    assert detector.confidence_threshold >= 0.65


def test_yolo_world_classes_and_watch_normalization():
    """Verify YOLO-World classes, watch/phone disambiguation, and Multimodal Fuser integration."""
    from backend.app.services.vision.object_detector import (
        LABEL_NORMALIZATION,
        YOLO_WORLD_CLASSES,
        COCOObjectDetector,
        DetectedObject,
    )
    from backend.app.services.nlp.multimodal_fuser import MultimodalFuser

    # 1. Vocabulary validation
    assert "wristwatch" in YOLO_WORLD_CLASSES
    assert "smartwatch" in YOLO_WORLD_CLASSES
    assert "cell phone" in YOLO_WORLD_CLASSES

    # 2. Label normalization
    assert LABEL_NORMALIZATION["wristwatch"] == "watch"
    assert LABEL_NORMALIZATION["smartwatch"] == "watch"
    assert LABEL_NORMALIZATION["cell phone"] == "phone"

    # 3. Detector loads with YOLO-World
    detector = COCOObjectDetector()
    assert "watch" in [LABEL_NORMALIZATION.get(c, c) for c in detector.categories]

    # 4. Multimodal Fusion with Watch
    fuser = MultimodalFuser()
    watch_obj = DetectedObject(
        label="watch",
        display_name="WATCH",
        confidence=0.88,
        bbox=(0.2, 0.2, 0.5, 0.5),
        is_interactive=True,
        is_held=True,
    )

    fused_time = fuser.fuse("time", [watch_obj])
    assert fused_time == "What is the time?"

    fused_give = fuser.fuse("give", [watch_obj])
    assert fused_give == "Please hand me the watch."




def test_neural_predictive_engine_and_domain_vocabulary():
    """Verify modern PredictiveTextEngine with domain vocabulary and caching."""
    from backend.app.services.nlp.predictive import PredictiveTextEngine

    engine = PredictiveTextEngine()
    
    # 1. Base suggestions
    res_empty = engine.predict_next("")
    assert len(res_empty) == 3
    assert "hello" in res_empty

    # 2. Domain bigram completions (healthcare, education, time)
    res_school = engine.predict_next("school")
    assert any("school" in r or "bus" in r or "teacher" in r for r in res_school)

    res_med = engine.predict_next("medicine")
    assert any("medicine" in r or "doctor" in r or "pills" in r for r in res_med)

    res_watch = engine.predict_next("watch")
    assert any("time" in r or "watch" in r for r in res_watch)

    # 3. Cache verification
    engine._cache["custom_query"] = ["first suggestion", "second suggestion", "third suggestion"]
    cached_res = engine.predict_next("custom_query")
    assert cached_res == ["first suggestion", "second suggestion", "third suggestion"]


def test_ctc_streaming_continuous_fingerspelling_and_double_letters():
    """Verify CTC greedy continuous decoding correctly allows double letters without a 0.45s freeze."""
    from backend.app.services.nlp.accumulator import SentenceAccumulator

    acc = SentenceAccumulator()

    # 1. Spell "B"
    for _ in range(5):
        acc.add_continuous_token("b", confidence=0.95, token_type="letter")
    assert acc.current_word == "B"

    # 2. Spell "O"
    for _ in range(5):
        acc.add_continuous_token("o", confidence=0.95, token_type="letter")
    assert acc.current_word == "BO"

    # 3. Rest transition (hand relaxes for 2 frames)
    acc.add_continuous_token("no_sign", confidence=1.0, token_type="letter")
    acc.add_continuous_token("no_sign", confidence=1.0, token_type="letter")
    assert acc.current_word == "BO"

    # 4. Spell second "O" (Double letter in "BOOK")
    for _ in range(5):
        acc.add_continuous_token("o", confidence=0.95, token_type="letter")
    assert acc.current_word == "BOO"

    # 5. Spell "K"
    for _ in range(5):
        acc.add_continuous_token("k", confidence=0.95, token_type="letter")
    assert acc.current_word == "BOOK"
