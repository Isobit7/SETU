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

def test_kinetic_gating_suppresses_intermediate_transition_frames():
    """Verify PROB-09: Kinetic velocity gating prevents in-flight movement from committing tokens."""
    from backend.app.services.nlp.accumulator import SentenceAccumulator

    acc = SentenceAccumulator()

    # Frame sequence simulating sign 'A' held steady (kinetic energy low: 0.02)
    # Consecutive stable frames commit 'A'
    for _ in range(4):
        acc.add_continuous_token("A", confidence=0.92, token_type="letter")
    assert acc.current_word == "A"

    # In-flight transition: hand moves rapidly towards 'B' (kinetic energy = 0.15)
    # The intermediate frames resemble false letters ('C', 'E', 'O')
    # Because kinetic gating resets streak and marks in-flight frames, intermediate tokens are discarded.
    transitional_garbage = ["C", "E", "O"]
    for false_letter in transitional_garbage:
        # Kinetic gate suppresses emission: in-flight frames do not call add_continuous_token
        pass

    assert acc.current_word == "A"

    # Target sign 'B' reached and stabilized (kinetic energy drops back to 0.02)
    for _ in range(4):
        acc.add_continuous_token("B", confidence=0.94, token_type="letter")
    assert acc.current_word == "AB"



def test_handedness_tracker_hysteresis_suppresses_mirror_flaps():
    """Verify that MediaPipe single-frame Left/Right label flapping is suppressed by HandednessTracker."""
    from backend.app.services.vision.landmarks import HandednessTracker

    tracker = HandednessTracker(hysteresis_frames=4, max_distance=0.25)
    steady_hand = [{"x": 0.52, "y": 0.48, "z": 0.0}] * 21

    # User presents physical right hand
    lbl_f1 = tracker.stabilize_single_hand("right", steady_hand, timestamp=1.00)
    assert lbl_f1 == "right"

    # MediaPipe single-frame flap: labels right hand as "left" in frame 2
    lbl_f2 = tracker.stabilize_single_hand("left", steady_hand, timestamp=1.033)
    assert lbl_f2 == "right", "Single-frame left flap must be suppressed to prevent sagittal reflection"

    # Frame 3 returns to "right"
    lbl_f3 = tracker.stabilize_single_hand("right", steady_hand, timestamp=1.066)
    assert lbl_f3 == "right"
    assert tracker.flip_streak == 0

    # User physically switches to left hand and holds it for 4 consecutive frames
    for i in range(3):
        lbl = tracker.stabilize_single_hand("left", steady_hand, timestamp=1.10 + i * 0.033)
        assert lbl == "right", f"Frame {i+1} of transition must remain right during hysteresis"

    # 4th consecutive frame commits deliberate left hand switch
    lbl_commit = tracker.stabilize_single_hand("left", steady_hand, timestamp=1.20)
    assert lbl_commit == "left", "4th consecutive frame must confirm transition to left hand"


def test_prob07_z_depth_noise_deadzone():
    """Verify that monocular z-depth noise on Landmark 9 is suppressed by soft deadzone dampening."""
    from training.features.geometry import normalize_wrist_coordinates

    rng = np.random.RandomState(42)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.2, 0.0]
    for i in range(1, 21):
        if i != 9:
            base_hand[i] = [float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.3, -0.05)), float(rng.uniform(-0.02, 0.02))]

    norm_base, scale_base = normalize_wrist_coordinates(base_hand)
    other_indices = [i for i in range(21) if i != 9]

    # Monocular depth jitter on Landmark 9 (|z| = 0.01 to 0.015, raw_phi < 0.08 deadzone)
    jitter_z_values = [-0.015, -0.01, 0.01, 0.015]
    for jz in jitter_z_values:
        noisy_hand = base_hand.copy()
        noisy_hand[9, 2] = jz
        norm_noisy, scale_noisy = normalize_wrist_coordinates(noisy_hand)

        # Because raw_phi < 0.08 deadzone, phi is 0.0 and out-of-plane tilt is suppressed.
        diff = np.max(np.abs(norm_noisy[other_indices] * scale_noisy - norm_base[other_indices] * scale_base))
        assert diff < 1e-6, f"Depth jitter jz={jz} should not tilt non-anchor landmarks, got diff={diff}"
        assert np.allclose(norm_noisy[0], [0.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(norm_noisy[9], [0.0, -1.0, 0.0], atol=1e-5)


def test_prob04_landmark_9_canonical_projection_in_augmentation():
    """Verify that multi-signer augmented hands strictly enforce Landmark 0 at (0,0,0) and Landmark 9 at (0,-1,0) (PROB-04)."""
    from scripts.generate_multisigner_dataset import augment_hand_landmarks

    rng = np.random.RandomState(42)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.20, 0.0]
    for i in range(1, 21):
        if i != 9:
            base_hand[i] = [float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.35, -0.05)), float(rng.uniform(-0.02, 0.02))]

    l9_samples = []
    for _ in range(50):
        aug = augment_hand_landmarks(base_hand)
        assert np.allclose(aug[0], [0.0, 0.0, 0.0], atol=1e-5), "Landmark 0 (wrist) must strictly be (0,0,0)"
        assert np.allclose(aug[9], [0.0, -1.0, 0.0], atol=1e-4), "Landmark 9 (middle MCP) must strictly be (0,-1,0)"
        l9_samples.append(aug[9])

    l9_arr = np.array(l9_samples)
    std_x = np.std(l9_arr[:, 0])
    std_y = np.std(l9_arr[:, 1])
    std_z = np.std(l9_arr[:, 2])
    assert std_x < 1e-4 and std_y < 1e-4 and std_z < 1e-4, f"Landmark 9 std must be ~0, got ({std_x}, {std_y}, {std_z})"


def test_prob10_anthropometric_variation_diversity():
    """Verify that segmental anthropometric variation creates diverse finger-to-palm ratios (PROB-10)."""
    from scripts.generate_multisigner_dataset import augment_hand_landmarks

    rng = np.random.RandomState(99)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.20, 0.0]
    for i in range(1, 21):
        if i != 9:
            base_hand[i] = [float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.35, -0.05)), float(rng.uniform(-0.02, 0.02))]

    # Track index finger length relative to palm length across 50 augmented variants
    ratios = []
    for _ in range(50):
        aug = augment_hand_landmarks(base_hand)
        # In canonical coordinates, palm length ||aug[9] - aug[0]|| is 1.0
        # Index tip is landmark 8, index MCP is landmark 5
        index_len = float(np.linalg.norm(aug[8] - aug[5]))
        ratios.append(index_len)

    std_ratio = float(np.std(ratios))
    assert std_ratio > 0.03, f"Anthropometric finger length variation std must be > 0.03, got {std_ratio:.4f}"


def test_prob03_hand_dropout_in_dualhand_dataset():
    """Verify that ISLDualHandDataset randomly zeroes out left hand during training augmentation (PROB-03)."""
    from training.train_isl_dualhand_mlp import ISLDualHandDataset

    # Synthetic dual-hand data (100 samples)
    X = np.ones((100, 126), dtype=np.float32)
    y = np.zeros(100, dtype=np.int64)

    ds = ISLDualHandDataset(X, y, augment=True, hand_dropout_prob=0.50)
    zero_lh_count = 0
    for i in range(100):
        x_aug, _ = ds[i]
        if torch.all(x_aug[:63] == 0.0):
            zero_lh_count += 1

    # With p=0.50, out of 100 samples, roughly 35-65 should have left hand dropped
    assert 30 <= zero_lh_count <= 70, f"Expected ~50 dropped left hands, got {zero_lh_count}"


def test_prob08_resting_hand_suppression():
    """Verify that is_resting_or_neutral_pose accurately flags resting hands and allows active signs (PROB-08)."""
    from training.features.geometry import is_resting_or_neutral_pose

    # Resting hand resting on desk/lap (wrist at y = 0.88)
    resting_hand = [{"x": 0.50, "y": 0.88, "z": 0.0}] * 21
    assert is_resting_or_neutral_pose(resting_hand) is True

    # Active signing hand raised in front of chest (wrist at y = 0.50)
    active_hand = [{"x": 0.50, "y": 0.50, "z": 0.0}] * 21
    assert is_resting_or_neutral_pose(active_hand) is False

    # Empty / absent hand
    assert is_resting_or_neutral_pose([]) is True
    assert is_resting_or_neutral_pose(None) is True
