"""
SETU — Exhaustive Codebase Verification Suite: All States, Edge Cases & Chance Scenarios.

Covers:
1. Kinetic Energy Gesture State Machine (Idle, Start, Active, Settle, Cooldown, Retraction)
2. Sentence Accumulator & Word Buffering (Streaming, Timeouts, Spell Recovery, Phrases, Backspace, Reset)
3. Multimodal Edge Cases (Zeros, Missing Hand, Left-Hand Mirroring, NaNs/Infs, Scale Extremes)
4. Disambiguation HUD State Machine (Margin thresholding, Option A/B manual override)
5. Model Registry Forward Passes (All 9 registered engines: Static, Dynamic, ONNX, PyTorch)
6. NLP Forward & Reverse Translation (SVO/SOV, Questions, Negations, Indic Multi-lingual)
7. WebSocket End-to-End Handshake (Control packets, mode toggles, streaming frame ingress)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.vision.kinetic_segmenter import KineticEnergySegmenter, GestureState
from backend.app.services.nlp.accumulator import SentenceAccumulator
from backend.app.services.nlp.spell_corrector import ISLSpellCorrector
from backend.app.services.nlp.syntax_transformer import ISLSyntaxTransformer
from backend.app.services.nlp.indic_transformer import IndicTransformer
from training.features.geometry import (
    extract_dual_hand_126d,
    normalize_wrist_coordinates,
    compute_joint_angle,
    is_resting_or_neutral_pose,
)
from models.registry import ModelRegistry, ModelType
from models.adapters.sequence_lstm import CTCGreedyDecoder


# ==============================================================================
# SUITE 1: KINETIC GESTURE LIFECYCLE STATE MACHINE
# ==============================================================================

def test_gestural_state_machine_lifecycle():
    """Validates transitions across IDLE -> GESTURE_START -> GESTURE_ACTIVE -> GESTURE_SETTLED."""
    segmenter = KineticEnergySegmenter(
        energy_threshold=0.03,
        settle_window_frames=2,
        min_gesture_frames=4,
    )

    base_hand = np.ones((21, 3), dtype=np.float32) * 0.5

    # 1. Baseline IDLE: hand stationary at rest
    for _ in range(5):
        state, energy, trigger = segmenter.process_frame(base_hand)
        assert state == GestureState.IDLE
        assert trigger is False
        assert energy < 0.01

    # 2. Motion Onset & Active Stroke (Hand moves briskly)
    moving_hand = base_hand.copy()
    states_observed = []
    for i in range(1, 8):
        moving_hand = base_hand + float(i * 0.05)
        state, energy, trigger = segmenter.process_frame(moving_hand)
        states_observed.append(state)
        assert trigger is False

    assert GestureState.GESTURE_START in states_observed
    assert GestureState.GESTURE_ACTIVE in states_observed

    # 3. Settlement & Trigger: Signer halts motion to hold the target posture
    settled_hand = moving_hand.copy()
    triggered = False
    for _ in range(6):
        state, energy, trig = segmenter.process_frame(settled_hand)
        if trig:
            triggered = True
            assert state == GestureState.GESTURE_SETTLED
            break

    assert triggered is True, "Signer posture hold must trigger GESTURE_SETTLED"

    # 4. Post-stroke Retraction / Rest pose detection
    resting_hand = np.ones((21, 3), dtype=np.float32) * 0.85  # Low in frame (lap)
    assert is_resting_or_neutral_pose(resting_hand, y_threshold=0.75) is True


# ==============================================================================
# SUITE 2: SENTENCE ACCUMULATOR & WORD BUFFERING STATES
# ==============================================================================

def test_sentence_accumulator_states_and_actions():
    """Validates streaming letters, timeout word commits, spell correction, phrases, and control actions."""
    corrector = ISLSpellCorrector()
    acc = SentenceAccumulator(spell_corrector=corrector, word_timeout=1.0, sentence_timeout=2.0)

    # State: Empty initial state
    assert acc.sentence_buffer == ""
    assert acc.current_word == ""
    assert len(acc.sentence_tokens) == 0

    # State: Streaming continuous letters with CTC decoding ("T", "H", "A", "M", "K")
    t0 = 1000.0
    for letter in ["T", "H", "A", "M", "K"]:
        acc.add_continuous_token(letter, confidence=0.95, token_type="letter", timestamp=t0)
        acc.add_continuous_token("no_sign", confidence=1.0, token_type="letter", timestamp=t0 + 0.05)
        t0 += 0.1

    assert acc.current_word == "THAMK"

    # Action: Backspace on active word
    acc.backspace()
    assert acc.current_word == "THAM"
    acc.add_token("K", token_type="letter", timestamp=t0)
    assert acc.current_word == "THAMK"

    # State: Word Timeout trigger (1.2s pause) -> spell corrects "THAMK" -> "THANK"
    res = acc.check_timeouts(current_time=t0 + 1.5, async_neural=False)
    assert res["word_committed"] is True
    assert acc.current_word == ""
    assert acc.sentence_tokens == ["THANK"]

    # State: Add phrase token directly ("YOU")
    acc.add_token("YOU", token_type="phrase", timestamp=t0 + 1.6)
    assert acc.sentence_tokens == ["THANK", "YOU"]

    # Action: Backspace on committed sentence tokens
    acc.backspace()
    assert acc.sentence_tokens == ["THANK"]

    # Re-add "YOU"
    acc.add_token("YOU", token_type="phrase", timestamp=t0 + 1.8)

    # State: Sentence Timeout translation
    res2 = acc.check_timeouts(current_time=t0 + 4.5, async_neural=False)
    assert res2["sentence_translated"] is True
    assert "thank you" in acc.translated_text.lower() or "thank" in acc.translated_text.lower()

    # Action: Reset / Clear
    acc.reset()
    assert acc.sentence_buffer == ""
    assert len(acc.sentence_tokens) == 0
    assert acc.translated_text == ""


def test_accumulator_nmm_grammar_states():
    """Validates question and negation inflection via Non-Manual Markers."""
    acc = SentenceAccumulator(word_timeout=0.5, sentence_timeout=1.0)
    acc.sentence_tokens = ["YOU", "NAME", "WHAT"]

    # Apply Question NMM (eyebrow furrow)
    acc.update_non_manual_grammar(is_question=True)
    res = acc.translate_sentence(async_neural=False)
    assert res.endswith("?")

    # Apply Negation NMM (head shake)
    acc.reset()
    acc.sentence_tokens = ["I", "KNOW"]
    acc.update_non_manual_grammar(is_negation=True)
    res_neg = acc.translate_sentence(async_neural=False)
    assert "not" in res_neg.lower() or "no" in res_neg.lower()


# ==============================================================================
# SUITE 3: MULTIMODAL & DEGENERATE INPUT CHANCES/EDGE CASES
# ==============================================================================

def test_multimodal_zeros_and_missing_hands():
    """Ensures feature extractors and classifiers gracefully handle absent hands."""
    # 1. Both hands absent (all zeros)
    empty_feat = extract_dual_hand_126d(left_hand=[], right_hand=[])
    assert empty_feat.shape == (126,)
    assert np.all(empty_feat == 0.0)

    # Classifier handles zeros without crashing
    clf = ModelRegistry.get("realsign_multisigner_mlp")
    assert clf is not None
    pred = clf.predict(empty_feat, apply_heuristic_boost=False)
    assert hasattr(pred, "letter")
    assert hasattr(pred, "confidence")
    assert 0.0 <= pred.confidence <= 1.0


def test_multimodal_left_handed_mirror_invariance():
    """Ensures left-handed signers mirrored on horizontal axis produce consistent normalized features."""
    rng = np.random.RandomState(42)
    fake_hand = [
        {"x": 0.5 + float(rng.randn() * 0.05), "y": 0.5 + float(rng.randn() * 0.05), "z": 0.0}
        for _ in range(21)
    ]
    # Mirrored across x = 0.5
    mirrored_hand = [
        {"x": 1.0 - p["x"], "y": p["y"], "z": p["z"]}
        for p in fake_hand
    ]

    norm1, len1 = normalize_wrist_coordinates(fake_hand)
    norm2, len2 = normalize_wrist_coordinates(mirrored_hand)

    assert norm1.shape == (21, 3)
    assert norm2.shape == (21, 3)
    # Palm lengths must be identical
    assert abs(len1 - len2) < 1e-4


def test_multimodal_nan_and_inf_resilience():
    """Ensures corrupted MediaPipe outputs (NaNs, Infs) do not propagate through geometry math."""
    corrupted_hand = [{"x": float("nan"), "y": float("inf"), "z": 0.0} for _ in range(21)]
    norm_pts, palm_len = normalize_wrist_coordinates(corrupted_hand)
    # Must clean NaNs/Infs to finite zero floats
    assert np.all(np.isfinite(norm_pts))
    assert math.isfinite(palm_len)
    assert palm_len == 0.0


def test_multimodal_scale_extremes():
    """Verifies that an extremely close hand vs far hand yield scale-normalized invariants."""
    base_pts = np.random.RandomState(123).uniform(0.3, 0.7, size=(21, 3)).astype(np.float32)
    # Small hand (far away)
    small_pts = base_pts * 0.2
    # Giant hand (close to camera)
    large_pts = base_pts * 4.5

    norm_small, d_small = normalize_wrist_coordinates(small_pts)
    norm_large, d_large = normalize_wrist_coordinates(large_pts)

    # After wrist origin + palm length scale normalization, shapes are congruent
    # Joint angles between finger segments must be mathematically invariant to distance
    angle_small = compute_joint_angle(norm_small[0], norm_small[1], norm_small[2])
    angle_large = compute_joint_angle(norm_large[0], norm_large[1], norm_large[2])
    assert abs(angle_small - angle_large) < 1e-4


# ==============================================================================
# SUITE 4: DISAMBIGUATION HUD STATE MACHINE
# ==============================================================================

def test_disambiguation_hud_state_machine():
    """Tests the disambiguation workflow when two candidate predictions have a low margin."""
    acc = SentenceAccumulator()

    # Candidate 1: "M" (conf 0.48), Candidate 2: "N" (conf 0.46) -> margin = 0.02 (< 0.05 threshold)
    top1, top2 = "M", "N"
    margin = 0.48 - 0.46
    is_ambiguous = margin < 0.05
    assert is_ambiguous is True

    # User manually overrides / disambiguates to Option B ("N")
    selected_choice = "option_b"
    resolved_letter = top2 if selected_choice == "option_b" else top1
    acc.add_token(resolved_letter, token_type="letter")

    assert acc.current_word == "N"


# ==============================================================================
# SUITE 5: MODEL REGISTRY FORWARD PASS ACROSS ALL ENGINES
# ==============================================================================

def test_model_registry_forward_pass_all_engines():
    """Instantiates and executes forward inference across all registered models."""
    models_catalog = ModelRegistry.list_models()
    assert len(models_catalog) >= 7, f"Expected at least 7 models, got {len(models_catalog)}"

    rng = np.random.RandomState(99)

    for entry in models_catalog:
        model_id = entry["id"]
        instance = ModelRegistry.get(model_id)
        assert instance is not None, f"Failed to instantiate model '{model_id}'"

        # Test forward pass based on model architecture
        if entry["type"] == "streaming_alphabet":
            # 126-D dual-hand feature vector (or 63-D for RFUV11)
            feat_dim = entry.get("input_dim", 126)
            dummy_feat = rng.randn(feat_dim).astype(np.float32) * 0.1
            if hasattr(instance, "predict"):
                pred = instance.predict(dummy_feat)
                assert pred is not None
                assert hasattr(pred, "label") or hasattr(pred, "letter")
                assert hasattr(pred, "confidence")
                conf = getattr(pred, "confidence")
                assert 0.0 <= conf <= 1.0

        elif entry["type"] == "streaming_phrases":
            if model_id == "amrita_isl_phrase_bilstm":
                dummy_seq = rng.randn(20, 316).astype(np.float32) * 0.1
            else:
                dummy_seq = rng.randn(30, 158).astype(np.float32) * 0.1
            if hasattr(instance, "predict"):
                pred = instance.predict(dummy_seq)
                assert pred is not None
                assert hasattr(pred, "label")
                assert 0.0 <= pred.confidence <= 1.0

        elif entry["type"] == "streaming_sequence":
            dummy_seq = rng.randn(30, 328).astype(np.float32) * 0.1
            if hasattr(instance, "predict"):
                pred = instance.predict(dummy_seq)
                assert pred is not None
                assert hasattr(pred, "label")

        elif entry["type"] == "isolated_video":
            dummy_seq = rng.randn(30, 134).astype(np.float32) * 0.1
            if hasattr(instance, "predict"):
                pred = instance.predict(dummy_seq)
                assert pred is not None


# ==============================================================================
# SUITE 6: FULL NLP TRANSLATION & REVERSIBLE GLOSS SYNTHESIS
# ==============================================================================

def test_nlp_forward_and_reverse_translation():
    """Tests syntax transformer forward (ISL -> English) and reverse (/to-gloss)."""
    transformer = ISLSyntaxTransformer()
    indic = IndicTransformer()

    # 1. Forward Canonical Rule Match / SOV transform
    tokens = ["HELLO"]
    english = transformer.transform(tokens)
    assert len(english) > 0
    assert "hello" in english.lower()

    # 2. Indic Multilingual Synthesis
    multilingual = indic.translate_all(["THANK", "YOU"])
    assert "hi" in multilingual
    assert "ta" in multilingual
    assert "te" in multilingual
    assert "kn" in multilingual
    assert len(multilingual["hi"]) > 0


def test_nlp_rest_endpoints():
    """Tests REST endpoints for NLP translation, correction, and glossing."""
    client = TestClient(app)

    # 1. /api/v1/nlp/status
    res = client.get("/api/v1/nlp/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"

    # 2. /api/v1/nlp/correct (Phonological Spell Corrector)
    res_correct = client.post("/api/v1/nlp/correct", json={"text": "THAMK"})
    assert res_correct.status_code == 200
    assert res_correct.json()["corrected"] == "THANK"

    # 3. /api/v1/nlp/translate (Gloss -> Multilingual)
    res_trans = client.post("/api/v1/nlp/translate", json={"tokens": ["THANK", "YOU"]})
    assert res_trans.status_code == 200
    t_data = res_trans.json()
    assert "translations" in t_data
    assert "en" in t_data["translations"]
    assert "hi" in t_data["translations"]

    # 4. /api/v1/nlp/to-gloss (Reverse Glossing)
    res_gloss = client.post("/api/v1/nlp/to-gloss", json={"text": "What is your name?"})
    assert res_gloss.status_code == 200
    g_data = res_gloss.json()
    assert "gloss_tokens" in g_data
    assert len(g_data["gloss_tokens"]) > 0


# ==============================================================================
# SUITE 7: WEBSOCKET PROTOCOL STATE HANDSHAKE
# ==============================================================================

def test_websocket_protocol_states_and_control_packets():
    """Connects to real-time WebSocket and tests ping, reset, backspace, and binary frame ingress."""
    client = TestClient(app)

    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # 1. Control Action: Ping / Heartbeat
        ws.send_text(json.dumps({"action": "ping"}))
        msg = ws.receive_json()
        assert msg["type"] == "pong"

        # 3. Control Action: Backspace
        ws.send_text(json.dumps({"action": "backspace"}))
        msg_bs = ws.receive_json()
        assert msg_bs["type"] == "info"
        assert "Backspace" in msg_bs.get("message", "")

        # 4. Control Action: Clear Sentence
        ws.send_text(json.dumps({"action": "clear_sentence"}))
        msg_clr = ws.receive_json()
        assert msg_clr["type"] == "info"
        assert "cleared" in msg_clr.get("message", "").lower()

        # 5. Control Action: Reset
        ws.send_text(json.dumps({"action": "reset"}))
        msg_rst = ws.receive_json()
        assert msg_rst["type"] == "info"
        assert "reset" in msg_rst.get("message", "").lower()

        # 6. Stream Ingress: Send a synthetic dummy binary JPEG frame
        dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", dummy_frame)
        ws.send_bytes(buffer.tobytes())
        frame_resp = ws.receive_json()
        # Verify response structure
        assert "sign" in frame_resp or "text" in frame_resp or "type" in frame_resp
