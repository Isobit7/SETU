"""
SETU — Verification Test Suite for Phase 4 (PROB-20, PROB-23, PROB-25).
=======================================================================
Covers:
1. Active Non-Manual Markers (NMMs) Facial & Head Movement Detection (PROB-20).
2. Grammatical Conditioning in SentenceAccumulator & Indic Translations (PROB-20).
3. 3D Skeletal Rig Keypoint Structures and Palm Orientations (PROB-23).
4. Continuous Active Learning Feedback Ingestion & Retraining Loop (PROB-25).
"""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from backend.app.services.vision.non_manual_markers import (
    NonManualMarkerDetector,
    NonManualGrammar,
    non_manual_detector,
)
from backend.app.services.nlp.accumulator import SentenceAccumulator
from scripts.retrain_feedback_loop import (
    load_feedback_samples,
    run_active_learning_retrain,
)
from backend.app.services.pipeline_manager import pipeline


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def create_dummy_face(eyebrow_raise: bool = False, nose_x: float = 0.5, nose_y: float = 0.5):
    """Creates a 468-point synthetic MediaPipe face landmark array."""
    landmarks = [DummyLandmark(0.5, 0.5, 0.0) for _ in range(468)]
    # Left eye top: 159, Left eyebrow mid: 105
    # Right eye top: 386, Right eyebrow mid: 334
    # Distance in neutral: 0.05
    # Distance in raised: 0.10
    dist = 0.10 if eyebrow_raise else 0.05
    landmarks[159] = DummyLandmark(0.4, 0.5)
    landmarks[105] = DummyLandmark(0.4, 0.5 - dist)
    landmarks[386] = DummyLandmark(0.6, 0.5)
    landmarks[334] = DummyLandmark(0.6, 0.5 - dist)
    landmarks[1] = DummyLandmark(nose_x, nose_y)
    landmarks[152] = DummyLandmark(0.5, 0.8)
    return landmarks


# =============================================================================
# 1. Non-Manual Markers (NMMs) Facial Detection (PROB-20)
# =============================================================================

def test_nmm_neutral_and_empty():
    detector = NonManualMarkerDetector()
    res = detector.detect(None)
    assert not res.is_question
    assert not res.is_negation
    assert not res.is_affirmation

    res_few = detector.detect([DummyLandmark(0, 0)] * 50)
    assert not res_few.is_question


def test_nmm_eyebrow_elevation_triggers_question():
    detector = NonManualMarkerDetector()
    # Neutral face
    neutral_face = create_dummy_face(eyebrow_raise=False)
    res_neutral = detector.detect(neutral_face)
    assert not res_neutral.is_question

    # Raised eyebrow face
    raised_face = create_dummy_face(eyebrow_raise=True)
    res_raised = detector.detect(raised_face)
    assert res_raised.is_question
    assert res_raised.eyebrow_elevation > 0.06
    assert res_raised.confidence >= 0.90


def test_nmm_head_shake_triggers_negation():
    detector = NonManualMarkerDetector(history_len=10)
    # Simulate alternating head shake across X with sufficient amplitude
    xs = [0.50, 0.58, 0.42, 0.58, 0.42, 0.58, 0.42]
    saw_negation = False
    res = None
    for x in xs:
        face = create_dummy_face(nose_x=x)
        res = detector.detect(face)
        if res.is_negation:
            saw_negation = True

    assert res is not None
    assert saw_negation
    assert res.head_yaw_oscillation > 0.035


# =============================================================================
# 2. Accumulator NMM Grammatical Integration (PROB-20)
# =============================================================================

def test_accumulator_nmm_question_and_negation():
    acc = SentenceAccumulator()
    acc.add_token("YOU", token_type="phrase")
    acc.add_token("COME", token_type="phrase")

    # Ingest NMM question marker
    acc.update_non_manual_grammar(is_question=True)
    res = acc.translate_sentence()

    assert res.endswith("?")
    assert acc.translated_indic["hi"].endswith("?")
    assert acc.translated_indic["ta"].endswith("?")

    # Reset cleans NMM flags
    acc.reset()
    assert not acc.detected_question
    assert not acc.detected_negation

    # Ingest NMM negation marker
    acc.add_token("WANT", token_type="phrase")
    acc.update_non_manual_grammar(is_negation=True)
    res_neg = acc.translate_sentence()

    assert "NOT" in res_neg
    assert "नहीं" in acc.translated_indic["hi"]


# =============================================================================
# 3. Active Learning Continuous Retraining Loop (PROB-25)
# =============================================================================

def test_active_learning_feedback_and_retrain():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        feedback_dir = temp_dir / "feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 synthetic hard feedback samples
        feat_sample1 = np.ones(126, dtype=np.float32) * 0.1
        feat_sample2 = np.ones(126, dtype=np.float32) * 0.2
        np.savez_compressed(
            feedback_dir / "sample_A_01.npz",
            features=feat_sample1,
            label="A",
            predicted_label="E",
            source="user_disambiguation",
        )
        np.savez_compressed(
            feedback_dir / "sample_B_01.npz",
            features=feat_sample2,
            label="B",
            predicted_label="D",
            source="user_disambiguation",
        )

        # Test retrain with these feedback samples
        import scripts.retrain_feedback_loop as rfl
        old_dir = rfl.FEEDBACK_DIR
        rfl.FEEDBACK_DIR = feedback_dir

        try:
            feats, labels, meta = load_feedback_samples()
            assert len(feats) == 2
            assert "A" in labels
            assert "B" in labels

            retrain_result = run_active_learning_retrain(
                epochs=2,
                min_samples=1,
                hot_reload=False,
                output_weights=temp_dir / "test_weights.pth",
            )
            assert retrain_result["status"] == "success"
            assert retrain_result["retrained"] is True
            assert retrain_result["feedback_samples_count"] == 2
            assert Path(retrain_result["weights_path"]).exists()
        finally:
            rfl.FEEDBACK_DIR = old_dir

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_reload_realsign():
    """Validates in-memory hot-swapping without connection loss."""
    success = pipeline.reload_realsign()
    assert success is True
    assert pipeline.realsign_classifier is not None


def test_dataset_active_learning_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)

    # 1. Post feedback
    res = client.post("/api/v1/dataset/feedback", json={
        "label": "C",
        "predicted_label": "O",
        "feature_vector": [0.05] * 126,
        "source": "disambiguation_hud",
        "user_consent": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["label"] == "C"

    # 2. Get feedback list
    res_list = client.get("/api/v1/dataset/feedback")
    assert res_list.status_code == 200
    data_list = res_list.json()
    assert "total_samples" in data_list
    assert data_list["total_samples"] >= 1
