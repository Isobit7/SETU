"""
Unit and integration tests for SETU ISL MVP Model and Classifier.
Verifies:
1. Weight loading and shape invariants
2. Deterministic inference on real feature vectors
3. Sequence pooling output dimensions
4. Hand presence gating and unknown / no-sign fallback behavior
5. Confidence thresholding
"""

import json
from pathlib import Path

import numpy as np
import pytest

from setu.features.schema import (
    LEFT_PRESENCE_INDEX,
    RIGHT_PRESENCE_INDEX,
    INTER_THUMB_DISTANCE_INDEX,
    CONTACT_SCORE_INDEX,
    INTER_HAND_TOTAL_SLICE,
)
from setu.models.isl_mvp_classifier import (
    ISLMVPClassifier,
    ISLPrediction,
    featurize_sequence_1264,
)


@pytest.fixture
def weights_path():
    p = Path("models/isl_mvp_model.npz")
    assert p.exists(), f"Model weights missing at {p}. Run training/train_mvp_model.py first."
    return p


@pytest.fixture
def classifier(weights_path):
    return ISLMVPClassifier(weights_path)


def test_model_loading_and_attributes(classifier):
    """Test model loads proper weight shapes and class names."""
    assert classifier.num_classes == 6
    assert classifier.weights.shape == (1264, 6)
    assert classifier.bias.shape == (6,)
    assert classifier.mean.shape == (1264,)
    assert classifier.scale.shape == (1264,)
    assert set(classifier.class_names) == {
        "hello", "thank_you", "good_morning", "i", "you", "no_sign"
    }


def test_featurize_sequence_1264_shape():
    """Verify pooling returns exactly (1264,) float32 vector."""
    dummy_seq = np.random.randn(30, 158).astype(np.float32)
    feat = featurize_sequence_1264(dummy_seq)
    assert feat.shape == (1264,)
    assert feat.dtype == np.float32
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_zero_hand_presence_routes_to_no_sign(classifier):
    """If hands are absent from the entire window, classifier must return no_sign immediately."""
    zero_seq = np.zeros((30, 158), dtype=np.float32)
    pred = classifier.predict_features(zero_seq)

    assert isinstance(pred, ISLPrediction)
    assert pred.label == "no_sign"
    assert pred.is_known_sign is False
    assert pred.confidence == 1.0


def test_hand_presence_gate_both_hands_absent(classifier):
    """Regression: both hands absent (flags 146=0, 147=0) triggers immediate no_sign."""
    seq = np.zeros((30, 158), dtype=np.float32)
    assert seq[:, RIGHT_PRESENCE_INDEX].sum() == 0.0
    assert seq[:, LEFT_PRESENCE_INDEX].sum() == 0.0

    pred = classifier.predict_features(seq)
    assert isinstance(pred, ISLPrediction)
    assert pred.label == "no_sign"
    assert pred.confidence == 1.0
    assert pred.is_known_sign is False
    assert pred.probabilities["no_sign"] == 1.0


def test_hand_presence_gate_right_hand_only(classifier):
    """Regression: right hand present (146=1.0), left absent (147=0.0) must not trigger zero-hand gate."""
    seq = np.zeros((30, 158), dtype=np.float32)
    seq[:, RIGHT_PRESENCE_INDEX] = 1.0
    seq[:, LEFT_PRESENCE_INDEX] = 0.0
    # Provide synthetic coordinates for right hand
    seq[:, :73] = np.random.uniform(-0.2, 0.2, size=(30, 73)).astype(np.float32)
    # Inter-hand features are strictly zero
    assert np.all(seq[:, INTER_HAND_TOTAL_SLICE] == 0.0)

    pred = classifier.predict_features(seq)
    # Must proceed to model inference, not short-circuited by hand-presence gate
    assert not (pred.label == "no_sign" and pred.confidence == 1.0 and pred.probabilities["no_sign"] == 1.0)


def test_hand_presence_gate_left_hand_only(classifier):
    """Regression: left hand present (147=1.0), right absent (146=0.0) must not trigger zero-hand gate."""
    seq = np.zeros((30, 158), dtype=np.float32)
    seq[:, RIGHT_PRESENCE_INDEX] = 0.0
    seq[:, LEFT_PRESENCE_INDEX] = 1.0
    # Provide synthetic coordinates for left hand
    seq[:, 73:146] = np.random.uniform(-0.2, 0.2, size=(30, 73)).astype(np.float32)
    # Inter-hand features are strictly zero
    assert np.all(seq[:, INTER_HAND_TOTAL_SLICE] == 0.0)

    pred = classifier.predict_features(seq)
    # Must proceed to model inference, not short-circuited by hand-presence gate
    assert not (pred.label == "no_sign" and pred.confidence == 1.0 and pred.probabilities["no_sign"] == 1.0)


def test_hand_presence_gate_both_hands_present(classifier):
    """Regression: both hands present (146=1.0, 147=1.0) passes gate properly."""
    seq = np.zeros((30, 158), dtype=np.float32)
    seq[:, RIGHT_PRESENCE_INDEX] = 1.0
    seq[:, LEFT_PRESENCE_INDEX] = 1.0
    seq[:, :146] = np.random.uniform(-0.2, 0.2, size=(30, 146)).astype(np.float32)

    pred = classifier.predict_features(seq)
    assert not (pred.label == "no_sign" and pred.confidence == 1.0 and pred.probabilities["no_sign"] == 1.0)


def test_hand_presence_gate_all_zero_inter_hand_features(classifier):
    """Regression: valid hand presence with all-zero inter-hand features must not trigger no_sign."""
    seq = np.zeros((30, 158), dtype=np.float32)
    seq[:, RIGHT_PRESENCE_INDEX] = 1.0
    seq[:, LEFT_PRESENCE_INDEX] = 0.0
    seq[:, :73] = np.random.uniform(-0.2, 0.2, size=(30, 73)).astype(np.float32)

    # Explicitly ensure inter-hand features (including thumb distance 156 and contact 157) are zero
    seq[:, INTER_HAND_TOTAL_SLICE] = 0.0
    assert seq[0, INTER_THUMB_DISTANCE_INDEX] == 0.0
    assert seq[0, CONTACT_SCORE_INDEX] == 0.0

    pred = classifier.predict_features(seq)
    assert not (pred.label == "no_sign" and pred.confidence == 1.0 and pred.probabilities["no_sign"] == 1.0)


def test_no_sign_decision_independent_of_inter_hand_indices(classifier):
    """
    Regression: verifies the no-sign gate decision depends ONLY on indices 146 and 147,
    and is strictly independent of thumb distance (156) or contact score (157).
    """
    # Case A: Both hands absent (146=0, 147=0), but old buggy indices (156, 157) are nonzero
    seq_absent = np.zeros((30, 158), dtype=np.float32)
    seq_absent[:, RIGHT_PRESENCE_INDEX] = 0.0
    seq_absent[:, LEFT_PRESENCE_INDEX] = 0.0
    seq_absent[:, INTER_THUMB_DISTANCE_INDEX] = 0.75  # Index 156
    seq_absent[:, CONTACT_SCORE_INDEX] = 0.95         # Index 157

    pred_absent = classifier.predict_features(seq_absent)
    # Must STILL trigger the zero-hand gate because true presence flags are 0
    assert pred_absent.label == "no_sign"
    assert pred_absent.confidence == 1.0
    assert pred_absent.is_known_sign is False
    assert pred_absent.probabilities["no_sign"] == 1.0

    # Case B: Hand is present (146=1.0), but old buggy indices (156, 157) are strictly 0.0
    seq_present = np.zeros((30, 158), dtype=np.float32)
    seq_present[:, RIGHT_PRESENCE_INDEX] = 1.0
    seq_present[:, LEFT_PRESENCE_INDEX] = 0.0
    seq_present[:, :73] = np.random.uniform(-0.1, 0.1, size=(30, 73)).astype(np.float32)
    seq_present[:, INTER_THUMB_DISTANCE_INDEX] = 0.0
    seq_present[:, CONTACT_SCORE_INDEX] = 0.0

    pred_present = classifier.predict_features(seq_present)
    # Must NOT trigger the zero-hand gate
    assert not (pred_present.label == "no_sign" and pred_present.confidence == 1.0 and pred_present.probabilities["no_sign"] == 1.0)



def test_low_confidence_routes_to_unknown(classifier):
    """When confidence is below threshold on an uncertain sign, label should indicate unknown/no_sign."""
    # Create an artificial flat distribution test
    pred = classifier.predict_features(
        np.random.normal(0, 0.1, size=(30, 158)).astype(np.float32),
        confidence_threshold=0.9999,  # Impossibly high threshold
    )
    assert pred.is_known_sign is False
    assert pred.label in ("unknown", "no_sign")


def test_inference_on_real_samples(classifier):
    """Verify deterministic inference on actual INCLUDE processed test samples."""
    manifest_path = Path("data/manifests/setu_isl_mvp_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check a verified 'thank_you' sample
    ty_sample = next(s for s in manifest["samples"] if s["sample_id"] == "include_thank_you_s06_9948")
    ty_data = np.load(ty_sample["features_path"])["features"]
    ty_pred = classifier.predict_features(ty_data)

    assert ty_pred.label == "thank_you"
    assert ty_pred.is_known_sign is True
    assert ty_pred.confidence > 0.80

    # Check a verified 'i' sample
    i_sample = next(s for s in manifest["samples"] if s["sample_id"] == "include_i_s12_0014")
    i_data = np.load(i_sample["features_path"])["features"]
    i_pred = classifier.predict_features(i_data)

    assert i_pred.label == "i"
    assert i_pred.is_known_sign is True
    assert i_pred.confidence > 0.80

    # Check a resting no_sign sample
    ns_sample = next(s for s in manifest["samples"] if s["sample_id"] == "no_sign_signer_03_empty_010")
    ns_data = np.load(ns_sample["features_path"])["features"]
    ns_pred = classifier.predict_features(ns_data)

    assert ns_pred.label == "no_sign"
    assert ns_pred.is_known_sign is False
