"""Tests for the PyTorch ISL Sequence LSTM classifier adapter."""

import numpy as np
import pytest
import torch
from pathlib import Path

from setu.models.isl_sequence_lstm_classifier import (
    ISLSequenceLSTMClassifier,
    ISLSequencePrediction,
    DEFAULT_SEQUENCE_LSTM_MODEL_PATH,
)


@pytest.fixture(scope="module")
def classifier():
    if not DEFAULT_SEQUENCE_LSTM_MODEL_PATH.exists():
        pytest.skip(f"Model checkpoint not found at {DEFAULT_SEQUENCE_LSTM_MODEL_PATH}")
    return ISLSequenceLSTMClassifier(DEFAULT_SEQUENCE_LSTM_MODEL_PATH)


def test_classifier_initialization(classifier):
    assert classifier.num_classes == len(classifier.class_names)
    assert classifier.input_size == 328
    assert "hello" in classifier.class_names
    assert "thank_you" in classifier.class_names
    assert "namaste" in classifier.class_names
    assert "no_sign" in classifier.class_names
    assert classifier.device is not None


def test_classifier_predict_sequence_shape_30x158(classifier):
    # Create realistic test sequence of shape (30, 158) - tests auto-velocity expansion to 328-D
    seq = np.random.randn(30, 158).astype(np.float32) * 0.1
    seq[:, 146] = 1.0  # right hand present
    seq[:, 147] = 1.0  # left hand present

    pred = classifier.predict_sequence(seq, confidence_threshold=0.40)
    assert isinstance(pred, ISLSequencePrediction)
    assert pred.label in classifier.class_names or pred.label == "unknown"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_3) == 3
    assert len(pred.probabilities) == classifier.num_classes
    assert np.isclose(sum(pred.probabilities.values()), 1.0, atol=1e-3)


def test_classifier_predict_sequence_shape_30x164(classifier):
    # Create test sequence of shape (30, 164) - 158 holistic + 6 facial ratios
    seq = np.random.randn(30, 164).astype(np.float32) * 0.1
    seq[:, 146] = 1.0
    seq[:, 147] = 1.0

    pred = classifier.predict_sequence(seq, confidence_threshold=0.40)
    assert isinstance(pred, ISLSequencePrediction)
    assert pred.label in classifier.class_names or pred.label == "unknown"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_3) == 3
    assert len(pred.probabilities) == classifier.num_classes
    assert np.isclose(sum(pred.probabilities.values()), 1.0, atol=1e-3)


def test_classifier_predict_sequence_shape_30x316(classifier):
    # Create 316-D test sequence (158 landmark coordinates + 158 velocities)
    seq = np.random.randn(30, 316).astype(np.float32) * 0.1
    seq[:, 146] = 1.0
    seq[:, 147] = 1.0

    pred = classifier.predict_sequence(seq, confidence_threshold=0.40)
    assert isinstance(pred, ISLSequencePrediction)
    assert pred.label in classifier.class_names or pred.label == "unknown"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_3) == 3
    assert len(pred.probabilities) == classifier.num_classes
    assert np.isclose(sum(pred.probabilities.values()), 1.0, atol=1e-3)


def test_classifier_predict_sequence_shape_30x328(classifier):
    # Create direct 328-D test sequence (164 coords/ratios + 164 velocities)
    seq = np.random.randn(30, 328).astype(np.float32) * 0.1
    seq[:, 146] = 1.0
    seq[:, 147] = 1.0

    pred = classifier.predict_sequence(seq, confidence_threshold=0.40)
    assert isinstance(pred, ISLSequencePrediction)
    assert pred.label in classifier.class_names or pred.label == "unknown"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_3) == 3
    assert len(pred.probabilities) == classifier.num_classes
    assert np.isclose(sum(pred.probabilities.values()), 1.0, atol=1e-3)


def test_classifier_short_sequence(classifier):
    # Sequence shorter than 10 frames should return no_sign
    short_seq = np.random.randn(5, 158).astype(np.float32)
    pred = classifier.predict_sequence(short_seq)
    assert pred.label == "no_sign"
    assert pred.is_known_sign is False


def test_classifier_invalid_shape(classifier):
    invalid = np.random.randn(30, 63).astype(np.float32)
    with pytest.raises(ValueError):
        classifier.predict_sequence(invalid)
