"""
SETU — Unit Tests for Unified 287-Class Dynamic ISL Recognition
Verifies:
1. Deduplication: exactly 287 classes, zero duplicates in vocabulary.
2. All 29 overlapping signs mapped without collisions.
3. Unified dynamic adapter initializes, fuses predictions, and outputs valid results.
4. Resting hand (no_sign) rejection properly suppresses false positives.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from models.adapters.dynamic_unified import UnifiedDynamicAdapter, UnifiedDynamicResult


@pytest.fixture(scope="module")
def unified_adapter():
    adapter = UnifiedDynamicAdapter.get_instance()
    return adapter


def test_label_map_exact_deduplicated_count():
    map_path = Path("models/metadata/dynamic_unified_287_label_map.json")
    assert map_path.exists(), "Unified label map must exist"

    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("classes", [])
    assert len(classes) == 287, f"Expected 287 unique classes, got {len(classes)}"
    assert len(set(classes)) == 287, "Class names must be strictly unique with no duplicate keys"
    assert data.get("duplicate_signs_count") == 29, "Expected 29 deduplicated overlapping signs"


def test_adapter_duplicate_mapping(unified_adapter):
    assert unified_adapter.num_classes == 287
    assert len(unified_adapter.duplicate_unified_indices) == 29
    assert len(unified_adapter.amrita_to_unified_idx) == 53
    assert len(unified_adapter.include_to_unified_idx) == 263

    # Check key duplicate signs exist in unified classes
    for sign in ["hello", "thank_you", "good_morning", "i", "you", "father", "mother", "today", "tomorrow"]:
        assert sign in unified_adapter.label_to_index, f"Expected '{sign}' to be present in unified vocabulary"


def test_adapter_forward_inference_zero_error(unified_adapter):
    # Simulated active signing feature arrays
    feat316 = np.ones((20, 316), dtype=np.float32) * 0.5
    feat134 = np.ones((30, 134), dtype=np.float32) * 0.5

    result = unified_adapter.predict(feat316, feat134, confidence_threshold=0.35)
    assert isinstance(result, UnifiedDynamicResult)
    assert result.raw_class_idx >= 0
    assert len(result.top5_candidates) == 5
    assert len(result.top_3) == 3

    # Ensure no duplicate labels in top-5 candidates
    top5_labels = [c.label for c in result.top5_candidates]
    assert len(set(top5_labels)) == len(top5_labels), "Top-5 candidates must contain distinct labels"


def test_resting_hand_no_sign_suppression(unified_adapter):
    # Simulated resting array
    feat316 = np.zeros((20, 316), dtype=np.float32)
    feat134 = np.zeros((30, 134), dtype=np.float32)

    result = unified_adapter.predict(feat316, feat134, confidence_threshold=0.35)
    # Zero activity should result in resting posture / no_sign rejection
    assert not result.is_known_sign
    assert result.label == "no_sign"
    assert result.display_name == "Waiting for a sign..."

