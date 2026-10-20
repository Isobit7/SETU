"""Tests for the custom sign dataset collection utility and training aggregation pipeline."""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.record_custom_signs import (
    sanitize_label,
    draw_hud,
    draw_centered_text,
    record_sign_samples,
)
from training.train_sequence_lstm import (
    BASE_CLASSES,
    ISLSequenceDataset,
    load_sequence_dataset,
    load_manifest_data,
    temporal_resample_np,
    augment_sequence,
)


def test_sanitize_label():
    assert sanitize_label("Namaste") == "namaste"
    assert sanitize_label("THANK YOU") == "thank_you"
    assert sanitize_label("  good morning! ") == "good_morning"
    assert sanitize_label("help-me_123") == "help-me_123"
    assert sanitize_label("   ") == "custom_sign"


def test_draw_hud_and_centered_text():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud = draw_hud(
        frame=frame,
        label="test_label",
        take_idx=1,
        total_takes=5,
        status_text="RECORDING",
        progress=0.5,
        status_color=(0, 255, 0),
        hand_detected=True,
    )
    assert hud.shape == (480, 640, 3)
    assert np.any(hud > 0)  # text and overlays drawn

    draw_centered_text(hud, "Countdown 3", font_scale=1.0, color=(0, 255, 255))
    assert np.any(hud > 0)


def test_mock_record_sign_samples():
    """Verifies that record_sign_samples in mock headless mode captures exactly (30, 164) arrays."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        saved = record_sign_samples(
            label="namaste",
            num_samples=2,
            camera_idx=0,
            countdown_sec=1,
            output_dir=out_dir,
            headless=True,
            mock=True,
        )

        assert len(saved) == 2
        for fpath in saved:
            assert fpath.exists()
            assert fpath.name.startswith("custom_namaste_")
            assert fpath.name.endswith(".npz")

            with np.load(fpath) as data:
                assert "features" in data
                assert "label" in data
                features = data["features"]
                assert features.shape == (30, 164)
                assert features.dtype == np.float32
                assert str(data["label"]) == "namaste"


def test_load_sequence_dataset_with_custom_files():
    """Verifies load_sequence_dataset aggregates custom files and generates dynamic label map."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir(parents=True)
        label_map_file = tmp_path / "label_map.json"

        # Create 3 synthetic custom files for "namaste" and 1 for "water"
        for i in range(3):
            seq = np.random.randn(30, 164).astype(np.float32)
            np.savez(proc_dir / f"custom_namaste_{1000 + i}.npz", features=seq, label="namaste")
        np.savez(proc_dir / "custom_water_2000.npz", features=np.random.randn(30, 164).astype(np.float32), label="water")

        splits, class_names, label_to_idx = load_sequence_dataset(
            manifest_path=Path("non_existent_manifest.json"),
            processed_dir=proc_dir,
            label_map_path=label_map_file,
            custom_split_ratios=(0.6, 0.2, 0.2),
            update_label_map=True,
        )

        # Check classes: base classes preserved, custom classes added alphabetically, no_sign last
        assert "namaste" in class_names
        assert "water" in class_names
        assert class_names[-1] == "no_sign"
        assert len(class_names) == len(BASE_CLASSES) + 2

        # Check label_map.json file
        assert label_map_file.exists()
        with open(label_map_file, "r", encoding="utf-8") as f:
            lmap = json.load(f)
        assert lmap["num_classes"] == len(class_names)
        assert lmap["class_names"] == class_names
        assert "namaste" in lmap["custom_classes"]
        assert "water" in lmap["custom_classes"]

        # Check splits
        assert len(splits["train"]) >= 2
        total_custom = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total_custom == 4


def test_isl_sequence_dataset_balancing_and_velocity_328():
    """Verifies ISLSequenceDataset converts 164-D to 328-D and balances class counts."""
    # Class 0: 1 sample; Class 1: 3 samples
    samples = [
        (np.random.randn(30, 164).astype(np.float32), 0, "sample_c0_0"),
        (np.random.randn(30, 164).astype(np.float32), 1, "sample_c1_0"),
        (np.random.randn(30, 164).astype(np.float32), 1, "sample_c1_1"),
        (np.random.randn(30, 164).astype(np.float32), 1, "sample_c1_2"),
    ]

    # Test without augmentation
    eval_ds = ISLSequenceDataset(samples, augment=False)
    assert len(eval_ds) == 4
    x, y, sid = eval_ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (30, 328)  # Velocity feature vector generated
    assert isinstance(y, torch.Tensor)
    assert y.item() == 0

    # Test with augmentation and class balancing
    train_ds = ISLSequenceDataset(
        samples,
        augment=True,
        augment_factor=10,
        balance_classes=True,
        target_samples_per_class=30,
    )
    # Both classes should be augmented to at least 30 samples each -> total >= 60
    assert len(train_ds) >= 60

    # Count samples per class
    counts = {}
    for i in range(len(train_ds)):
        feat, lbl, _ = train_ds[i]
        assert feat.shape == (30, 328)
        c = lbl.item()
        counts[c] = counts.get(c, 0) + 1

    assert counts[0] >= 30
    assert counts[1] >= 30
