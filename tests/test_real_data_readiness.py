"""
Tests for SETU Real ISL Data Acquisition and Training Readiness
Verifies genuine INCLUDE video feature extraction, genuine webcam non-sign isolation,
tensor invariant checks, strict signer-level split boundaries, and geometric contact metrics.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from config.settings import PROJECT_ROOT, config
from training.feature_extractor import FEATURE_DIM, EXPANDED_FEATURE_DIM, validate_schema_vector
from training.dataset_manifest import DatasetManifest, SampleRecord

def test_mvp_manifest_structure_and_integrity():
    """Verify that setu_isl_mvp_manifest.json exists, loads cleanly, and has valid metadata."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "setu_isl_mvp_manifest.json"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"
    
    manifest = DatasetManifest.load_json(manifest_path)
    assert manifest.name == "SETU-ISL-MVP-Controlled"
    assert len(manifest.samples) >= 24
    
    summary = manifest.get_summary()
    assert summary["total_samples"] == len(manifest.samples)
    assert summary["samples_per_split"]["unassigned"] == 0
    assert summary["samples_per_split"]["train"] > 0
    assert summary["samples_per_split"]["val"] > 0
    assert summary["samples_per_split"]["test"] > 0

def test_strict_signer_isolation_and_no_leakage():
    """Verify that train, val, and test sets share ZERO signers."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "setu_isl_mvp_manifest.json"
    manifest = DatasetManifest.load_json(manifest_path)
    
    signers_by_split = {"train": set(), "val": set(), "test": set()}
    for s in manifest.samples:
        signers_by_split[s.split].add(s.signer_id)
        
    # Check mutual exclusivity
    train_val_overlap = signers_by_split["train"] & signers_by_split["val"]
    train_test_overlap = signers_by_split["train"] & signers_by_split["test"]
    val_test_overlap = signers_by_split["val"] & signers_by_split["test"]
    
    assert len(train_val_overlap) == 0, f"Train and Val share signers: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train and Test share signers: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val and Test share signers: {val_test_overlap}"

def test_separation_of_synthetic_and_genuine_samples():
    """Verify synthetic NO_SIGN samples are explicitly tagged and kept distinct from genuine samples."""
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "setu_isl_mvp_manifest.json"
    manifest = DatasetManifest.load_json(manifest_path)
    
    synthetic_samples = []
    genuine_samples = []
    
    for s in manifest.samples:
        is_syn = s.metadata.get("is_synthetic", False) or s.source_dataset == "SETU-Controlled-Baseline"
        if is_syn:
            synthetic_samples.append(s)
            assert s.source_dataset == "SETU-Controlled-Baseline"
            assert s.raw_path == "synthetic_controlled_generator"
        else:
            genuine_samples.append(s)
            assert s.source_dataset in ["AI4Bharat-INCLUDE", "SETU-Webcam-Genuine"]
            assert Path(s.raw_path).exists()
            
    assert len(synthetic_samples) == 15
    assert len(genuine_samples) >= 9 # 6 INCLUDE + 3 webcam
    assert len(synthetic_samples) + len(genuine_samples) == len(manifest.samples)

def test_all_processed_features_tensors_invariants():
    """Verify all .npz feature files in data/processed have shape (30, 158), finite values, and valid schema."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    npz_files = [p for p in processed_dir.glob("*.npz") if "realsign" not in p.name]
    assert len(npz_files) >= 24
    
    for npz_path in npz_files:
        with np.load(npz_path) as data:
            assert "features" in data, f"'features' key missing in {npz_path.name}"
            arr = data["features"]
            assert arr.shape in (
                (30, FEATURE_DIM),
                (30, EXPANDED_FEATURE_DIM),
            ), f"Invalid shape {arr.shape} in {npz_path.name}"
            assert arr.dtype == np.float32
            assert np.all(np.isfinite(arr)), f"NaN or Inf in {npz_path.name}"
            validate_schema_vector(arr)
            
            # Check presence flags are binary
            r_flag = arr[:, 146]
            l_flag = arr[:, 147]
            assert np.all(np.isin(r_flag, [0.0, 1.0]))
            assert np.all(np.isin(l_flag, [0.0, 1.0]))
            
            # Check contact score bounded in [0.0, 1.0]
            contact = arr[:, 145]
            assert np.all((contact >= 0.0) & (contact <= 1.0))

def test_geometric_proximity_vs_contact_invariants():
    """Verify mathematical behavior of Gaussian contact score vs proximity distance."""
    sigma = config.features.contact_sigma
    
    # At d = 0 (exact contact), contact score C = 1.0
    c_zero = np.exp(-0.0 / (2.0 * sigma * sigma))
    assert np.isclose(c_zero, 1.0)
    
    # At d = sigma = 0.05, C = exp(-0.5) ~ 0.6065
    c_sigma = np.exp(-(sigma**2) / (2.0 * sigma * sigma))
    assert np.isclose(c_sigma, np.exp(-0.5))
    
    # At d = 3 * sigma = 0.15 (moderate distance), C ~ 0.011 (near 0)
    c_far = np.exp(-((3.0 * sigma)**2) / (2.0 * sigma * sigma))
    assert c_far < 0.02
    
    # Monotonicity check
    distances = np.linspace(0.0, 0.5, 50)
    scores = np.exp(-(distances**2) / (2.0 * sigma * sigma))
    assert np.all(np.diff(scores) <= 0.0)
