"""
Tests for SETU Training & Pipeline Subsystem
Tests manifest creation, signer-stratified splitting, NO_SIGN generation,
and feature extraction from video with MediaPipe Tasks integration.
"""
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from training.dataset_manifest import DatasetManifest, SampleRecord, compute_file_sha256
from training.no_sign_generator import NoSignGenerator
from training.extract_features import FeatureExtractionPipeline
from backend.app.services.vision import HolisticLandmarkerWrapper

def test_no_sign_generator_shapes_and_invariants():
    """Verify NoSignGenerator creates valid (30, 158) sequences across scenarios."""
    gen = NoSignGenerator(sequence_length=30)
    
    empty_seq = gen.generate_empty_sequence()
    assert empty_seq.shape == (30, 158)
    assert empty_seq.dtype == np.float32
    assert np.all(empty_seq == 0.0)
    
    resting_seq = gen.generate_resting_hands_sequence(seed=123)
    assert resting_seq.shape == (30, 158)
    assert resting_seq.dtype == np.float32
    
    face_seq = gen.generate_face_touch_sequence(seed=456)
    assert face_seq.shape == (30, 158)
    assert face_seq.dtype == np.float32
    
    batch = gen.generate_batch(count_per_scenario=2, base_seed=10)
    assert len(batch) == 6 # 3 scenarios * 2
    for seq in batch:
        assert seq.shape == (30, 158)

def test_dataset_manifest_signer_splits_and_leakage_prevention():
    """Verify signer-level partitioning prevents identity and video leakage."""
    manifest = DatasetManifest(name="TestManifest")
    
    # 5 samples across 3 signers
    manifest.add_sample(SampleRecord(
        sample_id="s1_hello_01", label="hello", class_id=0, signer_id="signer_A",
        raw_path="dummy/path1", features_path="dummy/feat1", num_frames=30
    ))
    manifest.add_sample(SampleRecord(
        sample_id="s1_water_01", label="water", class_id=3, signer_id="signer_A",
        raw_path="dummy/path2", features_path="dummy/feat2", num_frames=30
    ))
    manifest.add_sample(SampleRecord(
        sample_id="s2_hello_02", label="hello", class_id=0, signer_id="signer_B",
        raw_path="dummy/path3", features_path="dummy/feat3", num_frames=30
    ))
    manifest.add_sample(SampleRecord(
        sample_id="s3_help_01", label="help", class_id=2, signer_id="signer_C",
        raw_path="dummy/path4", features_path="dummy/feat4", num_frames=30
    ))
    
    # Split by signer
    manifest.split_by_signer(
        train_signers=["signer_A"],
        val_signers=["signer_B"],
        test_signers=["signer_C"]
    )
    
    summary = manifest.get_summary()
    assert summary["total_samples"] == 4
    assert summary["samples_per_split"]["train"] == 2
    assert summary["samples_per_split"]["val"] == 1
    assert summary["samples_per_split"]["test"] == 1
    assert summary["samples_per_signer"]["signer_A"] == 2
    
    # Verify overlapping signer error
    with pytest.raises(ValueError, match="mutually exclusive"):
        manifest.split_by_signer(
            train_signers=["signer_A"],
            val_signers=["signer_A"],  # Overlapping!
            test_signers=["signer_C"]
        )

def test_manifest_json_roundtrip():
    """Verify manifest serializes to JSON and reloads with identical content."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "manifest.json"
        
        manifest = DatasetManifest(name="RoundtripTest", version="1.0.0")
        manifest.add_sample(SampleRecord(
            sample_id="test_01", label="hello", class_id=0, signer_id="S1",
            raw_path="raw/v1.mp4", features_path="feat/v1.npz", num_frames=30,
            split="train", sha256_checksum="abc123hash",
            metadata={"fps": 30.0, "hand_presence_ratio": 0.95}
        ))
        manifest.save_json(out_json)
        
        loaded = DatasetManifest.load_json(out_json)
        assert loaded.name == "RoundtripTest"
        assert len(loaded.samples) == 1
        s = loaded.samples[0]
        assert s.sample_id == "test_01"
        assert s.label == "hello"
        assert s.sha256_checksum == "abc123hash"
        assert s.metadata["hand_presence_ratio"] == 0.95

def test_feature_extraction_pipeline_video_processing():
    """
    Creates a temporary synthetic video, processes it through FeatureExtractionPipeline
    with local holistic_landmarker.task, and verifies (30, 158) output and metadata.
    """
    task_path = Path("models/tasks/holistic_landmarker.task")
    if not task_path.exists():
        pytest.skip("Local holistic_landmarker.task not found for pipeline test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid_path = tmp_path / "synthetic_test.mp4"
        feat_dir = tmp_path / "features"
        
        # Create a synthetic 15-frame video (640x480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, 15.0, (640, 480))
        for _ in range(15):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a white circle to simulate motion
            cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
            out.write(frame)
        out.release()
        
        # Initialize pipeline and landmarker
        pipeline = FeatureExtractionPipeline(target_sequence_length=30)
        with HolisticLandmarkerWrapper(task_model_path=task_path) as landmarker:
            record = pipeline.process_sample(
                raw_video_path=vid_path,
                output_features_dir=feat_dir,
                sample_id="syn_sample_01",
                label="test_gesture",
                class_id=99,
                signer_id="synthetic_signer",
                landmarker=landmarker
            )
            
        assert record.sample_id == "syn_sample_01"
        assert record.num_frames == 30
        assert Path(record.features_path).exists()
        
        # Load extracted .npz features and ensure file handle is closed on Windows
        with np.load(record.features_path) as data:
            assert "features" in data
            feat = data["features"]
            assert feat.shape == (30, 158)
            assert feat.dtype == np.float32
        
        # Verify metadata
        assert record.metadata["source_frames"] == 15
        assert record.metadata["resampled_frames"] == 30
        assert len(record.sha256_checksum) == 64 # Valid SHA-256 hex string
