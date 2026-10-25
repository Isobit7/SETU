"""
SETU - MVP Dataset Manifest Builder
Builds initial controlled dataset manifest and populates baseline NO_SIGN
feature samples adhering strictly to the 158-D schema.
"""

from pathlib import Path
import numpy as np

from training.isl_dataset_manifest_schema import DatasetManifest, SampleRecord, compute_file_sha256
from training.generate_negative_resting_samples import NoSignGenerator
from config.settings import PROJECT_ROOT

def build_mvp_manifest() -> DatasetManifest:
    manifest = DatasetManifest(name="SETU-ISL-MVP-Controlled", version="1.0.0")
    
    # 1. Authentic ISL Vocabulary Classes
    classes = [
        (0, "hello", "1-Hand", "INCLUDE: Greetings/48. Hello/"),
        (1, "thank_you", "1-Hand", "INCLUDE: Greetings/86. Thank you/"),
        (2, "help", "2-Hand", "INCLUDE: Society/38. Help/"),
        (3, "water", "1-Hand", "INCLUDE: Home/72. Water/"),
        (4, "father", "1-Hand", "INCLUDE: People/14. Father/"),
        (5, "mother", "1-Hand", "INCLUDE: People/47. Mother/"),
        (6, "yes", "1-Hand", "INCLUDE: Adjectives/118. Yes/"),
        (7, "no", "1-Hand/2-Hand", "INCLUDE: Adjectives/119. No/"),
        (8, "home", "2-Hand", "INCLUDE: Home/40. Home/"),
        (9, "work", "2-Hand", "INCLUDE: Jobs/18. Work/"),
        (10, "no_sign", "Background", "SETU Controlled Generator")
    ]
    
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate initial verified NO_SIGN feature bundles (15 samples across 3 signers)
    generator = NoSignGenerator(sequence_length=30)
    signers = ["signer_01", "signer_02", "signer_03"]
    
    sample_idx = 0
    for s_idx, signer in enumerate(signers):
        # 1 empty sequence, 2 resting sequences, 2 face touch sequences per signer
        scenarios = [
            ("empty", generator.generate_empty_sequence()),
            ("resting_1", generator.generate_resting_hands_sequence(seed=100 + s_idx * 10)),
            ("resting_2", generator.generate_resting_hands_sequence(seed=101 + s_idx * 10)),
            ("face_touch_1", generator.generate_face_touch_sequence(seed=200 + s_idx * 10)),
            ("face_touch_2", generator.generate_face_touch_sequence(seed=201 + s_idx * 10)),
        ]
        
        for name, seq in scenarios:
            sample_id = f"no_sign_{signer}_{name}_{sample_idx:03d}"
            feat_path = processed_dir / f"{sample_id}.npz"
            np.savez_compressed(feat_path, features=seq)
            
            checksum = compute_file_sha256(feat_path)
            
            record = SampleRecord(
                sample_id=sample_id,
                label="no_sign",
                class_id=10,
                signer_id=signer,
                raw_path="synthetic_controlled_generator",
                features_path=str(feat_path),
                num_frames=30,
                sha256_checksum=checksum,
                source_dataset="SETU-Controlled-Baseline",
                license="CC BY 4.0",
                metadata={"scenario": name, "hand_presence_ratio": 0.0 if name == "empty" else 1.0}
            )
            manifest.add_sample(record)
            sample_idx += 1

    # 3. Partition NO_SIGN samples strictly by signer
    manifest.split_by_signer(
        train_signers=["signer_01"], # 5 samples (~33%)
        val_signers=["signer_02"],   # 5 samples (~33%)
        test_signers=["signer_03"]   # 5 samples (~33%)
    )

    # 4. Save manifest to data/manifests/setu_isl_mvp_manifest.json
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "setu_isl_mvp_manifest.json"
    manifest.save_json(manifest_path)
    print(f"Manifest saved to: {manifest_path}")
    print("Summary:", manifest.get_summary())
    return manifest

if __name__ == "__main__":
    build_mvp_manifest()
