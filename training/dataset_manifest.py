"""
SETU - Dataset Manifest Management
Manages sample records, signer-stratified splits, SHA-256 verification,
and dataset balance statistics to prevent data and signer leakage.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import hashlib

@dataclass
class SampleRecord:
    """Represents a single video or sequence sample in the SETU corpus."""
    sample_id: str
    label: str
    class_id: int
    signer_id: str
    raw_path: str
    features_path: str
    num_frames: int
    split: str = "unassigned"  # 'train', 'val', 'test', 'unassigned'
    sha256_checksum: str = ""
    source_dataset: str = "INCLUDE"
    license: str = "CC BY 4.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

class DatasetManifest:
    """
    Manages collection of sample records, export/import, and signer-isolated splits.
    """
    def __init__(self, name: str = "SETU-ISL-MVP", version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self.samples: List[SampleRecord] = []

    def add_sample(self, record: SampleRecord) -> None:
        """Adds a sample record to the manifest."""
        self.samples.append(record)

    def split_by_signer(
        self,
        train_signers: List[str],
        val_signers: List[str],
        test_signers: List[str]
    ) -> None:
        """
        Assigns each sample to 'train', 'val', or 'test' based strictly on signer ID.
        Prevents video and signer identity leakage across evaluation partitions.
        """
        train_set = set(train_signers)
        val_set = set(val_signers)
        test_set = set(test_signers)

        # Check for overlapping signer sets
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Signer sets must be mutually exclusive to prevent identity leakage!")

        for sample in self.samples:
            if sample.signer_id in train_set:
                sample.split = "train"
            elif sample.signer_id in val_set:
                sample.split = "val"
            elif sample.signer_id in test_set:
                sample.split = "test"
            else:
                sample.split = "unassigned"

    def get_summary(self) -> Dict[str, Any]:
        """Calculates dataset statistics: sample counts, class balance, and split distribution."""
        total = len(self.samples)
        by_class: Dict[str, int] = {}
        by_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
        by_signer: Dict[str, int] = {}
        missing_hands_counts: Dict[str, int] = {}

        for s in self.samples:
            by_class[s.label] = by_class.get(s.label, 0) + 1
            by_split[s.split] = by_split.get(s.split, 0) + 1
            by_signer[s.signer_id] = by_signer.get(s.signer_id, 0) + 1
            
            # Check hand presence metadata if available
            ratio = s.metadata.get("hand_presence_ratio", 1.0)
            if ratio < 0.5:
                missing_hands_counts[s.label] = missing_hands_counts.get(s.label, 0) + 1

        return {
            "total_samples": total,
            "classes_count": len(by_class),
            "samples_per_class": by_class,
            "samples_per_split": by_split,
            "samples_per_signer": by_signer,
            "low_hand_presence_samples": missing_hands_counts
        }

    def save_json(self, output_path: Path) -> None:
        """Serializes manifest to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "version": self.version,
            "summary": self.get_summary(),
            "samples": [asdict(s) for s in self.samples]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, json_path: Path) -> "DatasetManifest":
        """Loads manifest from JSON."""
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(name=data.get("name", "SETU"), version=data.get("version", "1.0.0"))
        for s_dict in data.get("samples", []):
            manifest.add_sample(SampleRecord(**s_dict))
        return manifest

def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
