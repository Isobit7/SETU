"""
SETU - Reusable Feature Extraction Pipeline
Processes raw ISL video files through MediaPipe Tasks HolisticLandmarker and
FeatureExtractor158 to produce fixed-length (30, 158) feature arrays and metadata records.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from setu.features.schema import FEATURE_DIM, validate_schema_vector
from setu.features.extractor import FeatureExtractor158
from setu.vision.landmarks import HolisticLandmarkerWrapper
from training.dataset_manifest import SampleRecord, compute_file_sha256

class FeatureExtractionPipeline:
    """
    Standardized video feature extraction pipeline enforcing the 158-D contract
    and fixed temporal sequence length (30 frames) with uniform resampling.
    """
    def __init__(self, target_sequence_length: int = 30) -> None:
        self.target_len = target_sequence_length

    def extract_from_video(
        self,
        video_path: Path,
        landmarker: HolisticLandmarkerWrapper
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts 158-D features frame-by-frame from a video and resamples to target_len.
        Returns:
            (sequence_array_30x158, extraction_metadata)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        raw_frames_features = []
        frames_with_hands = 0
        total_frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                total_frames += 1

                # Detect keypoints on raw un-flipped frame
                mp_res = landmarker.detect(frame)
                vec = FeatureExtractor158.extract_from_mediapipe_result(mp_res)
                
                # Check hand presence (flags at 146 and 147)
                if vec[146] == 1.0 or vec[147] == 1.0:
                    frames_with_hands += 1

                raw_frames_features.append(vec)
        finally:
            cap.release()

        if total_frames == 0:
            raise ValueError(f"Video {video_path} contained 0 readable frames!")

        raw_arr = np.array(raw_frames_features, dtype=np.float32)

        # Resample temporally to target_len (e.g. 30 frames)
        if total_frames == self.target_len:
            resampled = raw_arr
        elif total_frames > self.target_len:
            # Uniform temporal downsampling
            indices = np.linspace(0, total_frames - 1, self.target_len, dtype=int)
            resampled = raw_arr[indices]
        else:
            # Temporal padding: repeat final frame to fill sequence
            pad_count = self.target_len - total_frames
            padding = np.repeat(raw_arr[-1:], pad_count, axis=0)
            resampled = np.vstack([raw_arr, padding])

        # Validate sequence invariants
        validate_schema_vector(resampled)

        hand_presence_ratio = frames_with_hands / total_frames if total_frames > 0 else 0.0
        metadata = {
            "source_frames": total_frames,
            "resampled_frames": self.target_len,
            "fps": float(fps),
            "width": width,
            "height": height,
            "frames_with_hands": frames_with_hands,
            "hand_presence_ratio": float(hand_presence_ratio)
        }

        return resampled, metadata

    def process_sample(
        self,
        raw_video_path: Path,
        output_features_dir: Path,
        sample_id: str,
        label: str,
        class_id: int,
        signer_id: str,
        landmarker: HolisticLandmarkerWrapper,
        source_dataset: str = "INCLUDE",
        license_str: str = "CC BY 4.0"
    ) -> SampleRecord:
        """
        Extracts features from a video, saves the .npz array, and returns a verified SampleRecord.
        """
        raw_video_path = Path(raw_video_path)
        output_features_dir = Path(output_features_dir)
        output_features_dir.mkdir(parents=True, exist_ok=True)

        features_path = output_features_dir / f"{sample_id}.npz"
        
        # 1. Feature extraction
        features, meta = self.extract_from_video(raw_video_path, landmarker)

        # 2. Save compressed features
        np.savez_compressed(features_path, features=features)

        # 3. Compute checksums
        raw_checksum = compute_file_sha256(raw_video_path)
        meta["features_sha256"] = compute_file_sha256(features_path)

        return SampleRecord(
            sample_id=sample_id,
            label=label,
            class_id=class_id,
            signer_id=signer_id,
            raw_path=str(raw_video_path),
            features_path=str(features_path),
            num_frames=self.target_len,
            sha256_checksum=raw_checksum,
            source_dataset=source_dataset,
            license=license_str,
            metadata=meta
        )
