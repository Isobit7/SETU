"""
SETU - Centralized Configuration
Immutable defaults and runtime configuration for current vision, feature extraction,
and sequence buffering components. Non-secret settings only.
"""

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class FeatureSettings:
    """Settings governing the 158-feature representation."""
    feature_dim: int = 158
    single_hand_coords_dim: int = 63
    single_hand_geom_dim: int = 10
    presence_dim: int = 2
    inter_hand_dim: int = 10
    contact_sigma: float = 0.05
    min_scale_epsilon: float = 1e-6

@dataclass(frozen=True)
class VisionSettings:
    """Settings governing camera capture and landmark detection."""
    camera_index: int = int(os.getenv("SETU_CAMERA_INDEX", "0"))
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_user_preview: bool = True
    task_model_path: Path = PROJECT_ROOT / "models" / "tasks" / "holistic_landmarker.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

@dataclass(frozen=True)
class SequenceSettings:
    """Settings governing temporal buffering for dynamic signs."""
    sequence_length: int = 30
    frame_stride: int = 2
    min_confidence_threshold: float = 0.70
    debounce_stable_frames: int = 5
    repeat_hold_frames: int = 25

@dataclass(frozen=True)
class SETUConfig:
    """Aggregated system configuration."""
    features: FeatureSettings = FeatureSettings()
    vision: VisionSettings = VisionSettings()
    sequence: SequenceSettings = SequenceSettings()

# Default singleton instance
config = SETUConfig()
