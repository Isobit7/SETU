"""
SETU — Centralized Configuration & Environment Validation
Unified settings governing vision, feature extraction, sequence buffering, and server parameters.
Supports environment variable overrides and immutable frozen configuration objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Resolve root directory of the SETU project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_env_bool(key: str, default: bool) -> bool:
    """Helper to parse boolean environment variables safely."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def get_env_int(key: str, default: int) -> int:
    """Helper to parse integer environment variables safely."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Helper to parse float environment variables safely."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class FeatureSettings:
    """Settings governing the 158-feature representation."""
    schema_version: str = "1.0.0"
    feature_dim: int = 158
    single_hand_coords_dim: int = 63
    single_hand_geom_dim: int = 10
    presence_dim: int = 2
    inter_hand_dim: int = 10
    contact_sigma: float = 0.05
    min_scale_epsilon: float = 1e-6
    realsign_dim: int = 126


@dataclass(frozen=True)
class VisionSettings:
    """Settings governing camera capture and MediaPipe landmark detection."""
    camera_index: int = field(default_factory=lambda: get_env_int("SETU_CAMERA_INDEX", 0))
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    mirror_user_preview: bool = True
    task_model_path: Path = PROJECT_ROOT / "models" / "tasks" / "holistic_landmarker.task"
    hand_task_model_path: Path = PROJECT_ROOT / "models" / "tasks" / "hand_landmarker.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    camera_backend: str = field(default_factory=lambda: os.getenv("SETU_CAMERA_BACKEND", "DSHOW"))


@dataclass(frozen=True)
class SequenceSettings:
    """Settings governing temporal buffering and debounce thresholds for dynamic signs."""
    sequence_length: int = 30
    frame_stride: int = 2
    min_confidence_threshold: float = 0.70
    debounce_stable_frames: int = 5
    repeat_hold_frames: int = 25
    confidence_threshold_alphabet: float = field(
        default_factory=lambda: get_env_float("CONFIDENCE_THRESHOLD_ALPHABET", 0.30)
    )
    confidence_threshold_sequence: float = field(
        default_factory=lambda: get_env_float("CONFIDENCE_THRESHOLD_SEQUENCE", 0.35)
    )
    streak_alphabet: int = field(default_factory=lambda: get_env_int("STREAK_ALPHABET", 2))
    streak_sequence: int = field(default_factory=lambda: get_env_int("STREAK_SEQUENCE", 2))
    cooldown_ms: int = field(default_factory=lambda: get_env_int("COOLDOWN_MS", 350))


@dataclass(frozen=True)
class ServerSettings:
    """Settings governing the FastAPI server and WebSocket connections."""
    host: str = field(default_factory=lambda: os.getenv("SETU_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: get_env_int("SETU_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("SETU_LOG_LEVEL", "INFO"))
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class SETUConfig:
    """Aggregated system configuration with centralized immutable defaults."""
    features: FeatureSettings = field(default_factory=FeatureSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    sequence: SequenceSettings = field(default_factory=SequenceSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    project_root: Path = PROJECT_ROOT


# Default singleton instance
config = SETUConfig()

# Top-level hyperparameter exports for direct access
CONFIDENCE_THRESHOLD_ALPHABET: float = config.sequence.confidence_threshold_alphabet
CONFIDENCE_THRESHOLD_SEQUENCE: float = config.sequence.confidence_threshold_sequence
STREAK_ALPHABET: int = config.sequence.streak_alphabet
STREAK_SEQUENCE: int = config.sequence.streak_sequence
COOLDOWN_MS: int = config.sequence.cooldown_ms

__all__ = [
    "config",
    "SETUConfig",
    "FeatureSettings",
    "VisionSettings",
    "SequenceSettings",
    "ServerSettings",
    "PROJECT_ROOT",
    "CONFIDENCE_THRESHOLD_ALPHABET",
    "CONFIDENCE_THRESHOLD_SEQUENCE",
    "STREAK_ALPHABET",
    "STREAK_SEQUENCE",
    "COOLDOWN_MS",
]
