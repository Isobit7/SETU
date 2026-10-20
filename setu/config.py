"""
SETU - Configuration Accessor
Re-exports settings from the root configs package for convenient in-package imports.
"""
from configs.settings import (
    config,
    SETUConfig,
    FeatureSettings,
    VisionSettings,
    SequenceSettings
)

# Real-Time Detection Thresholds & Temporal Debouncing Defaults
CONFIDENCE_THRESHOLD_ALPHABET = 0.30
CONFIDENCE_THRESHOLD_SEQUENCE = 0.35
STREAK_ALPHABET = 2
STREAK_SEQUENCE = 2
COOLDOWN_MS = 350

__all__ = [
    "config",
    "SETUConfig",
    "FeatureSettings",
    "VisionSettings",
    "SequenceSettings",
    "CONFIDENCE_THRESHOLD_ALPHABET",
    "CONFIDENCE_THRESHOLD_SEQUENCE",
    "STREAK_ALPHABET",
    "STREAK_SEQUENCE",
    "COOLDOWN_MS",
]
