"""
SETU — Centralized Configuration Package.
Provides typed, validated configuration parameters and project constants.
"""

from config.settings import (
    config,
    SETUConfig,
    FeatureSettings,
    VisionSettings,
    SequenceSettings,
    ServerSettings,
    PROJECT_ROOT,
    CONFIDENCE_THRESHOLD_ALPHABET,
    CONFIDENCE_THRESHOLD_SEQUENCE,
    STREAK_ALPHABET,
    STREAK_SEQUENCE,
    COOLDOWN_MS,
)

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
