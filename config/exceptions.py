"""
SETU — Centralized Exception Hierarchy
Provides explicit, domain-specific exceptions for feature extraction, vision,
camera capture, and model artifact handling. No silent error swallowing.
"""

from __future__ import annotations


class SETUError(Exception):
    """Base exception for all SETU framework errors."""
    pass


class FeatureValidationError(SETUError, ValueError):
    """Raised when a feature vector violates dimension, boundary, or value constraints."""
    pass


class LandmarkExtractionError(SETUError, ValueError):
    """Raised when landmark data extraction fails due to invalid structures or missing landmarks."""
    pass


class CameraError(SETUError):
    """Raised when hardware camera capture fails to initialize, open, or read frames."""
    pass


class ModelArtifactError(SETUError):
    """Raised when required model files, task bundles, or weights are missing or corrupt."""
    pass


class ConfigurationError(SETUError):
    """Raised when invalid or conflicting configuration options are provided."""
    pass


class InferenceError(SETUError):
    """Raised when model inference fails due to invalid inputs, shapes, or runtime faults."""
    pass


class AudioSynthesisError(SETUError):
    """Raised when text-to-speech audio synthesis fails."""
    pass


__all__ = [
    "SETUError",
    "FeatureValidationError",
    "LandmarkExtractionError",
    "CameraError",
    "ModelArtifactError",
    "ConfigurationError",
    "InferenceError",
    "AudioSynthesisError",
]
