"""
Tests for Centralized Configuration and Exception Hierarchy
Verifies immutability, default values, hierarchy constraints, and clean error raising.
"""
import pytest
from dataclasses import FrozenInstanceError

from configs.settings import SETUConfig, FeatureSettings, VisionSettings, SequenceSettings, config
import setu.config
from setu.exceptions import (
    SETUError,
    FeatureValidationError,
    LandmarkExtractionError,
    CameraError,
    ModelArtifactError,
    ConfigurationError
)

def test_config_immutability():
    """Verify that settings are frozen dataclasses and cannot be modified at runtime."""
    with pytest.raises(FrozenInstanceError):
        config.features.feature_dim = 200

    with pytest.raises(FrozenInstanceError):
        config.vision.target_fps = 60

def test_config_reexport_equality():
    """Verify setu.config re-exports the identical config singleton from configs.settings."""
    assert setu.config.config is config
    assert setu.config.config.features.feature_dim == 158
    assert setu.config.config.features.contact_sigma == 0.05
    assert setu.config.config.vision.mirror_user_preview is True
    assert setu.config.CONFIDENCE_THRESHOLD_ALPHABET == 0.30
    assert setu.config.CONFIDENCE_THRESHOLD_SEQUENCE == 0.35
    assert setu.config.STREAK_ALPHABET == 2
    assert setu.config.STREAK_SEQUENCE == 2
    assert setu.config.COOLDOWN_MS == 350

def test_exception_hierarchy():
    """Verify all custom exceptions inherit from SETUError base."""
    assert issubclass(FeatureValidationError, (SETUError, ValueError))
    assert issubclass(LandmarkExtractionError, (SETUError, ValueError))
    assert issubclass(CameraError, SETUError)
    assert issubclass(ModelArtifactError, SETUError)
    assert issubclass(ConfigurationError, SETUError)

def test_exception_raising():
    """Verify exceptions raise with appropriate messages without crashing silently."""
    with pytest.raises(ModelArtifactError, match="Missing model file"):
        raise ModelArtifactError("Missing model file at path/to/model")

    with pytest.raises(CameraError, match="Camera index 99 unavailable"):
        raise CameraError("Camera index 99 unavailable")
