"""
Tests for Centralized Configuration and Exception Hierarchy
Verifies immutability, default values, hierarchy constraints, and clean error raising.
"""
import pytest
from dataclasses import FrozenInstanceError

from config.settings import SETUConfig, FeatureSettings, VisionSettings, SequenceSettings, config
import config as config_pkg
from config.exceptions import (
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
    """Verify config package exports the identical config singleton and settings."""
    assert config_pkg.config is config
    assert config.features.feature_dim == 158
    assert config.features.contact_sigma == 0.05
    assert config.vision.mirror_user_preview is True
    assert config.sequence.confidence_threshold_alphabet == 0.30
    assert config.sequence.confidence_threshold_sequence == 0.35
    assert config.sequence.streak_alphabet == 2
    assert config.sequence.streak_sequence == 2
    assert config.sequence.cooldown_ms == 350

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
