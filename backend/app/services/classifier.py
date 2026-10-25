"""
SETU — Classifier Service Facade (Backward Compatibility).
The canonical implementation has been renamed to `realsign_classifier_service.py`.
"""

from backend.app.services.realsign_classifier_service import (
    CLASSES_26,
    RealSignMLP,
    RealSignInferenceService,
    classifier_service,
    normalize_landmarks_to_63d,
)

__all__ = [
    "CLASSES_26",
    "RealSignMLP",
    "RealSignInferenceService",
    "classifier_service",
    "normalize_landmarks_to_63d",
]
