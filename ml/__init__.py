"""SETU Machine Learning Module.

Includes feature extraction, dataset ingestion, model training, and evaluation.
"""

from ml.feature_extractor import (
    extract_realsign_hand_features,
    normalize_landmarks_to_63d,
)
from ml.evaluate import evaluate_model
from ml.train import RealSignDataset, train_model

__all__ = [
    "extract_realsign_hand_features",
    "normalize_landmarks_to_63d",
    "RealSignDataset",
    "train_model",
    "evaluate_model",
]
