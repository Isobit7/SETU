"""
SETU - Dataset Preparation and Training Subsystem
Provides manifest generation, feature extraction pipelines, signer-level splits,
background NO_SIGN sequence generation, evaluation, and ONNX graph export.
"""

from training.isl_dataset_manifest_schema import SampleRecord, DatasetManifest
from training.extract_holistic_158d_features import FeatureExtractionPipeline
from training.generate_negative_resting_samples import NoSignGenerator
from training.evaluate_realsign_mlp import evaluate_model, evaluate_predictions
from training.export_realsign_mlp_to_onnx import export_to_onnx, verify_numerical_parity
from training.extract_realsign_63d_features import (
    normalize_landmarks_to_63d,
    extract_realsign_hand_features,
)

__all__ = [
    "SampleRecord",
    "DatasetManifest",
    "FeatureExtractionPipeline",
    "NoSignGenerator",
    "evaluate_model",
    "evaluate_predictions",
    "export_to_onnx",
    "verify_numerical_parity",
    "normalize_landmarks_to_63d",
    "extract_realsign_hand_features",
]
