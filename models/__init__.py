"""
SETU — Decoupled Machine Learning Model Registry & Architectures
Houses ML model architectures, weights paths, swappable inference runtimes, and registration.
"""

from __future__ import annotations

from pathlib import Path

from models.registry import ModelMetadata, ModelRegistry, ModelType
from models.adapters.realsign import (
    CLASSES_26,
    RealSignClassifier,
    RealSignMLP,
    RealSignPrediction,
)
from models.adapters.mvp import ISLMvpClassifier, MVP_CLASSES, MVPClassifierPrediction
# Aliases for compatibility
ISLMVPClassifier = ISLMvpClassifier
ISLPrediction = MVPClassifierPrediction

from models.adapters.sequence_lstm import (
    ISLSequenceLSTMClassifier,
    SequencePrediction,
    ISLSequenceLSTM,
)
ISLSequencePrediction = SequencePrediction

from models.adapters.include import (
    IncludeClassifier,
    IncludePrediction,
    CandidatePrediction,
    INCLUDEInferenceService,
    INCLUDELstmConfig,
    INCLUDELstmModel,
    INCLUDEResult,
    extract_include_134_features,
)
from models.adapters.rf_uv11 import RFUV11Adapter

__all__ = [
    "ModelRegistry",
    "ModelType",
    "ModelMetadata",
    "CLASSES_26",
    "RealSignClassifier",
    "RealSignMLP",
    "RealSignPrediction",
    "ISLMvpClassifier",
    "ISLMVPClassifier",
    "MVP_CLASSES",
    "MVPClassifierPrediction",
    "ISLPrediction",
    "ISLSequenceLSTM",
    "ISLSequenceLSTMClassifier",
    "SequencePrediction",
    "ISLSequencePrediction",
    "IncludeClassifier",
    "IncludePrediction",
    "CandidatePrediction",
    "INCLUDEInferenceService",
    "INCLUDELstmConfig",
    "INCLUDELstmModel",
    "INCLUDEResult",
    "extract_include_134_features",
    "RFUV11Adapter",
]
