"""
SETU — Machine Learning Model Registry & Swappable Inference Engines
Decouples ML model architectures, weights, and inference runtimes from HTTP/WebSocket routing.
Allows models to be swapped, benchmarked, or upgraded safely without server restarts or router rewrites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from models.adapters.realsign import RealSignClassifier, RealSignMLP, RealSignPrediction
from models.adapters.mvp import ISLMvpClassifier
from models.adapters.sequence_lstm import ISLSequenceLSTMClassifier
from models.adapters.include import IncludeClassifier
from models.adapters.rf_uv11 import RFUV11Adapter

logger = logging.getLogger("setu.models.registry")


class ModelType(str, Enum):
    STREAMING_ALPHABET = "streaming_alphabet"
    STREAMING_PHRASE = "streaming_phrases"
    STREAMING_SEQUENCE = "streaming_sequence"
    ISOLATED_VIDEO = "isolated_video"
    VISION_TRACKER = "vision_tracker"


@dataclass
class ModelMetadata:
    """Metadata describing a registered model engine."""
    model_id: str
    name: str
    model_type: ModelType
    input_dim: int
    num_classes: int
    weights_path: Path
    description: str
    device: str = "cpu"
    version: str = "1.0.0"
    metrics: Dict[str, float] = field(default_factory=dict)
    is_ready: bool = False


class ModelRegistry:
    """Central registry for discovering, registering, and instantiating SETU ML inference models."""

    _instances: Dict[str, Any] = {}
    _factories: Dict[str, Callable[[], Any]] = {}
    _metadata: Dict[str, ModelMetadata] = {}

    @classmethod
    def register(
        cls,
        model_id: str,
        name: str,
        model_type: ModelType,
        input_dim: int,
        num_classes: int,
        weights_path: Path,
        description: str,
        factory: Callable[[], Any],
        device: str = "cpu",
        version: str = "1.0.0",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Registers a model factory and metadata."""
        meta = ModelMetadata(
            model_id=model_id,
            name=name,
            model_type=model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            weights_path=weights_path,
            description=description,
            device=device,
            version=version,
            metrics=metrics or {},
            is_ready=False,
        )
        cls._metadata[model_id] = meta
        cls._factories[model_id] = factory
        logger.debug("Registered model engine: %s (%s)", model_id, name)

    @classmethod
    def get(cls, model_id: str) -> Optional[Any]:
        """Retrieves or instantiates the singleton instance for a given model ID."""
        if model_id in cls._instances:
            return cls._instances[model_id]

        factory = cls._factories.get(model_id)
        if not factory:
            logger.warning("No factory registered for model_id '%s'", model_id)
            return None

        try:
            instance = factory()
            cls._instances[model_id] = instance
            if model_id in cls._metadata:
                cls._metadata[model_id].is_ready = True
            logger.info("Instantiated model engine '%s' via registry", model_id)
            return instance
        except Exception as e:
            logger.error("Failed to instantiate model '%s': %s", model_id, e, exc_info=True)
            return None

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """Returns catalog of all registered models for API consumption."""
        result = []
        for model_id, meta in cls._metadata.items():
            result.append({
                "id": meta.model_id,
                "name": meta.name,
                "type": meta.model_type.value,
                "classes_count": meta.num_classes,
                "input_dim": meta.input_dim,
                "weights_path": str(meta.weights_path),
                "is_ready": meta.is_ready or (model_id in cls._instances),
                "device": meta.device,
                "version": meta.version,
                "metrics": meta.metrics,
                "description": meta.description,
            })
        return result

    @classmethod
    def clear(cls) -> None:
        """Clears cached instances (useful for testing or hot-reloading)."""
        cls._instances.clear()
        for meta in cls._metadata.values():
            meta.is_ready = False


def _resolve_weight_path(name: str) -> Path:
    p1 = Path("models/weights") / name
    if p1.exists():
        return p1
    return Path("models") / name


# Register production model engines
ModelRegistry.register(
    model_id="realsign_26class_alphabet",
    name="RealSign ISL 26-Class Alphabet PyTorch MLP",
    model_type=ModelType.STREAMING_ALPHABET,
    input_dim=63,
    num_classes=26,
    weights_path=_resolve_weight_path("realsign_26class_mlp.pth"),
    description="RealSign 26-class ISL alphabet fingerspelling (A–Z) classifier trained on 21 wrist-normalized 3D landmarks.",
    factory=lambda: RealSignClassifier(_resolve_weight_path("realsign_26class_mlp.pth")),
    version="1.1.0",
    metrics={"test_accuracy": 0.9911, "macro_f1": 0.9908},
)

ModelRegistry.register(
    model_id="isl_mvp_5sign",
    name="ISL Conversational 5-Sign Ridge Classifier",
    model_type=ModelType.STREAMING_PHRASE,
    input_dim=158,
    num_classes=5,
    weights_path=_resolve_weight_path("isl_mvp_model.npz"),
    description="Conversational ISL MVP classifier recognizing 5 essential signs (HELLO, THANK_YOU, YES, NO, PLEASE).",
    factory=lambda: ISLMvpClassifier(_resolve_weight_path("isl_mvp_model.npz")),
    version="1.0.0",
    metrics={"accuracy": 1.0, "latency_ms": 0.42},
)

ModelRegistry.register(
    model_id="isl_sequence_lstm",
    name="ISL 7-Class Dynamic Phrase BiLSTM",
    model_type=ModelType.STREAMING_SEQUENCE,
    input_dim=158,
    num_classes=7,
    weights_path=_resolve_weight_path("isl_sequence_lstm.pth"),
    description="Bidirectional LSTM model processing 30-frame temporal landmark sequences for dynamic ISL expressions.",
    factory=lambda: ISLSequenceLSTMClassifier(_resolve_weight_path("isl_sequence_lstm.pth")),
    version="1.0.0",
    metrics={"accuracy": 0.942, "sequence_length": 30},
)

ModelRegistry.register(
    model_id="ai4bharat_include_263sign",
    name="AI4Bharat INCLUDE 263-Sign Video LSTM",
    model_type=ModelType.ISOLATED_VIDEO,
    input_dim=158,
    num_classes=263,
    weights_path=_resolve_weight_path("include_no_cnn_lstm.pth"),
    description="Large-vocabulary Indian Sign Language video classifier covering 263 concepts across 15 semantic domains.",
    factory=lambda: IncludeClassifier(_resolve_weight_path("include_no_cnn_lstm.pth")),
    version="1.0.0",
    metrics={"vocabulary_size": 263, "source": "AI4Bharat Zenodo 4010759"},
)

ModelRegistry.register(
    model_id="roboflow_yolov11_gestures",
    name="Roboflow ISL YOLOv11 Gesture Detector",
    model_type=ModelType.VISION_TRACKER,
    input_dim=640,
    num_classes=11,
    weights_path=Path("runs/detect/train/weights/best.pt"),
    description="Direct RGB object detection model for 11 common ISL signs without requiring MediaPipe coordinate extraction.",
    factory=lambda: RFUV11Adapter(),
    version="1.0.0",
    metrics={"mAP50": 0.887},
)

__all__ = [
    "ModelType",
    "ModelMetadata",
    "ModelRegistry",
]
