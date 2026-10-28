"""
SETU — Backend Pipeline Manager
Manages long-lived ML vision models, landmarkers, feature extractors, and warmup routines.
Decoupled from HTTP routing logic and easily swappable via models.ModelRegistry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from training.feature_extractor import FeatureExtractor158
from backend.app.services.vision import HolisticLandmarkerWrapper, HandLandmarkerWrapper
from models.adapters.realsign import RealSignClassifier
from models.adapters.mvp import ISLMvpClassifier
from models.adapters.sequence_lstm import ISLSequenceLSTMClassifier
from models.adapters.include import INCLUDEInferenceService, StreamingIncludeAdapter
from backend.app.services.vision.object_detector import COCOObjectDetector
from backend.app.services.vision.enhancer import AdaptiveLowLightEnhancer
from backend.app.services.nlp.multimodal_fuser import MultimodalFuser
from backend.app.services.nlp.predictive import PredictiveTextEngine
from backend.app.services.nlp.neural_translator import NeuralGlossTranslator, neural_translator

logger = logging.getLogger("setu.backend.pipeline")


def _find_weight_file(filename: str) -> Path:
    p1 = Path("models/weights") / filename
    if p1.exists():
        return p1
    return Path("models") / filename


class PipelineManager:
    """Manages long-lived ML vision models, landmark detectors, and inference engines."""

    def __init__(self) -> None:
        self.classifier: Optional[ISLMvpClassifier] = None
        self.realsign_classifier: Optional[RealSignClassifier] = None
        self.sequence_lstm_classifier: Optional[ISLSequenceLSTMClassifier] = None
        self.landmarker: Optional[HolisticLandmarkerWrapper] = None
        self.hand_landmarker: Optional[HandLandmarkerWrapper] = None
        self.extractor: Optional[FeatureExtractor158] = None
        self.include_service: Optional[INCLUDEInferenceService] = None
        self.streaming_include: Optional[StreamingIncludeAdapter] = None
        self.object_detector: Optional[COCOObjectDetector] = None
        self.enhancer: Optional[AdaptiveLowLightEnhancer] = None
        self.fuser: Optional[MultimodalFuser] = None
        self.predictive: Optional[PredictiveTextEngine] = None
        self.neural_translator: Optional[NeuralGlossTranslator] = None

        # Standard file paths
        self.model_path = _find_weight_file("isl_mvp_model.npz")
        self.realsign_model_path = _find_weight_file("realsign_26class_mlp.pth")
        self.sequence_lstm_model_path = _find_weight_file("isl_sequence_lstm.pth")
        self.task_path = Path("models/tasks/holistic_landmarker.task")
        self.hand_task_path = Path("models/tasks/hand_landmarker.task")
        self.is_ready = False

    def load(self) -> None:
        """Loads MediaPipe detectors, trained ISL classifiers, and warms up GPU/CPU."""
        if self.is_ready:
            return

        logger.info("Initializing SETU ML Pipeline...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path.resolve()}")
        if not self.task_path.exists():
            raise FileNotFoundError(f"Task bundle not found at {self.task_path.resolve()}")

        self.classifier = ISLMvpClassifier(self.model_path)
        self.landmarker = HolisticLandmarkerWrapper(task_model_path=self.task_path)
        self.landmarker.initialize()
        self.extractor = FeatureExtractor158()

        # Load RealSign 26-class PyTorch model
        if self.realsign_model_path.exists():
            try:
                self.realsign_classifier = RealSignClassifier(self.realsign_model_path)
                logger.info(
                    "RealSign 26-class PyTorch classifier loaded (%d classes: A-Z).",
                    self.realsign_classifier.num_classes,
                )
            except Exception as e:
                logger.warning("Could not pre-load RealSign classifier on startup: %s", e)
        else:
            logger.warning("RealSign weights not found at %s", self.realsign_model_path)

        # Load Dynamic ISL Sequence LSTM PyTorch model on GPU (CUDA)
        if self.sequence_lstm_model_path.exists():
            try:
                self.sequence_lstm_classifier = ISLSequenceLSTMClassifier(self.sequence_lstm_model_path)
                logger.info(
                    "Dynamic ISL Sequence LSTM classifier loaded on %s (%d classes).",
                    self.sequence_lstm_classifier.device,
                    self.sequence_lstm_classifier.num_classes,
                )
            except Exception as e:
                logger.warning("Could not pre-load ISL Sequence LSTM classifier on startup: %s", e)
        else:
            logger.warning("ISL Sequence LSTM weights not found at %s", self.sequence_lstm_model_path)

        # Load dedicated HandLandmarker for robust single-hand signing
        if self.hand_task_path.exists():
            try:
                self.hand_landmarker = HandLandmarkerWrapper(task_model_path=self.hand_task_path)
                self.hand_landmarker.initialize()
                logger.info("MediaPipe HandLandmarker loaded for alphabet sign tracking.")
            except Exception as e:
                logger.warning("Could not pre-load HandLandmarker on startup: %s", e)

        # Initialize INCLUDE pretrained model service
        try:
            self.include_service = INCLUDEInferenceService.get_instance()
            self.include_service.load()
            self.streaming_include = StreamingIncludeAdapter(service=self.include_service)
            logger.info("AI4Bharat INCLUDE service & streaming adapter loaded (%d classes).", len(self.include_service.label_map))
        except Exception as e:
            logger.warning("Could not pre-load INCLUDE model on startup: %s", e)

        # Initialize adaptive low-light enhancer and NLP fusers
        self.enhancer = AdaptiveLowLightEnhancer()
        self.fuser = MultimodalFuser()
        self.predictive = PredictiveTextEngine()
        self.neural_translator = neural_translator
        self.neural_translator.load(async_load=True)

        # Initialize COCO 80+ class object detector for Show-and-Sign interaction
        try:
            self.object_detector = COCOObjectDetector()
            self.object_detector.warmup()
            logger.info("COCO Object Detector initialized for Show-and-Sign interaction.")
        except Exception as e:
            logger.warning("Could not initialize COCO Object Detector: %s", e)

        # Startup Model Warmup
        self.warmup_models()

        self.is_ready = True
        logger.info("SETU ML Pipeline loaded successfully. Supported ISL phrases: %s", self.classifier.class_names)

    def warmup_models(self) -> None:
        """Runs dummy forward passes through neural network models with zero tensors."""
        if self.realsign_classifier is not None:
            try:
                in_dim = getattr(self.realsign_classifier, "input_dim", 126)
                dummy = np.ones(in_dim, dtype=np.float32) * 0.05
                self.realsign_classifier.predict(dummy, confidence_threshold=0.0, apply_smoothing=False)
                logger.info("RealSign 26-class PyTorch model warmed up successfully.")
            except Exception as e:
                logger.warning("RealSign warmup encountered error: %s", e)

        if self.sequence_lstm_classifier is not None:
            try:
                self.sequence_lstm_classifier.warmup()
                logger.info("ISL Sequence LSTM PyTorch model warmed up successfully.")
            except Exception as e:
                logger.warning("ISL Sequence LSTM warmup encountered error: %s", e)

    def close(self) -> None:
        """Releases C++ resources and ML models."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception as e:
                logger.warning("Error closing holistic landmarker: %s", e)
        if self.hand_landmarker is not None:
            try:
                self.hand_landmarker.close()
            except Exception as e:
                logger.warning("Error closing hand landmarker: %s", e)
        if self.streaming_include is not None:
            self.streaming_include.reset()
        self.landmarker = None
        self.hand_landmarker = None
        self.classifier = None
        self.realsign_classifier = None
        self.sequence_lstm_classifier = None
        self.extractor = None
        self.include_service = None
        self.streaming_include = None
        self.is_ready = False
        logger.info("SETU ML Pipeline shutdown completed.")


# Canonical singleton pipeline instance
pipeline = PipelineManager()
