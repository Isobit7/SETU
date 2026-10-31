"""
SETU — Backend Pipeline Manager & ML Lifecycle Orchestrator
===========================================================

Architectural Role:
-------------------
The `PipelineManager` is the central singleton orchestrator for all machine learning
inference models, computer vision landmark detectors, feature extraction pipelines,
and natural language translation modules in SETU.

Key Responsibilities:
---------------------
1. Lifecycle Management: Coordinates model loading, GPU/CPU initialization, memory warmup,
   and clean resource deallocation on server shutdown.
2. Swappable Registry Integration: Provides decoupled access to the unified `models.registry.ModelRegistry`.
3. Multi-Tier Model Dispatch:
   * RealSign Alphabet MLP (126-D Dual-Hand, 26 Classes): `models/weights/realsign_26class_mlp.pth`
   * Conversational Phrase MVP (1264-D Pooled, 6 Classes): `models/weights/isl_mvp_model.npz`
   * Continuous Sequence BiLSTM (328-D CSLR, 7 Classes): `models/weights/isl_sequence_lstm.pth`
   * AI4Bharat INCLUDE Pretrained LSTM (134-D Video, 263 Classes): `models/pretrained/include_no_cnn_lstm.pth`
   * Vision Landmarkers: Holistic & Dedicated Hand Landmarker task bundles in `models/tasks/`
   * Multi-Modal Context Fuser & Neural Gloss-to-English Translator
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


class PipelineManager:
    """
    Singleton Manager for long-lived ML models, vision landmark detectors, and NLP engines.
    
    Ensures models are instantiated once at application startup, warmed up on hardware devices
    (CUDA GPU or CPU), and safely shared across concurrent WebSocket connections without reloading.
    """

    def __init__(self) -> None:
        # Classifier engines
        self.classifier: Optional[ISLMvpClassifier] = None
        self.realsign_classifier: Optional[RealSignClassifier] = None
        self.sequence_lstm_classifier: Optional[ISLSequenceLSTMClassifier] = None
        self.include_service: Optional[INCLUDEInferenceService] = None
        self.streaming_include: Optional[StreamingIncludeAdapter] = None

        # Vision detectors & feature extraction
        self.landmarker: Optional[HolisticLandmarkerWrapper] = None
        self.hand_landmarker: Optional[HandLandmarkerWrapper] = None
        self.extractor: Optional[FeatureExtractor158] = None
        self.object_detector: Optional[COCOObjectDetector] = None
        self.enhancer: Optional[AdaptiveLowLightEnhancer] = None

        # NLP linguistic post-processing
        self.fuser: Optional[MultimodalFuser] = None
        self.predictive: Optional[PredictiveTextEngine] = None
        self.neural_translator: Optional[NeuralGlossTranslator] = None

        # Canonical explicit file paths
        self.model_path = Path("models/weights/isl_mvp_model.npz")
        self.realsign_model_path = Path("models/weights/realsign_26class_mlp.pth")
        self.sequence_lstm_model_path = Path("models/weights/isl_sequence_lstm.pth")
        self.task_path = Path("models/tasks/holistic_landmarker.task")
        self.hand_task_path = Path("models/tasks/hand_landmarker.task")
        self.is_ready = False

    def load(self) -> None:
        """
        Loads all MediaPipe detectors, neural classifiers, and NLP translators into memory.
        Executes pre-flight hardware warmup forward passes.
        """
        if self.is_ready:
            return

        logger.info("Initializing SETU ML Pipeline...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path.resolve()}")
        if not self.task_path.exists():
            raise FileNotFoundError(f"Task bundle not found at {self.task_path.resolve()}")

        # 1. Load Conversational ISL MVP 6-Class Model
        self.classifier = ISLMvpClassifier(self.model_path)
        
        # 2. Initialize MediaPipe Holistic Landmarker
        self.landmarker = HolisticLandmarkerWrapper(task_model_path=self.task_path)
        self.landmarker.initialize()
        self.extractor = FeatureExtractor158()

        # 3. Load RealSign 26-Class Bimanual PyTorch MLP
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

        # 4. Load Dynamic ISL Sequence LSTM PyTorch Model on GPU/CPU
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

        # 5. Load Dedicated HandLandmarker for High-Precision Fingerspelling
        if self.hand_task_path.exists():
            try:
                self.hand_landmarker = HandLandmarkerWrapper(task_model_path=self.hand_task_path)
                self.hand_landmarker.initialize()
                logger.info("MediaPipe HandLandmarker loaded for alphabet sign tracking.")
            except Exception as e:
                logger.warning("Could not pre-load HandLandmarker on startup: %s", e)

        # 6. Initialize AI4Bharat INCLUDE Pretrained Service
        try:
            self.include_service = INCLUDEInferenceService.get_instance()
            self.include_service.load()
            self.streaming_include = StreamingIncludeAdapter(service=self.include_service)
            logger.info("AI4Bharat INCLUDE service & streaming adapter loaded (%d classes).", len(self.include_service.label_map))
        except Exception as e:
            logger.warning("Could not pre-load INCLUDE model on startup: %s", e)

        # 7. Initialize Adaptive Vision Enhancer and NLP Engines
        self.enhancer = AdaptiveLowLightEnhancer()
        self.fuser = MultimodalFuser()
        self.predictive = PredictiveTextEngine()
        self.neural_translator = neural_translator
        self.neural_translator.load(async_load=True)

        # 8. Initialize COCO Object Detector for Show-and-Sign Interaction
        try:
            self.object_detector = COCOObjectDetector()
            self.object_detector.warmup()
            logger.info("COCO Object Detector initialized for Show-and-Sign interaction.")
        except Exception as e:
            logger.warning("Could not initialize COCO Object Detector: %s", e)

        # 9. Pre-flight Hardware Memory Warmup
        self.warmup_models()

        self.is_ready = True
        logger.info("SETU ML Pipeline loaded successfully. Supported ISL phrases: %s", self.classifier.class_names)

    def warmup_models(self) -> None:
        """
        Executes dummy forward passes through neural models using zero/synthetic tensors.
        Pre-allocates CUDA/CPU memory caches and initial kernel graph compilations to eliminate
        first-frame inference stuttering.
        """
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
        """
        Safely releases underlying C++ native memory allocations, MediaPipe task contexts,
        and PyTorch CUDA GPU tensor allocations on application shutdown.
        """
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
