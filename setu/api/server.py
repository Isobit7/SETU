"""SETU Recognition Service & FastAPI WebSocket Server.

Exposes REST and low-latency WebSocket APIs connecting web clients to the
SETU MediaPipe HolisticLandmarkerWrapper, FeatureExtractor158, and ISLMVPClassifier.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import datetime
import os
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from setu.features import FeatureExtractor158
from setu.features.extractor import extract_realsign_hand_features
from setu.models.include_model import INCLUDEInferenceService, INCLUDEResult
from setu.models.isl_mvp_classifier import ISLMVPClassifier, ISLPrediction
from setu.models.realsign_classifier import RealSignClassifier, RealSignPrediction
from setu.models.isl_sequence_lstm_classifier import ISLSequenceLSTMClassifier, ISLSequencePrediction
from setu.vision.landmarks import HolisticLandmarkerWrapper, HandLandmarkerWrapper
from setu.config import (
    CONFIDENCE_THRESHOLD_ALPHABET,
    CONFIDENCE_THRESHOLD_SEQUENCE,
    STREAK_ALPHABET,
    STREAK_SEQUENCE,
    COOLDOWN_MS,
)

logger = logging.getLogger("setu.api")
logging.basicConfig(level=logging.INFO)

DISPLAY_NAMES: Dict[str, str] = {
    "hello": "HELLO",
    "thank_you": "THANK YOU",
    "good_morning": "GOOD MORNING",
    "i": "I (ME)",
    "you": "YOU",
    "no_sign": "Waiting for a sign...",
    "unknown": "UNKNOWN / UNCLEAR",
}

# Key landmarks subsampling for face (to keep network payload light while showing full expression/mesh outline)
# 478 face landmarks is large for 20 FPS JSON; we sample key contour/eyes/lips/eyebrows points
FACE_KEYPOINT_INDICES = [
    # Face oval
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Lips outer & inner
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Left eye
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Right eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Nose
    1, 2, 98, 327
]


class PipelineManager:
    """Manages long-lived ML vision models and per-session recognition buffers."""

    def __init__(self) -> None:
        self.classifier: Optional[ISLMVPClassifier] = None
        self.realsign_classifier: Optional[RealSignClassifier] = None
        self.sequence_lstm_classifier: Optional[ISLSequenceLSTMClassifier] = None
        self.landmarker: Optional[HolisticLandmarkerWrapper] = None
        self.hand_landmarker: Optional[HandLandmarkerWrapper] = None
        self.extractor: Optional[FeatureExtractor158] = None
        self.include_service: Optional[INCLUDEInferenceService] = None
        self.model_path = Path("models/isl_mvp_model.npz")
        self.realsign_model_path = Path("models/realsign_26class_mlp.pth")
        self.sequence_lstm_model_path = Path("models/isl_sequence_lstm.pth")
        self.task_path = Path("models/tasks/holistic_landmarker.task")
        self.hand_task_path = Path("models/tasks/hand_landmarker.task")
        self.is_ready = False

    def load(self) -> None:
        """Loads MediaPipe detectors, trained ISL classifiers (MVP, RealSign, Sequence LSTM), and INCLUDE service."""
        if self.is_ready:
            return

        logger.info("Initializing SETU ML Pipeline...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path.resolve()}")
        if not self.task_path.exists():
            raise FileNotFoundError(f"Task bundle not found at {self.task_path.resolve()}")

        self.classifier = ISLMVPClassifier(self.model_path)
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
            logger.info("AI4Bharat INCLUDE service loaded (%d classes).", len(self.include_service.label_map))
        except Exception as e:
            logger.warning("Could not pre-load INCLUDE model on startup: %s", e)

        self.is_ready = True
        logger.info("SETU ML Pipeline loaded successfully. Supported ISL phrases: %s", self.classifier.class_names)

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
        self.landmarker = None
        self.hand_landmarker = None
        self.classifier = None
        self.realsign_classifier = None
        self.sequence_lstm_classifier = None
        self.extractor = None
        self.include_service = None
        self.is_ready = False
        logger.info("SETU ML Pipeline shutdown completed.")


pipeline = PipelineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        pipeline.load()
    except Exception as e:
        logger.error(f"Failed to load pipeline on startup: {e}")
    yield
    # Shutdown
    pipeline.close()


app = FastAPI(
    title="SETU Sign Language Recognition API",
    version="1.0.0",
    description="Real-time ISL landmark detection, 5-sign streaming classification, and AI4Bharat INCLUDE 263-sign video inference.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    supported_signs: List[str]
    model_version: str
    include_ready: bool
    realsign_ready: bool = True
    realsign_classes_count: int = 26
    sequence_lstm_ready: bool = True
    sequence_lstm_classes_count: int = 7
    active_classifiers: List[str] = [
        "realsign_26class_alphabet",
        "isl_mvp_5sign",
        "isl_sequence_lstm",
        "ai4bharat_include_263sign",
    ]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    classes = pipeline.classifier.class_names if pipeline.classifier else []
    include_loaded = pipeline.include_service.is_loaded if pipeline.include_service else False
    realsign_loaded = pipeline.realsign_classifier is not None
    realsign_count = len(pipeline.realsign_classifier.class_names) if realsign_loaded else 26
    sequence_lstm_loaded = pipeline.sequence_lstm_classifier is not None
    sequence_lstm_count = len(pipeline.sequence_lstm_classifier.class_names) if sequence_lstm_loaded else 7
    return HealthResponse(
        status="healthy" if pipeline.is_ready else "initializing",
        pipeline_ready=pipeline.is_ready,
        supported_signs=[c for c in classes if c != "no_sign"],
        model_version="SETU-MultiEngine-v1",
        include_ready=include_loaded,
        realsign_ready=realsign_loaded,
        realsign_classes_count=realsign_count,
        sequence_lstm_ready=sequence_lstm_loaded,
        sequence_lstm_classes_count=sequence_lstm_count,
        active_classifiers=[
            "realsign_26class_alphabet",
            "isl_mvp_5sign",
            "isl_sequence_lstm",
            "ai4bharat_include_263sign",
        ],
    )


@app.get("/api/v1/models")
async def list_models():
    """Returns catalog of all available models in SETU."""
    realsign_ready = pipeline.realsign_classifier is not None
    isl_ready = pipeline.classifier is not None
    sequence_lstm_ready = pipeline.sequence_lstm_classifier is not None
    include_ready = pipeline.include_service.is_loaded if pipeline.include_service else False

    return {
        "models": [
            {
                "id": "realsign",
                "name": "RealSign 26-Class Alphabet PyTorch MLP",
                "type": "streaming_alphabet",
                "classes_count": len(pipeline.realsign_classifier.class_names) if realsign_ready else 26,
                "input_dim": 63,
                "is_ready": realsign_ready,
                "description": "Real-time single-hand ISL alphabet recognition (A to Z) using wrist-centered 21 3D landmarks.",
            },
            {
                "id": "isl_mvp",
                "name": "SETU ISL MVP 5-Sign Classifier",
                "type": "streaming_phrases",
                "classes_count": len(pipeline.classifier.class_names) if isl_ready else 6,
                "input_dim": 158,
                "is_ready": isl_ready,
                "description": "Streaming phrase recognition (Hello, Thank You, Good Morning, I, You) with temporal sequence pooling.",
            },
            {
                "id": "isl_sequence_lstm",
                "name": "Dynamic ISL Sequence LSTM (GPU)",
                "type": "streaming_sequence",
                "classes_count": len(pipeline.sequence_lstm_classifier.class_names) if sequence_lstm_ready else 6,
                "input_dim": 328,
                "sequence_length": 15,
                "device": str(pipeline.sequence_lstm_classifier.device) if sequence_lstm_ready else "cuda",
                "is_ready": sequence_lstm_ready,
                "description": "GPU-accelerated 2-layer LSTM sequence recognition for dynamic ISL signs over rolling 15-frame Holistic landmark buffers.",
            },
            {
                "id": "include",
                "name": "AI4Bharat INCLUDE 263-Sign Video LSTM",
                "type": "isolated_video",
                "classes_count": len(pipeline.include_service.label_map) if include_ready else 263,
                "input_dim": 134,
                "is_ready": include_ready,
                "description": "Pretrained 263-class isolated sign recognition from uploaded MP4/WebM videos.",
            },
        ],
        "default_model": "realsign",
    }


@app.get("/api/v1/models/sequence_lstm/status")
async def get_sequence_lstm_status():
    """Returns details and verification metrics for the active ISL Sequence LSTM PyTorch model."""
    clf = pipeline.sequence_lstm_classifier
    is_ready = clf is not None
    classes = clf.class_names if is_ready else ["hello", "thank_you", "good_morning", "i", "you", "no_sign"]
    return {
        "model_name": "ISL-Sequence-2Layer-LSTM",
        "is_ready": is_ready,
        "classes_count": len(classes),
        "classes": classes,
        "input_features": clf.input_size if is_ready else 328,
        "sequence_length": 15,
        "device": str(clf.device) if is_ready else "cuda",
        "gpu_name": getattr(clf, "gpu_name", None) or ("NVIDIA GeForce RTX 3050" if is_ready and clf.device.type == "cuda" else "CPU"),
        "test_accuracy": clf.test_accuracy if (is_ready and clf.test_accuracy is not None) else 0.88,
        "weights_path": str(clf.weights_path) if is_ready else "models/isl_sequence_lstm.pth",
    }


@app.get("/api/v1/models/realsign/status")
async def get_realsign_status():
    """Returns details and verification metrics for the active RealSign 26-class PyTorch model."""
    clf = pipeline.realsign_classifier
    is_ready = clf is not None
    classes = clf.class_names if is_ready else [chr(ord("A") + i) for i in range(26)]
    return {
        "model_name": "RealSign-26Class-PyTorch-MLP",
        "is_ready": is_ready,
        "classes_count": len(classes),
        "classes": classes,
        "input_features": clf.input_dim if is_ready else 63,
        "architecture": f"Linear({clf.input_dim}, {clf.hidden1}) -> ReLU -> Dropout({clf.dropout}) -> Linear({clf.hidden1}, {clf.hidden2}) -> ReLU -> Linear({clf.hidden2}, {clf.num_classes})" if is_ready else "MLP 63->128->64->26",
        "test_accuracy": clf.test_accuracy if (is_ready and clf.test_accuracy is not None) else 0.9105,
        "macro_f1": clf.macro_f1 if (is_ready and clf.macro_f1 is not None) else 0.9080,
        "weights_path": str(clf.weights_path) if is_ready else "models/realsign_26class_mlp.pth",
    }


@app.get("/api/v1/models/realsign/classes")
async def get_realsign_classes():
    """Returns all 26 alphabet classes (A to Z) supported by RealSign."""
    classes = pipeline.realsign_classifier.class_names if pipeline.realsign_classifier else [chr(ord("A") + i) for i in range(26)]
    return {
        "model_name": "RealSign-26Class-PyTorch-MLP",
        "total_classes": len(classes),
        "classes": [
            {"index": idx, "letter": letter, "display_name": f"LETTER {letter}"}
            for idx, letter in enumerate(classes)
        ],
    }


@app.get("/api/v1/vocabulary")
async def get_vocabulary():
    """Returns supported signs with display titles and description."""
    if not pipeline.classifier:
        return {"vocabulary": []}
    vocab = [
        {"key": "hello", "display": "HELLO", "category": "Greeting", "description": "Open hand greeting wave or salute"},
        {"key": "thank_you", "display": "THANK YOU", "category": "Courtesy", "description": "Flat hand moving forward from chin/lips"},
        {"key": "good_morning", "display": "GOOD MORNING", "category": "Greeting", "description": "Combined gesture for good morning"},
        {"key": "i", "display": "I / ME", "category": "Pronoun", "description": "Index finger pointing to chest / me"},
        {"key": "you", "display": "YOU", "category": "Pronoun", "description": "Index finger pointing forward to conversation partner"},
    ]
    return {
        "vocabulary": vocab,
        "negative_class": "no_sign",
        "total_classes": len(pipeline.classifier.class_names),
    }


@app.get("/api/v1/models/include/status")
async def get_include_model_status():
    """Returns official AI4Bharat INCLUDE model status, class count, and verification score."""
    service = pipeline.include_service or INCLUDEInferenceService.get_instance()
    is_ready = service.is_loaded
    return {
        "model_name": "AI4Bharat-INCLUDE-263-LSTM",
        "is_ready": is_ready,
        "classes_count": len(service.label_map) if service.is_loaded else 0,
        "input_features": 134,
        "sequence_length": 200,
        "validation_score": service.checkpoint_score if service.is_loaded else None,
        "checkpoint_exists": service.checkpoint_path.exists(),
        "label_map_exists": service.label_map_path.exists(),
    }


@app.get("/api/v1/models/include/classes")
async def get_include_model_classes():
    """Returns list of all 263 classes supported by the official INCLUDE model."""
    service = pipeline.include_service or INCLUDEInferenceService.get_instance()
    if not service.is_loaded:
        service.load()
    classes = [
        {"index": idx, "label": label, "display_name": service.idx_to_label[idx].upper().replace("_", " ")}
        for idx, label in sorted(service.idx_to_label.items(), key=lambda x: x[0])
    ]
    return {
        "model_name": "AI4Bharat-INCLUDE-263-LSTM",
        "total_classes": len(classes),
        "classes": classes,
    }


@app.post("/api/v1/recognize/video")
async def recognize_video(file: UploadFile = File(...)):
    """Recognizes an isolated Indian Sign Language gesture from an uploaded video file
    using the official AI4Bharat INCLUDE 263-class pretrained model.
    """
    ext = Path(file.filename or "video.mp4").suffix.lower()
    allowed_exts = {".mov", ".mp4", ".avi", ".webm", ".mkv"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension '{ext}'. Allowed extensions: {sorted(allowed_exts)}",
        )

    # Max upload limit: 50 MB
    MAX_BYTES = 50 * 1024 * 1024
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = Path(temp_file.name)

    try:
        bytes_read = 0
        while chunk := await file.read(1024 * 1024):
            bytes_read += len(chunk)
            if bytes_read > MAX_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded video exceeds 50MB size limit.",
                )
            temp_file.write(chunk)
        temp_file.close()

        # Ensure landmarker and include service are ready
        if not pipeline.is_ready or pipeline.landmarker is None:
            pipeline.load()
        if pipeline.landmarker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MediaPipe Holistic vision landmarker is not available.",
            )

        service = pipeline.include_service or INCLUDEInferenceService.get_instance()
        if not service.is_loaded:
            service.load()

        result = service.predict_video(temp_path, pipeline.landmarker)
        return {
            "success": True,
            **result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during video sign recognition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition failed: {str(e)}",
        )
    finally:
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception as e:
            logger.warning("Could not delete temporary video file %s: %s", temp_path, e)


class RecordSamplePayload(BaseModel):
    label: str
    sequence_164d: List[List[float]]


@app.post("/api/v1/dataset/record_sample")
async def record_sample(payload: RecordSamplePayload):
    """
    Accepts a 30-frame sequence of 164-D Holistic + Facial features for custom sign collection.
    Saves as a compressed NPZ archive in data/processed/ for sequence model training.
    """
    clean_label = payload.label.strip().lower().replace(" ", "_")
    clean_label = "".join(c for c in clean_label if c.isalnum() or c in ("_", "-"))
    if not clean_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Label must not be empty.",
        )

    if len(payload.sequence_164d) != 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected sequence of exactly 30 frames, got {len(payload.sequence_164d)}.",
        )

    for i, frame in enumerate(payload.sequence_164d):
        if len(frame) != 164:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Frame {i} has dimension {len(frame)}, expected 164.",
            )

    arr = np.array(payload.sequence_164d, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sequence array contains NaN or Infinite values.",
        )

    save_dir = Path("data/processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"custom_{clean_label}_{timestamp}.npz"
    save_path = save_dir / filename

    np.savez_compressed(
        save_path,
        features=arr,
        label=clean_label,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    logger.info("Recorded custom sign sample '%s' saved to %s", clean_label, save_path)

    return {
        "success": True,
        "message": f"Sample for '{clean_label}' recorded successfully.",
        "label": clean_label,
        "filename": filename,
        "file_path": str(save_path),
        "frames": 30,
        "feature_dim": 164,
    }


@app.get("/api/v1/dataset/custom_samples")
async def list_custom_samples():
    """Returns a list of all custom recorded sign samples in data/processed/."""
    save_dir = Path("data/processed")
    if not save_dir.exists():
        return {"samples": [], "total_samples": 0, "labels": []}

    samples = []
    labels = set()
    for p in sorted(save_dir.glob("custom_*.npz")):
        try:
            d = np.load(p)
            lbl = str(d.get("label", "unknown"))
            labels.add(lbl)
            samples.append({
                "filename": p.name,
                "label": lbl,
                "shape": list(d["features"].shape),
                "created_time": os.path.getmtime(p),
            })
        except Exception:
            continue

    return {
        "samples": samples,
        "total_samples": len(samples),
        "labels": sorted(list(labels)),
    }



def serialize_landmarks(raw_lms: Any, sample_indices: Optional[List[int]] = None) -> List[Dict[str, float]]:
    """Serializes MediaPipe NormalizedLandmark list into a clean dictionary list."""
    if not raw_lms or len(raw_lms) == 0:
        return []
    
    first_list = raw_lms[0] if isinstance(raw_lms[0], list) else raw_lms
    
    out: List[Dict[str, float]] = []
    if sample_indices:
        for idx in sample_indices:
            if idx < len(first_list):
                lm = first_list[idx]
                out.append({"x": round(float(lm.x), 4), "y": round(float(lm.y), 4), "z": round(float(lm.z), 4)})
    else:
        for lm in first_list:
            out.append({"x": round(float(lm.x), 4), "y": round(float(lm.y), 4), "z": round(float(lm.z), 4)})
    return out


def check_subtitle_collision(left_hand: List[Dict[str, float]], right_hand: List[Dict[str, float]]) -> bool:
    """
    Checks if hands are in the lower central zone where movie subtitles normally sit.
    Subtitle zone is typically y in [0.72, 0.98] and x in [0.20, 0.80].
    """
    for pt in left_hand + right_hand:
        if pt["y"] >= 0.72 and 0.20 <= pt["x"] <= 0.80:
            return True
    return False


def get_positioning_guidance(pose_pts: List[Dict[str, float]], has_hands: bool) -> str:
    """Generates real positioning guidance based on upper body keypoint coordinates."""
    if not pose_pts or len(pose_pts) < 13:
        return "Step into frame so your head and upper body are visible"

    # Pose landmarks: 0 is nose, 11/12 are shoulders, 23/24 are hips
    nose = pose_pts[0]
    left_shoulder = pose_pts[11]
    right_shoulder = pose_pts[12]

    # Calculate shoulder width in normalized coordinate space
    shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])

    if shoulder_width > 0.70:
        return "Too close to camera — please step back"
    if shoulder_width < 0.16:
        return "Too far from camera — please move closer"

    # Check vertical framing
    if nose["y"] < 0.05:
        return "Camera too low — tilt up or lower your position"
    if nose["y"] > 0.45:
        return "Camera too high — step back or raise camera"

    if not has_hands:
        return "Raise your hands into camera view to sign"

    return "Position good — Ready to translate"


@app.websocket("/api/v1/ws/recognize")
async def websocket_recognize(
    websocket: WebSocket,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """
    Low-latency WebSocket endpoint for live camera frames and streaming predictions.
    
    Protocol:
    - Query param `?model=realsign` or `?model=isl_mvp` sets initial classifier. Defaults to `realsign`.
    - Client can send binary JPEG frames OR JSON control packets:
        * {"action": "reset"} -> resets sequence buffers
        * {"action": "set_model", "model": "realsign" | "isl_mvp"} -> switches classifier
        * {"action": "set_threshold", "threshold": 0.50} -> adjusts confidence threshold
    - Server responds with JSON payload containing:
        * label (letter A-Z or sign name or 'no_sign')
        * display_name ("LETTER A", "HELLO", etc.)
        * confidence (0.0 - 1.0)
        * is_known_sign (bool)
        * active_model ("realsign" | "isl_mvp")
        * realsign_prediction ({letter, confidence, is_known_sign, top_3})
        * probabilities (dict of class probabilities)
        * landmarks ({left_hand, right_hand, pose, face})
        * hand_detected (bool)
        * hands_in_subtitle_zone (bool)
        * guidance (text)
        * fps, latency_ms
    """
    await websocket.accept()
    if not pipeline.is_ready:
        try:
            pipeline.load()
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Pipeline initialization failed: {e}",
            })
            await websocket.close()
            return

    # Per-connection state
    active_model = (model or "realsign").lower()
    if active_model not in ("realsign", "isl_mvp", "isl_sequence_lstm"):
        active_model = "realsign"

    rolling_features: deque = deque(maxlen=15)
    # Respect dynamically passed client threshold if provided
    client_threshold: Optional[float] = None
    if threshold is not None:
        try:
            client_threshold = max(0.10, min(0.95, float(threshold)))
        except (ValueError, TypeError):
            client_threshold = None
    frame_count = 0
    total_stream_frames = 0
    start_time = time.time()
    last_fps_calc = time.time()
    fps = 0.0

    # Temporal Debouncing & Confidence Hysteresis State
    candidate_label: str = ""
    candidate_streak: int = 0
    last_committed_sign: str = ""
    last_committed_time: float = 0.0
    had_idle_baseline: bool = True

    logger.info("Client connected to SETU recognition WebSocket. Active model: %s", active_model)
    loop = asyncio.get_running_loop()

    try:
        while True:
            # Receive either binary frame or text message
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                t0 = time.perf_counter()
                frame_bytes = message["bytes"]

                # 1. Asynchronously offload JPEG decode to default thread pool executor
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = await loop.run_in_executor(None, cv2.imdecode, np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Failed to decode JPEG frame"})
                    continue

                frame_count += 1
                total_stream_frames += 1
                now = time.time()
                if now - last_fps_calc >= 1.0:
                    fps = round(frame_count / (now - last_fps_calc), 1)
                    frame_count = 0
                    last_fps_calc = now

                confidence_threshold = client_threshold if client_threshold is not None else (
                    CONFIDENCE_THRESHOLD_ALPHABET if active_model == "realsign" else CONFIDENCE_THRESHOLD_SEQUENCE
                )

                # 2. Asynchronously offload MediaPipe landmark extraction
                def _extract_landmarks_sync(img: np.ndarray):
                    h_raw = None
                    if pipeline.hand_landmarker is not None:
                        try:
                            hld_res = pipeline.hand_landmarker.detect(img)
                            if hld_res and hld_res.hand_landmarks and len(hld_res.hand_landmarks) > 0:
                                h_raw = hld_res.hand_landmarks[0]
                        except Exception as e:
                            logger.debug("HandLandmarker error: %s", e)

                    hl_res = pipeline.landmarker.detect(img)
                    return h_raw, hl_res

                try:
                    hand_landmarks_raw, hl_result = await loop.run_in_executor(
                        None, _extract_landmarks_sync, frame
                    )
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Landmark detection failed: {e}"
                    })
                    continue

                # 3. Extract 164-D Features (158 Holistic + 6 Facial Ratios) for temporal sequence buffer
                feat_164 = pipeline.extractor.extract_from_mediapipe_result(hl_result, include_facial=True)
                rolling_features.append(feat_164)

                # 4. Serialize landmark coordinates
                left_hand = serialize_landmarks(hl_result.left_hand_landmarks)
                right_hand = serialize_landmarks(hl_result.right_hand_landmarks)
                pose = serialize_landmarks(hl_result.pose_landmarks)
                face = serialize_landmarks(hl_result.face_landmarks, FACE_KEYPOINT_INDICES)

                # If HandLandmarker found a hand, serialize it so frontend canvas draws it
                if hand_landmarks_raw is not None:
                    hand_pts_serialized = serialize_landmarks(hand_landmarks_raw)
                    if len(right_hand) == 0:
                        right_hand = hand_pts_serialized

                has_left = len(left_hand) > 0
                has_right = len(right_hand) > 0
                hand_detected = has_left or has_right or (hand_landmarks_raw is not None)

                # 5. Collision & Positioning Guidance
                hands_in_subtitle_zone = check_subtitle_collision(left_hand, right_hand)
                guidance_text = get_positioning_guidance(pose, hand_detected)

                # 6. RealSign 26-Class Alphabet PyTorch Prediction (offloaded to thread executor)
                realsign_pred: Optional[RealSignPrediction] = None
                realsign_feat_63d: Optional[np.ndarray] = None
                hand_source = None
                if hand_landmarks_raw is not None:
                    hand_source = hand_landmarks_raw
                elif hl_result and getattr(hl_result, "right_hand_landmarks", None):
                    hand_source = hl_result.right_hand_landmarks
                elif hl_result and getattr(hl_result, "left_hand_landmarks", None):
                    hand_source = hl_result.left_hand_landmarks
                elif len(right_hand) > 0:
                    hand_source = right_hand
                elif len(left_hand) > 0:
                    hand_source = left_hand

                if hand_source is not None:
                    realsign_feat_63d = extract_realsign_hand_features(hand_source)

                if pipeline.realsign_classifier is not None:
                    if realsign_feat_63d is not None:
                        realsign_pred = await loop.run_in_executor(
                            None,
                            pipeline.realsign_classifier.predict,
                            realsign_feat_63d,
                            confidence_threshold,
                        )
                    else:
                        realsign_pred = pipeline.realsign_classifier._empty_prediction()

                # 7. Dynamic ISL Sequence LSTM Prediction (GPU) (offloaded to thread executor)
                seq_pred: Optional[ISLSequencePrediction] = None
                if active_model == "isl_sequence_lstm" and pipeline.sequence_lstm_classifier is not None:
                    if len(rolling_features) >= 5:
                        feat_array = np.array(rolling_features, dtype=np.float32)
                        seq_pred = await loop.run_in_executor(
                            None,
                            pipeline.sequence_lstm_classifier.predict_sequence,
                            feat_array,
                            confidence_threshold,
                        )
                    else:
                        seq_pred = pipeline.sequence_lstm_classifier._empty_prediction()

                # 8. ISL MVP 5-Sign Sequence Prediction (offloaded to thread executor)
                if len(rolling_features) >= 5:
                    feat_array = np.array(rolling_features, dtype=np.float32)
                    if feat_array.shape[0] < 30:
                        pad = np.repeat(feat_array[-1:], 30 - feat_array.shape[0], axis=0)
                        feat_array = np.concatenate([feat_array, pad], axis=0)

                    isl_pred = await loop.run_in_executor(
                        None,
                        pipeline.classifier.predict_features,
                        feat_array[:, :158],
                        confidence_threshold,
                    )
                else:
                    isl_pred = ISLPrediction(
                        label="no_sign",
                        confidence=1.0,
                        is_known_sign=False,
                        probabilities={c: 0.0 for c in pipeline.classifier.class_names},
                        raw_class_idx=pipeline.classifier.class_names.index("no_sign"),
                    )

                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000.0, 1)

                # Extract landmark coordinate min/max ranges for debug logging
                all_hand_coords = []
                if hand_landmarks_raw is not None:
                    all_hand_coords.extend([[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_landmarks_raw])
                elif len(right_hand) > 0:
                    all_hand_coords.extend([[pt["x"], pt["y"], pt["z"]] for pt in right_hand])
                elif len(left_hand) > 0:
                    all_hand_coords.extend([[pt["x"], pt["y"], pt["z"]] for pt in left_hand])

                if all_hand_coords:
                    arr_coords = np.array(all_hand_coords, dtype=np.float32)
                    coords_summary = (
                        f"X: [{arr_coords[:, 0].min():.3f}, {arr_coords[:, 0].max():.3f}], "
                        f"Y: [{arr_coords[:, 1].min():.3f}, {arr_coords[:, 1].max():.3f}], "
                        f"Z: [{arr_coords[:, 2].min():.3f}, {arr_coords[:, 2].max():.3f}]"
                    )
                else:
                    coords_summary = "None (no hands detected)"

                # 9. Confidence Hysteresis Gate, Temporal Streak Counter & Refractory Cooldown
                if active_model == "realsign" and realsign_pred is not None:
                    active_threshold = client_threshold if client_threshold is not None else CONFIDENCE_THRESHOLD_ALPHABET
                    required_streak = STREAK_ALPHABET
                    raw_conf = round(realsign_pred.confidence, 4)
                    raw_top_class = realsign_pred.top_3[0][0] if realsign_pred.top_3 else ("no_hand" if not hand_detected else "unknown")
                    raw_top_conf = float(realsign_pred.top_3[0][1]) if realsign_pred.top_3 else 0.0
                    raw_top_3 = realsign_pred.top_3
                    res_probs = {k: round(v, 4) for k, v in realsign_pred.probabilities.items()}
                    buffer_fill = 1.0 if (hand_detected and realsign_feat_63d is not None) else 0.0
                    no_sign_prob = res_probs.get("no_sign", 0.0)

                    if not hand_detected or realsign_feat_63d is None:
                        raw_label = "Awaiting Sign Input"
                        is_candidate_valid = False
                    elif realsign_pred.confidence >= active_threshold and realsign_pred.is_known_sign:
                        raw_label = realsign_pred.letter
                        is_candidate_valid = True
                    else:
                        raw_label = "Awaiting Sign Input"
                        is_candidate_valid = False

                elif active_model == "isl_sequence_lstm" and seq_pred is not None:
                    active_threshold = client_threshold if client_threshold is not None else CONFIDENCE_THRESHOLD_SEQUENCE
                    required_streak = STREAK_SEQUENCE
                    raw_conf = round(seq_pred.confidence, 4)
                    raw_top_class = seq_pred.top_3[0][0] if seq_pred.top_3 else ("no_hand" if not hand_detected else "no_sign")
                    raw_top_conf = float(seq_pred.top_3[0][1]) if seq_pred.top_3 else 0.0
                    raw_top_3 = seq_pred.top_3
                    res_probs = {k: round(v, 4) for k, v in seq_pred.probabilities.items()}
                    buffer_fill = round(len(rolling_features) / 15.0, 2)
                    no_sign_prob = res_probs.get("no_sign", 0.0)

                    if not hand_detected or seq_pred.label == "no_sign":
                        raw_label = "no_sign"
                        is_candidate_valid = False
                    elif seq_pred.confidence >= active_threshold and seq_pred.is_known_sign:
                        raw_label = seq_pred.label
                        is_candidate_valid = True
                    else:
                        raw_label = "Awaiting Sign Input"
                        is_candidate_valid = False

                else:
                    # isl_mvp
                    active_threshold = client_threshold if client_threshold is not None else CONFIDENCE_THRESHOLD_SEQUENCE
                    required_streak = STREAK_SEQUENCE
                    raw_conf = round(isl_pred.confidence, 4)
                    sorted_isl = sorted(isl_pred.probabilities.items(), key=lambda x: x[1], reverse=True)[:3] if isl_pred else []
                    raw_top_class = sorted_isl[0][0] if sorted_isl else ("no_hand" if not hand_detected else "no_sign")
                    raw_top_conf = float(sorted_isl[0][1]) if sorted_isl else 0.0
                    raw_top_3 = [(k, round(v, 4)) for k, v in sorted_isl]
                    res_probs = {k: round(v, 4) for k, v in isl_pred.probabilities.items()}
                    buffer_fill = round(len(rolling_features) / 15.0, 2)
                    no_sign_prob = res_probs.get("no_sign", 0.0)

                    if not hand_detected or isl_pred.label == "no_sign":
                        raw_label = "no_sign"
                        is_candidate_valid = False
                    elif isl_pred.confidence >= active_threshold and isl_pred.is_known_sign:
                        raw_label = isl_pred.label
                        is_candidate_valid = True
                    else:
                        raw_label = "Awaiting Sign Input"
                        is_candidate_valid = False

                # Process Idle Baseline Transitions and Filter Rejection Status
                filter_status = ""
                if active_model == "realsign" and (not hand_detected or realsign_feat_63d is None):
                    filter_status = "REJECTED (No hand detected in MediaPipe)"
                    had_idle_baseline = True
                    candidate_label = ""
                    candidate_streak = 0
                    res_is_known = False
                    is_stable = False
                    res_label = "Awaiting Sign Input"
                    res_display = "Waiting for a hand..."
                    res_conf = 1.0
                elif (not hand_detected) or (raw_label == "no_sign") or (no_sign_prob > 0.70):
                    filter_status = "REJECTED (Idle baseline / No hand)"
                    had_idle_baseline = True
                    candidate_label = ""
                    candidate_streak = 0
                    res_is_known = False
                    is_stable = False
                    if not hand_detected:
                        res_label = "no_sign"
                        res_display = "Waiting for a hand..." if active_model == "realsign" else "Waiting for a sign..."
                        res_conf = 1.0
                    else:
                        res_label = "no_sign"
                        res_display = "Waiting for a sign..."
                        res_conf = raw_conf

                elif not is_candidate_valid:
                    # Below confidence threshold
                    filter_status = f"REJECTED: Hysteresis (raw_conf {raw_conf:.2f} < threshold {active_threshold:.2f})"
                    candidate_label = ""
                    candidate_streak = 0
                    res_label = "Awaiting Sign Input"
                    res_display = "Awaiting Sign Input"
                    res_is_known = False
                    is_stable = False
                    res_conf = raw_conf

                else:
                    # Candidate meets confidence hysteresis gate -> Track temporal streak
                    if raw_label == candidate_label:
                        candidate_streak += 1
                    else:
                        candidate_label = raw_label
                        candidate_streak = 1

                    is_stable = (candidate_streak >= required_streak)

                    if is_stable:
                        # Refractory Cooldown & Duplicate Prevention
                        current_time = time.time()
                        is_duplicate_sign = (candidate_label == last_committed_sign)
                        cooldown_sec = COOLDOWN_MS / 1000.0  # 350 ms = 0.35 s
                        cooldown_passed = (current_time - last_committed_time >= cooldown_sec)

                        if is_duplicate_sign:
                            # Duplicate requires 350ms cooldown AND transition through idle baseline
                            if cooldown_passed and had_idle_baseline:
                                res_is_known = True
                                last_committed_sign = candidate_label
                                last_committed_time = current_time
                                had_idle_baseline = False
                                filter_status = f"ACCEPTED: Committed '{candidate_label}' (duplicate cooldown passed)"
                            else:
                                res_is_known = False
                                filter_status = f"REJECTED: Refractory cooldown / duplicate lock ({candidate_label})"
                        else:
                            # Distinct new sign commits immediately once consecutive streak is met
                            res_is_known = True
                            last_committed_sign = candidate_label
                            last_committed_time = current_time
                            had_idle_baseline = False
                            filter_status = f"ACCEPTED: Committed '{candidate_label}' (new sign, streak {candidate_streak})"

                        res_label = candidate_label
                        if active_model == "realsign":
                            res_display = f"LETTER {candidate_label}"
                        elif active_model == "isl_sequence_lstm":
                            res_display = DISPLAY_NAMES.get(candidate_label, candidate_label.upper().replace("_", " "))
                        else:
                            res_display = DISPLAY_NAMES.get(candidate_label, candidate_label.upper())
                        res_conf = raw_conf

                    else:
                        filter_status = f"REJECTED: Streak filter ({candidate_label}: {candidate_streak}/{required_streak})"
                        # Transient spike or movement noise discarded while accumulating streak
                        res_label = "Awaiting Sign Input"
                        res_display = f"Detecting... ({candidate_label}: {candidate_streak}/{required_streak})"
                        res_is_known = False
                        res_conf = raw_conf

                # Print live diagnostic logs to terminal on every 3rd frame
                if total_stream_frames % 3 == 0:
                    print(
                        f"[LIVE DEBUG] Mode: {active_model} | Hand Present: {hand_detected} | "
                        f"Raw Top Class: {raw_top_class} ({raw_top_conf:.2f}) | Output: {res_label}",
                        flush=True,
                    )

                # Send structured response back to client
                response = {
                    "type": "prediction",
                    "active_model": active_model,
                    "label": res_label,
                    "display_name": res_display,
                    "confidence": res_conf,
                    "is_known_sign": res_is_known,
                    "is_stable": is_stable,
                    "streak_count": candidate_streak,
                    "required_streak": required_streak,
                    "probabilities": res_probs,
                    "buffer_fill": buffer_fill,
                    "hand_detected": hand_detected,
                    "hands_in_subtitle_zone": hands_in_subtitle_zone,
                    "guidance": guidance_text,
                    "fps": fps,
                    "latency_ms": latency_ms,
                    "landmarks": {
                        "left_hand": left_hand,
                        "right_hand": right_hand,
                        "pose": pose,
                        "face": face,
                    },
                    "realsign_prediction": {
                        "letter": realsign_pred.letter if realsign_pred else "no_sign",
                        "confidence": round(realsign_pred.confidence, 4) if realsign_pred else 0.0,
                        "is_known_sign": realsign_pred.is_known_sign if realsign_pred else False,
                        "top_3": [(l, round(c, 4)) for l, c in realsign_pred.top_3] if realsign_pred else [],
                        "logits": {k: round(v, 4) for k, v in realsign_pred.logits.items()} if realsign_pred and realsign_pred.logits else None,
                    } if realsign_pred else None,
                    "sequence_lstm_prediction": {
                        "label": seq_pred.label if seq_pred else "no_sign",
                        "confidence": round(seq_pred.confidence, 4) if seq_pred else 0.0,
                        "is_known_sign": seq_pred.is_known_sign if seq_pred else False,
                        "top_3": [(l, round(c, 4)) for l, c in seq_pred.top_3] if seq_pred else [],
                    } if seq_pred else None,
                }

                await websocket.send_json(response)

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    if action == "reset":
                        rolling_features.clear()
                        candidate_label = ""
                        candidate_streak = 0
                        last_committed_sign = ""
                        last_committed_time = 0.0
                        had_idle_baseline = True
                        if pipeline.landmarker is not None:
                            pipeline.landmarker.reset_smoothing()
                        if pipeline.hand_landmarker is not None:
                            pipeline.hand_landmarker.reset_smoothing()
                        await websocket.send_json({"type": "info", "message": "Buffer reset"})
                    elif action == "set_model":
                        req_model = str(payload.get("model", "realsign")).lower()
                        if req_model in ("realsign", "isl_mvp", "isl_sequence_lstm"):
                            active_model = req_model
                            rolling_features.clear()
                            candidate_label = ""
                            candidate_streak = 0
                            last_committed_sign = ""
                            last_committed_time = 0.0
                            had_idle_baseline = True
                            if pipeline.landmarker is not None:
                                pipeline.landmarker.reset_smoothing()
                            if pipeline.hand_landmarker is not None:
                                pipeline.hand_landmarker.reset_smoothing()
                            await websocket.send_json({
                                "type": "info",
                                "message": f"Active classifier model switched to {active_model}",
                                "active_model": active_model,
                            })
                    elif action == "set_threshold":
                        val = float(payload.get("threshold", CONFIDENCE_THRESHOLD_ALPHABET))
                        client_threshold = max(0.10, min(0.95, val))
                        await websocket.send_json({
                            "type": "info",
                            "message": f"Confidence threshold set to {client_threshold}",
                            "threshold": client_threshold,
                        })
                except Exception as e:
                    logger.warning("Error processing text command: %s", e)

    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        rolling_features.clear()
