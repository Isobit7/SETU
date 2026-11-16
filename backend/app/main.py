"""
SETU — Consolidated Enterprise FastAPI Application & WebSocket Service
======================================================================
Architecture Role:
    This file is the master ASGI (Asynchronous Server Gateway Interface) entrypoint.
    It performs 4 critical system operations:
    1. Lifespan Management: Pre-loads heavyweight neural network weights (MediaPipe, RealSign MLP)
       into memory during server startup so initial incoming WebSocket requests face zero cold-start delay.
    2. Cross-Origin Resource Sharing (CORS): Permits the React/Vite frontend (running on port 5173)
       to access REST and WebSocket endpoints securely.
    3. Modular API Router Aggregation: Mounts specialized REST domains (/predict, /models, /nlp,
       /vocabulary, /dataset, /video) under a unified '/api' prefix.
    4. Duplex WebSocket Streaming: Establishes the high-throughput, low-latency streaming pipeline
       ('/ws/recognize') transferring 30 FPS video frames to the ML inference engine.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# API Router Imports (Modular REST endpoints partitioned by domain)
# ---------------------------------------------------------------------------
from backend.app.api.endpoints.dataset import router as dataset_router
from backend.app.api.endpoints.models import router as models_router
from backend.app.api.endpoints.predict import router as predict_router
from backend.app.api.endpoints.video import router as video_router
from backend.app.api.endpoints.vocabulary import router as vocabulary_router
from backend.app.api.endpoints.nlp import router as nlp_router

# ---------------------------------------------------------------------------
# Core Machine Learning & Pipeline Service Imports
# ---------------------------------------------------------------------------
from backend.app.services.realsign_classifier_service import classifier_service
from backend.app.services.pipeline_manager import pipeline
from backend.app.websocket.recognize import websocket_recognize

# Configure structured system logging
logger = logging.getLogger("setu.backend")
logging.basicConfig(level=logging.INFO)


class HealthResponse(BaseModel):
    """
    Pydantic schema defining the structured response returned by the /health endpoint.
    Used by frontend health checkers and load balancers to verify model readiness.
    """
    status: str
    pipeline_ready: bool
    model_loaded: bool
    device: str
    num_classes: int
    supported_signs: List[str]
    model_version: str
    include_ready: bool
    realsign_ready: bool = True
    realsign_classes_count: int = 26
    sequence_lstm_ready: bool = True
    sequence_lstm_classes_count: int = 7
    active_classifiers: List[str] = [
        "realsign_multisigner_mlp",
        "realsign_26class_alphabet",
        "isl_numerals",
        "isl_hindi_vowels",
        "isl_sequence_lstm",
        "isl_dynamic_unified_287",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager:
    Replaces deprecated on_event('startup') and on_event('shutdown').
    - Startup phase: Pre-warms the RealSign neural classifier and MediaPipe Holistic pipeline.
      This allocates GPU/CPU memory once, preventing frame drops during the first user connection.
    - Shutdown phase: Gracefully releases camera handles and clears PyTorch CUDA caches.
    """
    logger.info("Starting up SETU Unified API & WebSocket Engine...")
    try:
        classifier_service.load_model()
        logger.info("RealSign MLP classifier loaded successfully.")
    except Exception as e:
        logger.error("Failed to load RealSignMLP during startup: %s", e)

    try:
        pipeline.load()
        logger.info("MediaPipe Holistic Vision pipeline loaded successfully.")
    except Exception as e:
        logger.error("Failed to load Vision pipeline on startup: %s", e)

    yield  # Application serves incoming HTTP requests and WebSockets here

    logger.info("Shutting down SETU ML Services...")
    pipeline.close()


# Instantiate the core FastAPI application
app = FastAPI(
    title="SETU Sign Language Recognition & Translation API",
    description="Enterprise REST & WebSocket Server connecting clients to MediaPipe and PyTorch inference engines.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) Middleware
# Required so the frontend (http://localhost:5173) can communicate with backend (http://localhost:8000)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request Correlation ID & Latency Telemetry Middleware
# Stamps each incoming HTTP request with a unique UUID and calculates millisecond duration.
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request, call_next):
    import time
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = str(duration_ms)
    logger.info(
        "[%s] %s %s -> %d (%sms)",
        request_id[:8],
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Mount Modular API Routers
# Organizes endpoints cleanly: /api/predict, /api/models, /api/nlp, etc.
# ---------------------------------------------------------------------------
app.include_router(predict_router, prefix="/api")      # Single-image inference
app.include_router(models_router, prefix="/api")       # Model catalogue & runtime status
app.include_router(video_router, prefix="/api")        # Batch video upload & processing
app.include_router(dataset_router, prefix="/api")      # User custom sign data recording
app.include_router(vocabulary_router, prefix="/api")   # Vocabulary list & domain packs
app.include_router(nlp_router, prefix="/api")          # Indic grammar translation & spelling


# ---------------------------------------------------------------------------
# Real-Time WebSocket Streaming Endpoints
# Handles continuous 30 FPS video streaming from frontend canvas.
# Mounted on two paths for backwards-compatibility.
# ---------------------------------------------------------------------------
@app.websocket("/api/v1/ws/recognize")
@app.websocket("/ws/recognize")
async def recognize_ws(
    websocket: WebSocket,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """
    Duplex WebSocket connection handler.
    Receives compressed JPEG frames, invokes MediaPipe landmark extraction,
    runs neural classification, debounces predictions, and returns JSON in <32ms.
    """
    await websocket_recognize(websocket, model=model, threshold=threshold)


# ---------------------------------------------------------------------------
# Root Welcome & Navigation Endpoint
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint welcoming users and directing them to frontend app and API documentation."""
    return {
        "name": "SETU Sign Language Recognition & Translation Engine",
        "status": "online",
        "frontend_url": "http://localhost:5173",
        "docs_url": "/docs",
        "health_url": "/health",
        "models_url": "/api/v1/models",
    }


# ---------------------------------------------------------------------------
# Consolidated System Health Check Endpoint
# Used by frontend to determine whether backend models are warmed up.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Evaluates backend subsystems and reports:
    1. Pipeline readiness (MediaPipe loaded)
    2. Active classifier weights (PyTorch/ONNX)
    3. Hardware acceleration device (CUDA GPU vs CPU)
    4. Supported vocabulary count (26 alphabets, 10 numerals, 49 phrases)
    """
    is_classifier_loaded = classifier_service.model is not None
    classes = pipeline.classifier.class_names if pipeline.classifier else []
    include_loaded = pipeline.include_service.is_loaded if pipeline.include_service else False
    realsign_loaded = pipeline.realsign_classifier is not None
    realsign_count = len(pipeline.realsign_classifier.class_names) if realsign_loaded else 26
    sequence_lstm_loaded = pipeline.sequence_lstm_classifier is not None
    sequence_lstm_count = len(pipeline.sequence_lstm_classifier.class_names) if sequence_lstm_loaded else 7

    is_healthy = pipeline.is_ready or is_classifier_loaded

    return HealthResponse(
        status="healthy" if is_healthy else "initializing",
        pipeline_ready=pipeline.is_ready,
        model_loaded=is_classifier_loaded,
        device=str(classifier_service.device),
        num_classes=realsign_count,
        supported_signs=[c for c in classes if c != "no_sign"],
        model_version="SETU-MultiEngine-v1",
        include_ready=include_loaded,
        realsign_ready=realsign_loaded,
        realsign_classes_count=realsign_count,
        sequence_lstm_ready=sequence_lstm_loaded,
        sequence_lstm_classes_count=sequence_lstm_count,
        active_classifiers=[
            "realsign_26class_alphabet",
            "isl_numerals",
            "isl_hindi_vowels",
            "isl_sequence_lstm",
            "isl_dynamic_unified_287",
        ],
    )


if __name__ == "__main__":
    import uvicorn
    # When executed directly, run Uvicorn on localhost port 8000
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
