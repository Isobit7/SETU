"""
SETU — Consolidated Enterprise FastAPI Application & WebSocket Service
Hosts all REST endpoints, ML model catalogs, video inference, and low-latency streaming WebSocket.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.api.endpoints.dataset import router as dataset_router
from backend.app.api.endpoints.models import router as models_router
from backend.app.api.endpoints.predict import router as predict_router
from backend.app.api.endpoints.video import router as video_router
from backend.app.api.endpoints.vocabulary import router as vocabulary_router
from backend.app.services.realsign_classifier_service import classifier_service
from backend.app.services.pipeline_manager import pipeline
from backend.app.websocket.recognize import websocket_recognize

logger = logging.getLogger("setu.backend")
logging.basicConfig(level=logging.INFO)


class HealthResponse(BaseModel):
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
        "realsign_26class_alphabet",
        "isl_mvp_5sign",
        "isl_sequence_lstm",
        "ai4bharat_include_263sign",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes vision models and classifier services on startup, closes them on shutdown."""
    logger.info("Starting up SETU Unified API & WebSocket Engine...")
    try:
        classifier_service.load_model()
    except Exception as e:
        logger.error("Failed to load RealSignMLP during startup: %s", e)

    try:
        pipeline.load()
    except Exception as e:
        logger.error("Failed to load Vision pipeline on startup: %s", e)

    yield

    logger.info("Shutting down SETU ML Services...")
    pipeline.close()


app = FastAPI(
    title="SETU Sign Language Recognition & Translation API",
    description="Enterprise REST & WebSocket Server connecting clients to MediaPipe and PyTorch inference engines.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured Request Logging & Correlation ID Middleware
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

# Mount Modular API Routers
app.include_router(predict_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(vocabulary_router, prefix="/api")


# WebSocket Streaming Endpoints
@app.websocket("/api/v1/ws/recognize")
@app.websocket("/ws/recognize")
async def recognize_ws(
    websocket: WebSocket,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Low-latency WebSocket streaming endpoint for live sign language recognition."""
    await websocket_recognize(websocket, model=model, threshold=threshold)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Consolidated health check endpoint reporting model and pipeline readiness."""
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
            "isl_mvp_5sign",
            "isl_sequence_lstm",
            "ai4bharat_include_263sign",
        ],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
