"""
SETU — 26-Class Sign Language Alphabet Recognition Backend
FastAPI server serving the trained PyTorch RealSignMLP model for real-time inference.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.api.endpoints.predict import router as predict_router
from backend.app.services.classifier import classifier_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setu.backend")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    model_path: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes model on server startup and cleans up on shutdown."""
    try:
        classifier_service.load_model()
    except Exception as e:
        logger.error("Failed to load RealSignMLP during startup: %s", e)
    yield


# FastAPI Application
app = FastAPI(
    title="SETU 26-Class Sign Language Alphabet API",
    description="Real-time alphabet recognition service powered by PyTorch RealSignMLP",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend applications (localhost:3000, 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(predict_router, prefix="/api")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting model status."""
    is_loaded = classifier_service.model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        device=str(classifier_service.device),
        num_classes=len(classifier_service.classes),
        model_path=str(classifier_service.model_path) if classifier_service.model_path else None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
