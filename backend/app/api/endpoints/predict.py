"""
SETU — Alphabet Prediction Endpoints
FastAPI router handling POST /api/predict requests.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.predict import LandmarkRequest, PredictionResponse
from backend.app.services.classifier import classifier_service

logger = logging.getLogger("setu.api.predict")

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict sign language alphabet letter",
    description="Accepts 63-D normalized feature vector or 21 3D hand landmarks and returns predicted letter and top-3 candidates.",
)
def predict_letter(payload: LandmarkRequest) -> PredictionResponse:
    """Classifies single-frame hand pose into one of 26 alphabet letters (A-Z)."""
    try:
        return classifier_service.predict_request(payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Inference execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {e}",
        )
