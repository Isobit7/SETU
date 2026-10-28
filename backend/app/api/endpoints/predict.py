"""
SETU — Alphabet Prediction Endpoints
FastAPI router handling POST /api/predict requests.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.predict import LandmarkRequest, PredictionResponse
from backend.app.services.realsign_classifier_service import classifier_service

logger = logging.getLogger("setu.api.predict")

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Indian Sign Language alphabet letter (A–Z)",
    description=(
        "Accepts single-hand (63-D, 89-D) or dual-hand (126-D) normalized feature vectors, "
        "or MediaPipe 3D hand landmarks (primary + optional secondary hand for bimanual ISL letters B, D, P, Q, T). "
        "Returns top-ranked predicted letter (A–Z) and top-3 candidates with confidence scores."
    ),
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
