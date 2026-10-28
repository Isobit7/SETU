"""
SETU — Prediction API Schemas (Pydantic v2)
Request and response models for real-time sign language alphabet recognition.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LandmarkRequest(BaseModel):
    """Payload for hand landmark prediction endpoint supporting single-hand and dual-hand ISL."""
    features: Optional[List[float]] = Field(
        default=None,
        description="Normalized feature vector: 126-D dual-hand (recommended for ISL), 89-D enriched, or 63-D single-hand",
        examples=[[0.0] * 126],
    )
    landmarks: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="21 primary hand landmarks (wrist + fingers), each with 'x', 'y', and optional 'z'",
    )
    secondary_landmarks: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="21 secondary hand landmarks for authentic bimanual ISL letters (B, D, P, Q, T)",
    )
    dominant_hand: Optional[str] = Field(
        default="right",
        description="'right' or 'left' (used for canonical coordinate mirroring)",
    )


class TopCandidate(BaseModel):
    """Single candidate prediction with letter and confidence score."""
    letter: str = Field(description="Recognized alphabet letter (A-Z)")
    confidence: float = Field(description="Softmax probability in range [0.0, 1.0]")


class PredictionResponse(BaseModel):
    """Structured response from sign language classifier."""
    letter: str = Field(description="Top-ranked alphabet letter (A-Z)")
    confidence: float = Field(description="Confidence of top-ranked letter")
    top_3: List[TopCandidate] = Field(
        description="Top 3 predicted candidates with confidence percentages"
    )


# Backward-compatible type aliases
PredictRequest = LandmarkRequest
PredictResponse = PredictionResponse
