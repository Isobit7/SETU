"""
SETU — Backward-Compatible Backend Entrypoint
Re-exports the consolidated FastAPI application and components from backend.app.
Allows:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app, HealthResponse
from backend.app.schemas.predict import (
    LandmarkRequest,
    PredictionResponse,
    TopCandidate,
    PredictRequest,
    PredictResponse,
)
from backend.app.services.realsign_classifier_service import (
    CLASSES_26,
    RealSignMLP,
    RealSignInferenceService,
    classifier_service,
    normalize_landmarks_to_63d,
)

# Compatibility aliases for legacy scripts
load_model = classifier_service.load_model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
