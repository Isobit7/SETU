"""
SETU — 26-Class Sign Language Alphabet Recognition Backend API
FastAPI server serving the trained PyTorch RealSignMLP model for real-time inference.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setu.backend")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Checkpoint candidate locations
MODEL_PATHS = [
    PROJECT_ROOT / "models" / "realsign_26class_mlp.pt",
    PROJECT_ROOT / "models" / "realsign_26class_mlp.pth",
    PROJECT_ROOT / "checkpoints" / "realsign_26class_mlp.pt",
    PROJECT_ROOT / "checkpoints" / "realsign_26class_mlp.pth",
]

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


# --- Model Definition ---
class RealSignMLP(nn.Module):
    """PyTorch MLP Classifier for 26-class ISL Hand Alphabet Recognition."""

    def __init__(
        self,
        input_dim: int = 63,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 26,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Normalization Utilities ---
def normalize_landmarks_to_63d(
    landmarks: Sequence[Union[Dict[str, float], Sequence[float]]],
    is_left_hand: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Transforms 21 3D hand landmarks into a canonical wrist-centered,
    scale- and rotation-invariant 63-D vector.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, received {len(landmarks)}")

    pts = np.empty((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if isinstance(lm, dict):
            pts[i, 0] = float(lm.get("x", 0.0))
            pts[i, 1] = float(lm.get("y", 0.0))
            pts[i, 2] = float(lm.get("z", 0.0))
        elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 3:
            pts[i, 0] = float(lm[0])
            pts[i, 1] = float(lm[1])
            pts[i, 2] = float(lm[2])
        else:
            raise ValueError(f"Invalid landmark at index {i}: {lm}")

    if is_left_hand:
        pts[:, 0] = 1.0 - pts[:, 0]

    # 1. Translate so wrist (0) is at origin (0, 0, 0)
    wrist = pts[0].copy()
    translated = pts - wrist

    # 2. Scale by Euclidean distance from wrist (0) to middle finger MCP (9)
    v_wrist_middle = translated[9]
    scale = float(np.linalg.norm(v_wrist_middle))
    if scale < eps:
        scale = 1.0
    scaled = translated / scale

    # 3. In-plane rotation around Z axis so (x, y) vector of middle MCP aligns with (0, -1)
    v_x = float(scaled[9, 0])
    v_y = float(scaled[9, 1])
    norm_xy = float(np.hypot(v_x, v_y))
    if norm_xy > eps:
        theta = float(np.arctan2(-v_x, -v_y))
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x_rot = scaled[:, 0] * cos_t - scaled[:, 1] * sin_t
        y_rot = scaled[:, 0] * sin_t + scaled[:, 1] * cos_t
        z_rot = scaled[:, 2].copy()
    else:
        x_rot = scaled[:, 0].copy()
        y_rot = scaled[:, 1].copy()
        z_rot = scaled[:, 2].copy()

    # 4. Out-of-plane rotation around X axis so (y', z') aligns with (-1, 0)
    y9 = float(y_rot[9])
    z9 = float(z_rot[9])
    norm_yz = float(np.hypot(y9, z9))
    if norm_yz > eps and abs(z9) > eps:
        phi = float(np.arctan2(z9, -y9))
        cos_p = float(np.cos(phi))
        sin_p = float(np.sin(phi))
        y_rot2 = y_rot * cos_p - z_rot * sin_p
        z_rot2 = y_rot * sin_p + z_rot * cos_p
        y_rot = y_rot2
        z_rot = z_rot2

    # Canonical exact alignment
    x_rot[0] = 0.0
    y_rot[0] = 0.0
    z_rot[0] = 0.0
    x_rot[9] = 0.0
    y_rot[9] = -1.0
    z_rot[9] = 0.0

    normalized = np.column_stack([x_rot, y_rot, z_rot]).astype(np.float32)
    return normalized.flatten()


# --- Pydantic Schemas ---
class PredictRequest(BaseModel):
    features: Optional[List[float]] = Field(
        default=None,
        description="63-dimensional normalized feature vector",
    )
    landmarks: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="21 hand landmarks, each with 'x', 'y', and optional 'z'",
    )
    dominant_hand: Optional[str] = Field(
        default="right",
        description="'right' or 'left' (for mirroring geometry)",
    )


class TopCandidate(BaseModel):
    letter: str
    confidence: float


class PredictResponse(BaseModel):
    letter: str
    confidence: float
    top_3: List[TopCandidate]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    model_path: Optional[str] = None


# Global model state
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[RealSignMLP] = None
_classes: List[str] = list(CLASSES_26)
_loaded_path: Optional[Path] = None


def load_model() -> RealSignMLP:
    """Loads and returns the RealSignMLP classifier."""
    global _model, _classes, _loaded_path

    target_path = None
    for p in MODEL_PATHS:
        if p.exists():
            target_path = p
            break

    if target_path is None:
        logger.warning("No checkpoint found at candidate paths: %s. Initializing fresh model.", MODEL_PATHS)
        model = RealSignMLP(input_dim=63, hidden1=128, hidden2=64, num_classes=26)
        model.to(_device)
        model.eval()
        _model = model
        _classes = list(CLASSES_26)
        return model

    logger.info("Loading RealSignMLP weights from: %s on device: %s", target_path, _device)
    checkpoint = torch.load(str(target_path), map_location=_device, weights_only=False)

    if isinstance(checkpoint, dict):
        arch = checkpoint.get("architecture", {})
        input_dim = arch.get("input_dim", 63)
        hidden1 = arch.get("hidden1", 128)
        hidden2 = arch.get("hidden2", 64)
        classes = checkpoint.get("classes", CLASSES_26)
        num_classes = len(classes)

        model = RealSignMLP(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
        )

        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
        classes = CLASSES_26
    else:
        model = RealSignMLP(input_dim=63, hidden1=128, hidden2=64, num_classes=26)
        model.load_state_dict(checkpoint)
        classes = CLASSES_26

    model.to(_device)
    model.eval()

    _model = model
    _classes = [str(c) for c in classes]
    _loaded_path = target_path
    logger.info("RealSignMLP successfully loaded (%d classes: %s)", len(_classes), _classes[:5])
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes model on server startup and cleans up on shutdown."""
    try:
        load_model()
    except Exception as e:
        logger.error("Failed to load RealSignMLP during startup: %s", e)
    yield


# --- FastAPI Application ---
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


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting model status."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        device=str(_device),
        num_classes=len(_classes),
        model_path=str(_loaded_path) if _loaded_path else None,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Predicts the sign language letter from either a 63-D feature vector
    or 21 3D hand landmarks.
    """
    global _model, _classes
    if _model is None:
        _model = load_model()

    # Determine input feature vector
    if payload.features is not None:
        if len(payload.features) != 63:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Feature vector must have exactly 63 elements, received {len(payload.features)}",
            )
        feat_array = np.array(payload.features, dtype=np.float32)
    elif payload.landmarks is not None:
        if len(payload.landmarks) != 21:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Landmarks list must have exactly 21 landmarks, received {len(payload.landmarks)}",
            )
        try:
            is_left = (payload.dominant_hand or "right").lower() == "left"
            feat_array = normalize_landmarks_to_63d(payload.landmarks, is_left_hand=is_left)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to normalize landmarks: {e}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'features' (63-D float array) or 'landmarks' (21 3D points) must be provided.",
        )

    # Tensor inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(feat_array).unsqueeze(0).to(_device)
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Top 3 candidates
        top_k = min(3, len(_classes))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        top_3: List[TopCandidate] = []
        for p, idx in zip(top_probs, top_indices):
            c_idx = idx.item()
            c_name = _classes[c_idx] if c_idx < len(_classes) else str(c_idx)
            top_3.append(TopCandidate(letter=c_name, confidence=round(float(p.item()), 4)))

    top_pred = top_3[0]
    return PredictResponse(
        letter=top_pred.letter,
        confidence=top_pred.confidence,
        top_3=top_3,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
