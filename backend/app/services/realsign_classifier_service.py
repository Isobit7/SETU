"""
SETU ΓÇö Classifier Service & PyTorch Model Definition
Provides the RealSignMLP architecture and inference engine for 26-class alphabet prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from backend.app.schemas.predict import LandmarkRequest, PredictionResponse, TopCandidate

logger = logging.getLogger("setu.classifier")

# 26 standard alphabet classes (A to Z)
CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


# --- PyTorch Model Architecture ---
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


# --- 63-D Landmark Normalization Utility ---
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


# --- Inference Service ---
class RealSignInferenceService:
    """Manages model checkpoint loading and live GPU/CPU inference."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[RealSignMLP] = None
        self.classes: List[str] = list(CLASSES_26)
        self.model_path: Optional[Path] = None

        self.candidate_paths = [
            self.project_root / "models" / "weights" / "realsign_26class_mlp.pth",
            self.project_root / "models" / "weights" / "realsign_26class_mlp.pt",
            self.project_root / "models" / "realsign_26class_mlp.pt",
            self.project_root / "models" / "realsign_26class_mlp.pth",
            self.project_root / "checkpoints" / "realsign_26class_mlp.pt",
            self.project_root / "checkpoints" / "realsign_26class_mlp.pth",
        ]

    def load_model(self) -> RealSignMLP:
        """Loads weights from disk into memory."""
        target_path = None
        for p in self.candidate_paths:
            if p.exists():
                target_path = p
                break

        if target_path is None:
            logger.warning(
                "No checkpoint found at %s. Initializing blank model.", self.candidate_paths
            )
            model = RealSignMLP(input_dim=63, hidden1=128, hidden2=64, num_classes=26)
            model.to(self.device)
            model.eval()
            self.model = model
            self.classes = list(CLASSES_26)
            return model

        logger.info("Loading RealSignMLP from %s on device: %s", target_path, self.device)
        checkpoint = torch.load(str(target_path), map_location=self.device, weights_only=False)

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

        model.to(self.device)
        model.eval()

        self.model = model
        self.classes = [str(c) for c in classes]
        self.model_path = target_path
        logger.info(
            "RealSignMLP successfully loaded (%d classes: %s)", len(self.classes), self.classes[:5]
        )
        return model

    def ensure_loaded(self) -> RealSignMLP:
        """Idempotently guarantees that model is loaded and ready."""
        if self.model is None:
            return self.load_model()
        return self.model

    @torch.no_grad()
    def predict_vector(self, feat_array: np.ndarray) -> PredictionResponse:
        """Runs fast tensor forward-pass on 63-D feature array."""
        model = self.ensure_loaded()

        input_tensor = torch.from_numpy(feat_array).unsqueeze(0).to(self.device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_k = min(3, len(self.classes))
        top_probs, top_indices = torch.topk(probs, k=top_k)

        top_3: List[TopCandidate] = []
        for p, idx in zip(top_probs, top_indices):
            c_idx = idx.item()
            c_name = self.classes[c_idx] if c_idx < len(self.classes) else str(c_idx)
            top_3.append(TopCandidate(letter=c_name, confidence=round(float(p.item()), 4)))

        top_pred = top_3[0]
        return PredictionResponse(
            letter=top_pred.letter,
            confidence=top_pred.confidence,
            top_3=top_3,
        )

    def predict_request(self, request: LandmarkRequest) -> PredictionResponse:
        """Processes LandmarkRequest with either raw 63-D features or 21 3D landmarks."""
        if request.features is not None:
            if len(request.features) != 63:
                raise ValueError(
                    f"Feature vector must have exactly 63 elements, received {len(request.features)}"
                )
            feat_array = np.array(request.features, dtype=np.float32)
        elif request.landmarks is not None:
            if len(request.landmarks) != 21:
                raise ValueError(
                    f"Landmarks list must have exactly 21 landmarks, received {len(request.landmarks)}"
                )
            is_left = (request.dominant_hand or "right").lower() == "left"
            feat_array = normalize_landmarks_to_63d(request.landmarks, is_left_hand=is_left)
        else:
            raise ValueError(
                "Either 'features' (63-D float array) or 'landmarks' (21 3D points) must be provided."
            )

        return self.predict_vector(feat_array)


# Shared singleton instance
classifier_service = RealSignInferenceService()
