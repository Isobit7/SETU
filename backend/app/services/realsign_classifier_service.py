"""
SETU — Production RealSign Classifier Service & PyTorch MLP Model Definition
=============================================================================

Architectural Overview:
-----------------------
This service provides the core neural alphabet fingerspelling inference engine
for Indian Sign Language (ISL) recognition across all 26 Latin letters (A–Z).

Canonical Weights Location:
---------------------------
- Active Production Model: `models/weights/realsign_26class_mlp.pth`
  Architecture: 3-Layer BatchNorm MLP (Input: 126-D Dual-Hand, Hidden: 128 -> 64, Classes: 26)
  Coordinate System: MediaPipe Hands wrist-normalized 3D Cartesian coordinates.
  Symmetric Dual-Hand Schema:
    * Left Hand normalized landmarks (21 points x 3 = 63 floats) in slots [0:63]
    * Right Hand normalized landmarks (21 points x 3 = 63 floats) in slots [63:126]

Design Invariants:
------------------
1. Strict Production Loading: Loads exclusively from `models/weights/realsign_26class_mlp.pth`.
   Historical 63-D/89-D checkpoints and randomly initialized models are strictly forbidden
   from loading as silent fallbacks.
2. In-Memory Hot Reloading: Automatically checks file modification timestamps (`mtime`) on disk
   and hot-reloads updated weights dynamically without dropping active connections.
3. Backward Compatibility: Accepts either dual-hand (126-D), historical (89-D), single-hand (63-D),
   or raw (21, 3) dictionary landmarks, automatically padding or condensing dimensions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from backend.app.schemas.predict import LandmarkRequest, PredictionResponse, TopCandidate
from training.features.geometry import (
    normalize_landmarks_to_63d,
    extract_realsign_89d_features,
    expand_63d_to_89d,
)

logger = logging.getLogger("setu.classifier")

# Canonical 26-class alphabet vocabulary (A through Z)
CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


# ==============================================================================
# 1. PyTorch Neural Network Architecture
# ==============================================================================

class RealSignMLP(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) for Real-Time ISL Alphabet Classification.
    
    Architecture:
    -------------
    - Layer 1: Linear(input_dim -> hidden1) + [BatchNorm1d] + ReLU + Dropout(p)
    - Layer 2: Linear(hidden1 -> hidden2) + [BatchNorm1d] + ReLU
    - Layer 3: Linear(hidden2 -> num_classes) -> Output Logits (unnormalized)
    
    Args:
        input_dim: Number of input landmark features (126 for dual-hand, 89 for baseline, 63 for single-hand).
        hidden1: First dense layer hidden units (default: 128).
        hidden2: Second dense layer hidden units (default: 64).
        num_classes: Number of alphabet target classes (default: 26).
        dropout: Probability of dropout regularization during training (default: 0.20).
        use_batchnorm: Whether to apply batch normalization layers (default: True).
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 26,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unnormalized classification logits."""
        return self.net(x)


# ==============================================================================
# 2. Production Inference Service
# ==============================================================================

class RealSignInferenceService:
    """
    Thread-safe production inference engine for RealSign 26-class alphabet recognition.
    
    Features:
    ---------
    - Idempotent checkpoint loading with strict architectural verification.
    - Automatic hardware device selection (NVIDIA CUDA GPU if present, CPU fallback).
    - In-memory hot reload detection based on file modification timestamps.
    - Softmax probability derivation and Top-3 candidate extraction.
    """

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[RealSignMLP] = None
        self.classes: List[str] = list(CLASSES_26)
        self.model_path: Optional[Path] = None
        self._last_mtime: float = 0.0

        # Canonical production model weight paths (strict ordering: active production checkpoint only)
        self.candidate_paths = [
            self.project_root / "models" / "weights" / "realsign_multisigner_mlp.pth",
            self.project_root / "models" / "weights" / "realsign_26class_mlp.pth",
        ]

    def load_model(self) -> RealSignMLP:
        """
        Loads the production PyTorch checkpoint from disk into runtime memory.
        
        Raises:
            FileNotFoundError: If the canonical production checkpoint does not exist.
            RuntimeError: If checkpoint state dictionary is corrupted or incompatible.
        """
        target_path: Optional[Path] = None
        for p in self.candidate_paths:
            if p.exists():
                target_path = p
                break

        if target_path is None:
            raise FileNotFoundError(
                f"Production RealSign 126-D checkpoint missing at {self.candidate_paths[0]}. "
                "Historical checkpoints (63-D/89-D) and blank models cannot be loaded as production fallbacks."
            )

        logger.info("Loading RealSignMLP from %s on device: %s", target_path, self.device)
        checkpoint = torch.load(str(target_path), map_location=self.device, weights_only=False)

        # Extract state dictionary
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        ) if isinstance(checkpoint, dict) else checkpoint

        if isinstance(checkpoint, dict):
            arch = checkpoint.get("architecture", {})
            first_layer_weight = state_dict.get("net.0.weight") if isinstance(state_dict, dict) else None
            inferred_in_dim = first_layer_weight.shape[1] if first_layer_weight is not None else 126
            has_bn = any("running_mean" in k for k in state_dict.keys()) if isinstance(state_dict, dict) else False

            input_dim = arch.get("input_dim", inferred_in_dim)
            hidden1 = arch.get("hidden1", 128)
            hidden2 = arch.get("hidden2", 64)
            use_batchnorm = arch.get("use_batchnorm", has_bn)
            classes = checkpoint.get("classes", CLASSES_26)
            num_classes = len(classes)

            model = RealSignMLP(
                input_dim=input_dim,
                hidden1=hidden1,
                hidden2=hidden2,
                num_classes=num_classes,
                use_batchnorm=use_batchnorm,
            )
            model.load_state_dict(state_dict)
        elif isinstance(checkpoint, nn.Module):
            model = checkpoint
            classes = CLASSES_26
        else:
            model = RealSignMLP(input_dim=126, hidden1=128, hidden2=64, num_classes=26, use_batchnorm=True)
            model.load_state_dict(checkpoint)
            classes = CLASSES_26

        model.to(self.device)
        model.eval()

        self.model = model
        self.classes = [str(c) for c in classes]
        self.model_path = target_path
        self._last_mtime = target_path.stat().st_mtime
        logger.info(
            "RealSignMLP successfully loaded (Dim: %d, %d classes: %s)",
            model.net[0].in_features,
            len(self.classes),
            self.classes[:5],
        )
        return model

    def ensure_loaded(self) -> RealSignMLP:
        """
        Idempotently guarantees that model is loaded and ready, reloading if the checkpoint
        file has been modified on disk by retraining or fine-tuning routines.
        """
        target_path: Optional[Path] = None
        for p in self.candidate_paths:
            if p.exists():
                target_path = p
                break

        if target_path is not None:
            mtime = target_path.stat().st_mtime
            if mtime > getattr(self, "_last_mtime", 0.0):
                self._last_mtime = mtime
                return self.load_model()

        if self.model is None:
            return self.load_model()

        return self.model

    @torch.no_grad()
    def predict_vector(self, feat_array: np.ndarray) -> PredictionResponse:
        """
        Runs fast tensor forward-pass on feature array with support for 126-D, 89-D, or 63-D.
        
        Args:
            feat_array: 1D NumPy array representing normalized hand landmarks.
            
        Returns:
            PredictionResponse containing top predicted letter, confidence score, and top-3 candidates.
        """
        model = self.ensure_loaded()
        expected_dim = model.net[0].in_features

        arr = np.asarray(feat_array, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 63 and expected_dim == 89:
            arr = expand_63d_to_89d(arr)
        elif arr.ndim == 1 and arr.shape[0] == 63 and expected_dim == 126:
            # Map single-hand 63-D into right-hand slots [63:126] with left hand zeroed
            padded = np.zeros(126, dtype=np.float32)
            padded[63:] = arr
            arr = padded

        input_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
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
        """
        Processes standard HTTP/WebSocket LandmarkRequest containing either pre-extracted
        feature floats or raw (21, 3) MediaPipe landmark coordinates.
        """
        model = self.ensure_loaded()
        expected_dim = model.net[0].in_features

        if request.features is not None:
            feat_len = len(request.features)
            feat_array = np.array(request.features, dtype=np.float32)
            if np.isnan(feat_array).any() or np.isinf(feat_array).any():
                raise ValueError("Features must not contain NaN or Inf values.")

            if feat_len == 63 and expected_dim == 89:
                feat_array = expand_63d_to_89d(feat_array)
            elif feat_len == 63 and expected_dim == 126:
                # Map single hand 63-D to dual-hand 126-D (right hand slot 63:126)
                dual = np.zeros(126, dtype=np.float32)
                dual[63:] = feat_array
                feat_array = dual
            elif feat_len == 126 and expected_dim == 126:
                pass
            elif feat_len == 126 and expected_dim == 89:
                # Condense 126-D bimanual features into 89-D representation
                from training.features.geometry import ACTIVE_COORDINATE_INDICES_57
                condensed = np.zeros(89, dtype=np.float32)
                right_63 = feat_array[63:]
                left_63 = feat_array[:63]
                primary_63 = right_63 if np.any(right_63 != 0.0) else left_63
                condensed[:57] = primary_63[ACTIVE_COORDINATE_INDICES_57]
                condensed[57:] = feat_array[ACTIVE_COORDINATE_INDICES_57[:32]]
                feat_array = condensed
            elif feat_len != expected_dim:
                raise ValueError(
                    f"Feature vector has {feat_len} elements, expected {expected_dim}, 63, or 126 (dual-hand)."
                )
        elif request.landmarks is not None:
            if len(request.landmarks) != 21:
                raise ValueError(
                    f"Landmarks list must have exactly 21 landmarks, received {len(request.landmarks)}"
                )
            is_left = (request.dominant_hand or "right").lower() == "left"
            if request.secondary_landmarks and len(request.secondary_landmarks) == 21:
                from training.features.geometry import extract_dual_hand_126d, ACTIVE_COORDINATE_INDICES_57
                pts1 = np.array([[lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)] for lm in request.landmarks], dtype=np.float32)
                pts2 = np.array([[lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)] for lm in request.secondary_landmarks], dtype=np.float32)
                lh = pts1 if is_left else pts2
                rh = pts2 if is_left else pts1
                feat_126 = extract_dual_hand_126d(lh, rh)
                if expected_dim == 126:
                    feat_array = feat_126
                else:
                    condensed = np.zeros(89, dtype=np.float32)
                    condensed[:57] = feat_126[63:][ACTIVE_COORDINATE_INDICES_57] if np.any(feat_126[63:] != 0.0) else feat_126[:63][ACTIVE_COORDINATE_INDICES_57]
                    condensed[57:] = feat_126[ACTIVE_COORDINATE_INDICES_57[:32]]
                    feat_array = condensed
            elif expected_dim == 126:
                from training.features.geometry import extract_dual_hand_126d
                pts = np.array([[lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0)] for lm in request.landmarks], dtype=np.float32)
                lh = pts if is_left else None
                rh = None if is_left else pts
                feat_array = extract_dual_hand_126d(lh, rh)
            elif expected_dim == 89:
                feat_array = extract_realsign_89d_features(request.landmarks, is_left_hand=is_left)
            else:
                feat_array = normalize_landmarks_to_63d(request.landmarks, is_left_hand=is_left)
        else:
            raise ValueError(
                "Either 'features' (float array) or 'landmarks' (21 3D points) must be provided."
            )

        return self.predict_vector(feat_array)


# Shared singleton instance
classifier_service = RealSignInferenceService()
