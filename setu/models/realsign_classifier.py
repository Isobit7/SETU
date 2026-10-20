"""
SETU — RealSign 26-Class PyTorch Alphabet Classifier Adapter.

Serves the trained PyTorch MLP model for real-time 26-class Indian Sign Language
alphabet recognition (A–Z) from 21 3D hand landmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from setu.utils.geometry import normalize_hand_landmarks

logger = logging.getLogger(__name__)

DEFAULT_REALSIGN_MODEL_PATH = Path("models/realsign_26class_mlp.pth")


@dataclass(frozen=True)
class RealSignPrediction:
    """Represents a discrete 26-class alphabet sign inference result."""
    letter: str
    confidence: float
    is_known_sign: bool
    probabilities: Dict[str, float]
    top_3: List[Tuple[str, float]]
    raw_class_idx: int
    logits: Optional[Dict[str, float]] = None


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


class RealSignClassifier:
    """
    Adapter and inference engine for RealSign 26-Class ISL Alphabet recognition.
    Consumes 21 3D hand landmarks (x, y, z), applies wrist-centering and palm span
    normalization, and classifies into letters A to Z using the PyTorch MLP.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.50

    def __init__(self, weights_path: Path | str = DEFAULT_REALSIGN_MODEL_PATH) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"RealSign model weights not found at {self.weights_path.resolve()}")

        checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)

        self.class_names: List[str] = checkpoint.get(
            "classes", [chr(ord("A") + i) for i in range(26)]
        )
        self.num_classes: int = len(self.class_names)
        arch = checkpoint.get("architecture", {})
        self.input_dim = arch.get("input_dim", 63)
        self.hidden1 = arch.get("hidden1", 128)
        self.hidden2 = arch.get("hidden2", 64)
        self.dropout = arch.get("dropout", 0.2)
        self.test_accuracy = checkpoint.get("test_accuracy", None)
        self.macro_f1 = checkpoint.get("macro_f1", None)

        self.model = RealSignMLP(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(
            "RealSignClassifier loaded from %s. Supported classes: %d (%s..%s). Accuracy: %s",
            self.weights_path,
            self.num_classes,
            self.class_names[0],
            self.class_names[-1],
            f"{self.test_accuracy * 100:.2f}%" if self.test_accuracy else "N/A",
        )

    def predict(
        self,
        features_63d: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> RealSignPrediction:
        """
        Accepts the 63-D normalized hand vector directly and runs inference.
        Returns RealSignPrediction with raw class logits and softmax probabilities across 26 letters (A–Z).
        """
        return self.predict_vector(features_63d, confidence_threshold=confidence_threshold)

    def predict_vector(
        self,
        features_63d: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> RealSignPrediction:
        """
        Run inference on a single 63-D normalized feature vector.
        """
        if features_63d.ndim != 1 or features_63d.shape[0] != self.input_dim:
            raise ValueError(f"Expected feature vector shape ({self.input_dim},), got {features_63d.shape}")

        if not np.all(np.isfinite(features_63d)):
            raise ValueError("Feature vector contains NaN or Inf values")

        with torch.no_grad():
            inp_tensor = torch.from_numpy(features_63d.astype(np.float32)).unsqueeze(0)
            logits_tensor = self.model(inp_tensor)
            logits_arr = logits_tensor.numpy()[0]
            probs = torch.softmax(logits_tensor, dim=1).numpy()[0]

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        best_letter = self.class_names[best_idx]
        best_conf = float(probs[best_idx])

        is_known = best_conf >= confidence_threshold
        top_3 = [(self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices[:3]]
        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        logits_dict = {name: float(logits_arr[i]) for i, name in enumerate(self.class_names)}

        final_letter = best_letter if is_known else "unknown"

        return RealSignPrediction(
            letter=final_letter,
            confidence=best_conf,
            is_known_sign=is_known,
            probabilities=probs_dict,
            top_3=top_3,
            raw_class_idx=best_idx,
            logits=logits_dict,
        )

    def predict_landmarks(
        self,
        landmarks_21x3: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> RealSignPrediction:
        """
        Preprocess 21 3D landmarks (wrist-centering and palm scale) and run inference.
        """
        if landmarks_21x3 is None or len(landmarks_21x3) != 21:
            return self._empty_prediction()

        pts = np.asarray(landmarks_21x3, dtype=np.float32)
        if pts.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {pts.shape}")

        norm_pts, _ = normalize_hand_landmarks(pts)
        feat_vec = norm_pts.flatten()
        return self.predict_vector(feat_vec, confidence_threshold=confidence_threshold)

    def predict_hand_points(
        self,
        raw_landmarks: Any,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> RealSignPrediction:
        """
        Helper accepting MediaPipe NormalizedLandmark objects or serialized dicts.
        """
        if not raw_landmarks or len(raw_landmarks) < 21:
            return self._empty_prediction()

        first_list = raw_landmarks[0] if isinstance(raw_landmarks[0], (list, tuple)) else raw_landmarks
        if len(first_list) < 21:
            return self._empty_prediction()

        coords: List[List[float]] = []
        for lm in first_list[:21]:
            if hasattr(lm, "x") and hasattr(lm, "y") and hasattr(lm, "z"):
                coords.append([float(lm.x), float(lm.y), float(lm.z)])
            elif isinstance(lm, dict) and "x" in lm and "y" in lm:
                coords.append([float(lm["x"]), float(lm["y"]), float(lm.get("z", 0.0))])
            elif isinstance(lm, (list, tuple)) and len(lm) >= 3:
                coords.append([float(lm[0]), float(lm[1]), float(lm[2])])
            else:
                return self._empty_prediction()

        pts = np.array(coords, dtype=np.float32)
        return self.predict_landmarks(pts, confidence_threshold=confidence_threshold)

    def _empty_prediction(self) -> RealSignPrediction:
        probs = {c: 0.0 for c in self.class_names}
        probs["no_sign"] = 1.0
        logits = {c: 0.0 for c in self.class_names}
        return RealSignPrediction(
            letter="no_sign",
            confidence=1.0,
            is_known_sign=False,
            probabilities=probs,
            top_3=[("no_sign", 1.0)],
            raw_class_idx=-1,
            logits=logits,
        )
