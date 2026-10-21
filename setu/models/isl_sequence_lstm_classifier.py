"""
SETU — Phase 4: Dynamic ISL Sequence LSTM Classifier Adapter.

Serves the trained 2-layer PyTorch LSTM sequence classifier on GPU (CUDA)
for real-time recognition of dynamic Indian Sign Language phrases and gestures
from 30-frame rolling buffers of 158-D Holistic landmark feature vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from setu.features.extractor import compute_velocity_features
from setu.features.schema import (
    FEATURE_DIM,
    EXPANDED_FEATURE_DIM,
    VELOCITY_FEATURE_DIM_316,
    VELOCITY_FEATURE_DIM_328,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LSTM_MODEL_PATH = Path("models/isl_sequence_lstm.pth")


@dataclass(frozen=True)
class ISLSequencePrediction:
    """Represents a discrete dynamic sequence sign inference result."""
    label: str
    confidence: float
    is_known_sign: bool
    probabilities: Dict[str, float]
    top_3: List[Tuple[str, float]]
    raw_class_idx: int


class ISLSequenceLSTM(nn.Module):
    """
    2-Layer Bidirectional PyTorch LSTM Sequence Classifier with Temporal Attention
    for 328-D ISL Landmark + Facial Ratios + Velocity Sequences.
    Input Shape: (Batch, 30, 328)
    Output Shape: (Batch, num_classes)
    """

    def __init__(
        self,
        input_size: int = 328,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Layer normalization to stabilize varying landmark scales and velocity magnitudes
        self.input_norm = nn.LayerNorm(input_size)

        # 2-layer Bidirectional LSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Temporal Attention Mechanism over the 30 time steps to weight salient gesture frames
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # Classifier head: Linear(256) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Linear(N_CLASSES)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Seq_len=30, Input_dim=316)
        Returns:
            logits: Tensor of shape (Batch, Num_classes=6)
        """
        x_norm = self.input_norm(x)
        # lstm_out: (Batch, Seq_len=30, Hidden_dim=256)
        lstm_out, _ = self.lstm(x_norm)

        # Compute temporal attention weights across the 30 time steps
        attn_scores = self.attention(lstm_out)  # (Batch, 30, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (Batch, 30, 1)

        # Salient feature aggregation via attention-weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (Batch, 256)

        logits = self.fc(context)
        return logits

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both logits and temporal attention weights for diagnostics/interpretability."""
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.fc(context)
        return logits, attn_weights.squeeze(-1)


class ISLSequenceLSTMClassifier:
    """
    Inference adapter for the PyTorch ISL Sequence LSTM model.
    Consumes a rolling buffer of 158-D feature frames (temporal sequence of length 30),
    runs GPU-accelerated forward inference, and outputs the top predicted sign,
    confidence score, and top-3 candidate distribution.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_SEQUENCE_LSTM_MODEL_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ISL Sequence LSTM model weights not found at {self.weights_path.resolve()}")

        # Auto-select CUDA if available
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)

        self.class_names: List[str] = checkpoint.get(
            "class_names", ["hello", "thank_you", "good_morning", "i", "you", "no_sign"]
        )
        self.num_classes: int = len(self.class_names)
        arch = checkpoint.get("model_architecture", {})
        self.input_size = arch.get("input_size", 328)
        self.hidden_size = arch.get("hidden_size", 128)
        self.num_layers = arch.get("num_layers", 2)
        self.dropout = arch.get("dropout", 0.3)
        self.bidirectional = arch.get("bidirectional", True)
        self.test_accuracy = checkpoint.get("test_accuracy", None)
        self.gpu_name = checkpoint.get("gpu_name", None)

        self.model = ISLSequenceLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "ISLSequenceLSTMClassifier initialized on %s (%s). Classes: %d (%s). Test Acc: %s",
            self.device,
            torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            self.num_classes,
            ", ".join(self.class_names),
            f"{self.test_accuracy * 100:.2f}%" if self.test_accuracy else "N/A",
        )

    def _empty_prediction(self) -> ISLSequencePrediction:
        """Returns a baseline no-sign prediction."""
        no_sign_idx = self.class_names.index("no_sign") if "no_sign" in self.class_names else 0
        probs = {c: 0.0 for c in self.class_names}
        probs[self.class_names[no_sign_idx]] = 1.0
        return ISLSequencePrediction(
            label="no_sign",
            confidence=1.0,
            is_known_sign=False,
            probabilities=probs,
            top_3=[("no_sign", 1.0)],
            raw_class_idx=no_sign_idx,
        )

    def warmup(self) -> None:
        """Executes a dummy forward pass with pre-allocated zero tensors to prime CUDA/CPU kernels."""
        dummy = np.zeros((30, self.input_size), dtype=np.float32)
        self.predict_sequence(dummy)

    @torch.inference_mode()
    def predict_sequence(
        self,
        features: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> ISLSequencePrediction:
        """
        Runs sequence inference over a temporal sequence of Holistic landmark feature vectors.
        Accepts either raw (T, 158) coordinates (which are auto-expanded to 316-D via velocity)
        or precomputed (T, 316) landmark+velocity feature vectors.

        Args:
            features: np.ndarray of shape (T, 158) or (T, 316).
            confidence_threshold: Minimum probability threshold for recognizing an active sign.

        Returns:
            ISLSequencePrediction with label, confidence, is_known_sign, probabilities, and top_3.
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D feature array of shape (T, 158), (T, 164), (T, 316), or (T, {self.input_size}), got {features.shape}"
            )

        if features.shape[1] in (158, 164, 316, 328):
            seq_features = compute_velocity_features(features, target_dim=self.input_size)
        else:
            raise ValueError(
                f"Expected feature shape (T, 158), (T, 164), (T, 316), or (T, {self.input_size}), got {features.shape}"
            )

        T = seq_features.shape[0]
        if T < 6:
            return self._empty_prediction()

        # Resample or pad to 15 frames if under 15; keep up to 30 frames
        if T < 15:
            pad = np.repeat(seq_features[-1:], 15 - T, axis=0)
            seq_proc = np.concatenate([seq_features, pad], axis=0)
        elif T > 30:
            seq_proc = seq_features[-30:]
        else:
            seq_proc = seq_features

        if not np.all(np.isfinite(seq_proc)):
            raise ValueError("Feature sequence contains NaN or Inf values")

        with torch.inference_mode():
            tensor = torch.from_numpy(seq_proc.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        best_label = self.class_names[best_idx]
        best_conf = float(probs[best_idx])

        is_known = (best_conf >= confidence_threshold) and (best_label != "no_sign")
        top_3 = [(self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices[:3]]
        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

        if best_label == "no_sign":
            final_label = "no_sign"
        elif is_known:
            final_label = best_label
        else:
            final_label = "unknown"

        return ISLSequencePrediction(
            label=final_label,
            confidence=best_conf,
            is_known_sign=is_known,
            probabilities=probs_dict,
            top_3=top_3,
            raw_class_idx=best_idx,
        )
