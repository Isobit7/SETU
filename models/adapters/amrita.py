"""
SETU — Amrita SLR Continuous Dynamic Phrase Classifier Adapter.

Serves the trained 2-Layer Bidirectional PyTorch LSTM with Temporal Attention
for real-time 49-class Indian Sign Language phrase recognition from 316-D spatio-temporal features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("setu.models.amrita")

DEFAULT_AMRITA_MODEL_PATH = Path("models/weights/amrita_isl_phrase_bilstm.pth")


@dataclass(frozen=True)
class AmritaPrediction:
    """Represents a continuous 49-class phrase inference result."""
    phrase: str
    confidence: float
    is_known_sign: bool
    probabilities: Dict[str, float]
    top_3: List[Tuple[str, float]]
    top_5: List[Tuple[str, float]]
    raw_class_idx: int

    @property
    def label(self) -> str:
        return self.phrase


class AmritaPhraseBiLSTM(nn.Module):
    """2-Layer Bidirectional PyTorch LSTM with Temporal Attention."""

    def __init__(
        self,
        input_size: int = 316,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 49,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2

        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.classifier(context)
        return logits


class AmritaPhraseClassifier:
    """Inference adapter for Amrita SLR 49-class dynamic phrase recognition."""

    def __init__(self, weights_path: Path | str = DEFAULT_AMRITA_MODEL_PATH) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Amrita model weights not found at {self.weights_path.resolve()}")

        ckpt = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
        self.class_names: List[str] = ckpt.get("class_names", [])
        self.num_classes: int = len(self.class_names)
        arch = ckpt.get("architecture", {})

        self.model = AmritaPhraseBiLSTM(
            input_size=arch.get("input_dim", 316),
            hidden_size=arch.get("hidden_dim", 128),
            num_layers=arch.get("num_layers", 2),
            num_classes=self.num_classes,
            dropout=arch.get("dropout", 0.3),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.test_accuracy = ckpt.get("test_accuracy", None)
        self.top3_accuracy = ckpt.get("top3_accuracy", None)
        logger.info(
            "AmritaPhraseClassifier loaded from %s (Classes: %d, Accuracy: %s).",
            self.weights_path,
            self.num_classes,
            f"{self.test_accuracy * 100:.2f}%" if self.test_accuracy else "N/A",
        )

    @torch.inference_mode()
    def predict(
        self,
        features_sequence: np.ndarray,
        confidence_threshold: float = 0.40,
    ) -> AmritaPrediction:
        """
        Accepts a (T, 316) or (1, T, 316) numpy array and returns AmritaPrediction.
        """
        feat = np.asarray(features_sequence, dtype=np.float32)
        if feat.ndim == 2:
            feat = np.expand_dims(feat, axis=0)
        
        inp = torch.from_numpy(feat)
        logits = self.model(inp).numpy()[0]

        # Calibrated Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = (exp_l / np.sum(exp_l)).astype(np.float32)

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        best_phrase = self.class_names[best_idx]
        p_top1 = float(probs[best_idx])

        is_known = p_top1 >= confidence_threshold
        final_phrase = best_phrase if is_known else "Awaiting Sign Input"

        top_3 = [(self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices[:3]]
        top_5 = [(self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices[:5]]
        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

        return AmritaPrediction(
            phrase=final_phrase,
            confidence=p_top1 if is_known else 0.0,
            is_known_sign=is_known,
            probabilities=probs_dict,
            top_3=top_3,
            top_5=top_5,
            raw_class_idx=best_idx,
        )

    def predict_sequence(
        self,
        features_sequence: np.ndarray,
        confidence_threshold: float = 0.40,
    ) -> AmritaPrediction:
        """Alias for predict() to maintain interface parity with ISLSequenceLSTMClassifier."""
        return self.predict(features_sequence, confidence_threshold=confidence_threshold)

    def _empty_prediction(self) -> AmritaPrediction:
        """Returns baseline empty prediction structure when no motion or hands are detected."""
        return AmritaPrediction(
            phrase="Awaiting Sign Input",
            confidence=0.0,
            is_known_sign=False,
            probabilities={},
            top_3=[],
            top_5=[],
            raw_class_idx=-1,
        )

