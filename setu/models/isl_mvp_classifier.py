"""
Inference classifier for SETU ISL MVP 5-sign (+ unknown/no-sign) recognition.

Consumes either:
1. A temporal sequence of 158-D features (shape: (T, 158), where T >= 10, typically 30).
2. Live frames / rolling landmarks, returning sign class, confidence %, and unknown/no-sign state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from setu.features.schema import LEFT_PRESENCE_INDEX, RIGHT_PRESENCE_INDEX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ISLPrediction:
    """Represents a discrete sign recognition inference result."""
    label: str
    confidence: float
    is_known_sign: bool
    probabilities: Dict[str, float]
    raw_class_idx: int


def featurize_sequence_1264(feat_seq: np.ndarray) -> np.ndarray:
    """
    Featurize a (T, 158) sequence into a 1,264-D representation.
    Pools mean, std, min, max, delta, early, mid, late phases.
    """
    assert feat_seq.ndim == 2 and feat_seq.shape[1] == 158, f"Expected (T, 158), got {feat_seq.shape}"
    T = feat_seq.shape[0]

    mean_f = np.mean(feat_seq, axis=0)
    std_f = np.std(feat_seq, axis=0)
    min_f = np.min(feat_seq, axis=0)
    max_f = np.max(feat_seq, axis=0)
    delta_f = feat_seq[-1] - feat_seq[0]

    t1 = max(1, T // 3)
    t2 = max(2, 2 * T // 3)
    early_f = np.mean(feat_seq[:t1], axis=0)
    mid_f = np.mean(feat_seq[t1:t2], axis=0)
    late_f = np.mean(feat_seq[t2:], axis=0)

    pooled = np.concatenate([
        mean_f, std_f, min_f, max_f, delta_f, early_f, mid_f, late_f
    ], axis=0).astype(np.float32)

    return pooled


class ISLMVPClassifier:
    """
    Pure NumPy inference engine for SETU ISL 5-Sign MVP.
    Provides fast, deterministic execution with no deep learning dependencies.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.50
    HAND_PRESENCE_THRESHOLD = 0.05

    def __init__(self, weights_path: Path | str):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ISL MVP model weights not found at {self.weights_path}")

        data = np.load(self.weights_path, allow_pickle=True)
        self.weights: np.ndarray = data["weights"]  # (1264, num_classes)
        self.bias: np.ndarray = data["bias"]  # (num_classes,)
        self.mean: np.ndarray = data["mean"]  # (1264,)
        self.scale: np.ndarray = data["scale"]  # (1264,)
        self.class_names: List[str] = [str(c) for c in data["class_names"]]
        self.num_classes: int = int(data["num_classes"])

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def predict_features(
        self,
        features: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> ISLPrediction:
        """
        Run inference on a sequence of 158-D features.
        
        Args:
            features: np.ndarray of shape (T, 158) or 1D pooled features (1264,)
            confidence_threshold: Minimum probability required to trigger active sign
        """
        # Check if hands were present in the sequence using schema presence flags (146, 147)
        if features.ndim == 2 and features.shape[1] == 158:
            right_pres = np.mean(features[:, RIGHT_PRESENCE_INDEX])
            left_pres = np.mean(features[:, LEFT_PRESENCE_INDEX])
            total_hand_presence = right_pres + left_pres

            # If no hands are visible across the entire window, instantly classify as no_sign
            if total_hand_presence < self.HAND_PRESENCE_THRESHOLD:
                probs_dict = {c: 0.0 for c in self.class_names}
                probs_dict["no_sign"] = 1.0
                return ISLPrediction(
                    label="no_sign",
                    confidence=1.0,
                    is_known_sign=False,
                    probabilities=probs_dict,
                    raw_class_idx=self.class_names.index("no_sign"),
                )

            pooled = featurize_sequence_1264(features)
        elif features.ndim == 1 and features.shape[0] == 1264:
            pooled = features
        else:
            raise ValueError(f"Expected features shape (T, 158) or (1264,), got {features.shape}")

        # Normalize and compute logits
        normed = (pooled - self.mean) / self.scale
        logits = normed @ self.weights + self.bias
        probs = self._softmax(logits)

        best_idx = int(np.argmax(probs))
        best_label = self.class_names[best_idx]
        best_conf = float(probs[best_idx])

        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

        # Active sign check:
        # 1. Must not be explicitly classified as 'no_sign'
        # 2. Must meet confidence threshold
        is_known = (best_label != "no_sign") and (best_conf >= confidence_threshold)

        # If below threshold, fall back to "unknown / no_sign"
        final_label = best_label if is_known else ("no_sign" if best_label == "no_sign" else "unknown")

        return ISLPrediction(
            label=final_label,
            confidence=best_conf,
            is_known_sign=is_known,
            probabilities=probs_dict,
            raw_class_idx=best_idx,
        )
