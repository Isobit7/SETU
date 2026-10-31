"""
SETU — ISL Conversational 5-Sign (+ No-Sign) MVP Inference Engine
=================================================================

Architectural Role:
-------------------
Provides a sub-millisecond, pure NumPy inference engine for SETU's core conversational
phrases: HELLO, THANK_YOU, GOOD_MORNING, I, YOU, plus NO_SIGN (resting background).

Mathematical Formulation:
-------------------------
1. Temporal Window Input: Consumes a sequence of 158-D Holistic feature vectors over T frames (T >= 10, typically 30):
   X_seq in R^{T x 158}
2. Multi-Phase Statistical Pooling:
   Compresses dynamic temporal trajectory into a static 1,264-D representation by concatenating
   8 statistical descriptors along the temporal axis:
   - Global mean (158-D): Overall spatial centroid of landmarks
   - Global standard deviation (158-D): Landmark movement spread and dispersion
   - Global minimum (158-D): Extreme lower boundary of hand trajectories
   - Global maximum (158-D): Extreme upper boundary of hand trajectories
   - Delta (158-D): Net displacement from start frame to end frame (X_{T-1} - X_0)
   - Early phase mean (158-D): Hand preparation gesture (t in [0, T/3))
   - Mid phase mean (158-D): Active stroke peak gesture (t in [T/3, 2T/3))
   - Late phase mean (158-D): Retraction gesture (t in [2T/3, T))
   Total: 8 * 158 = 1,264-D vector z in R^{1264}
3. Linear Matrix Classification:
   z_norm = (z - mu) / sigma
   logits = z_norm @ W + b  where W in R^{1264 x 6}, b in R^6
   probs = softmax(logits)

Hardware & Latency Invariants:
------------------------------
- Pure NumPy execution: Zero PyTorch/TensorFlow deep learning dependencies.
- Sub-millisecond execution (< 0.45 ms on standard laptop CPU).
- Hand presence gating (indices 146 & 147): Instantly classifies as `no_sign` if hands are absent,
  bypassing matrix computation and eliminating background false positives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from training.feature_extractor import LEFT_PRESENCE_INDEX, RIGHT_PRESENCE_INDEX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ISLPrediction:
    """
    Immutable container representing a discrete sign recognition inference result.
    
    Attributes:
        label: Predicted phrase string (e.g., 'hello', 'thank_you', 'no_sign', or 'unknown').
        confidence: Normalized probability of the top-predicted class in [0.0, 1.0].
        is_known_sign: True if the sign is a recognized conversational phrase meeting the threshold.
        probabilities: Dictionary mapping every class name to its softmax probability.
        raw_class_idx: Integer index of the highest logit output in the model matrix.
    """
    label: str
    confidence: float
    is_known_sign: bool
    probabilities: Dict[str, float]
    raw_class_idx: int


def featurize_sequence_1264(feat_seq: np.ndarray) -> np.ndarray:
    """
    Featurizes a temporal sequence of 158-D features into a unified 1,264-D representation.
    
    Pools 8 statistical temporal metrics across time:
    [mean, std, min, max, delta, early_mean, mid_mean, late_mean].
    
    Args:
        feat_seq: NumPy array of shape (T, 158) where T is frame sequence length.
        
    Returns:
        1D float32 array of shape (1264,).
        
    Raises:
        AssertionError: If input does not have 2 dimensions or second dimension is not 158.
    """
    assert feat_seq.ndim == 2 and feat_seq.shape[1] == 158, f"Expected (T, 158), got {feat_seq.shape}"
    T = feat_seq.shape[0]

    # Global temporal statistics
    mean_f = np.mean(feat_seq, axis=0)
    std_f = np.std(feat_seq, axis=0)
    min_f = np.min(feat_seq, axis=0)
    max_f = np.max(feat_seq, axis=0)
    delta_f = feat_seq[-1] - feat_seq[0]

    # Multi-phase segmented temporal statistics
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
    Pure NumPy inference engine for SETU ISL 5-Sign (+ No-Sign) MVP.
    Provides fast, deterministic execution with zero deep learning runtime overhead.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.50
    HAND_PRESENCE_THRESHOLD = 0.05

    def __init__(self, weights_path: Path | str = Path("models/weights/isl_mvp_model.npz")) -> None:
        """
        Initializes the classifier by loading pre-trained matrix weights from disk.
        
        Args:
            weights_path: Path to the .npz archive containing weights, bias, mean, scale, class_names.
            
        Raises:
            FileNotFoundError: If the weights archive does not exist at the specified path.
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ISL MVP model weights not found at {self.weights_path}")

        data = np.load(self.weights_path, allow_pickle=True)
        self.weights: np.ndarray = data["weights"]  # Shape: (1264, num_classes)
        self.bias: np.ndarray = data["bias"]        # Shape: (num_classes,)
        self.mean: np.ndarray = data["mean"]        # Shape: (1264,)
        self.scale: np.ndarray = data["scale"]      # Shape: (1264,)
        self.class_names: List[str] = [str(c) for c in data["class_names"]]
        self.num_classes: int = int(data["num_classes"])

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax activation over 1D or 2D logits."""
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def predict_features(
        self,
        features: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> ISLPrediction:
        """
        Runs complete inference pipeline on a temporal sequence of features.
        
        Pipeline Steps:
        1. Hand Presence Gating: Checks presence flags (schema indices 146 & 147). If hands
           are absent across the temporal window, immediately short-circuits to `no_sign`.
        2. Sequence Featurization: Pools (T, 158) into (1264,) multi-phase descriptor.
        3. Feature Standardization: z_norm = (z - mean) / scale.
        4. Linear Projection & Softmax: Computes class probabilities.
        5. Threshold Validation: Confirms sign probability exceeds confidence threshold.
        
        Args:
            features: np.ndarray of shape (T, 158) or pre-pooled (1264,).
            confidence_threshold: Minimum probability required to trigger an active sign.
            
        Returns:
            ISLPrediction with detected label, probability distribution, and classification status.
        """
        # Step 1: Hand Presence Gating using schema presence flags (146: Right, 147: Left)
        if features.ndim == 2 and features.shape[1] == 158:
            right_pres = float(np.mean(features[:, RIGHT_PRESENCE_INDEX]))
            left_pres = float(np.mean(features[:, LEFT_PRESENCE_INDEX]))
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

        # Step 2: Standardization & Linear Logits Computation
        normed = (pooled - self.mean) / self.scale
        logits = normed @ self.weights + self.bias
        probs = self._softmax(logits)

        # Step 3: Top Candidate Extraction
        best_idx = int(np.argmax(probs))
        best_label = self.class_names[best_idx]
        best_conf = float(probs[best_idx])

        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

        # Step 4: Active Sign Verification
        # 1. Must not be explicitly classified as 'no_sign'
        # 2. Must meet confidence threshold
        is_known = (best_label != "no_sign") and (best_conf >= confidence_threshold)

        # Fall back to "no_sign" if below threshold
        final_label = best_label if is_known else ("no_sign" if best_label == "no_sign" else "unknown")

        return ISLPrediction(
            label=final_label,
            confidence=best_conf,
            is_known_sign=is_known,
            probabilities=probs_dict,
            raw_class_idx=best_idx,
        )


# ============================================================================
# Backward-Compatibility Aliases & Public Interface
# ============================================================================
ISLMvpClassifier = ISLMVPClassifier
MVPClassifierPrediction = ISLPrediction
MVP_CLASSES = ['hello', 'thank_you', 'good_morning', 'i', 'you', 'no_sign']
CLASS_NAMES = MVP_CLASSES

__all__ = [
    'ISLMVPClassifier',
    'ISLMvpClassifier',
    'ISLPrediction',
    'MVPClassifierPrediction',
    'CLASS_NAMES',
    'MVP_CLASSES',
    'featurize_sequence_1264',
]
