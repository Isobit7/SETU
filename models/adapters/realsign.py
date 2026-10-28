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

from training.features.geometry import normalize_hand_landmarks, evaluate_alphabet_heuristics

logger = logging.getLogger(__name__)

def _resolve_realsign_path() -> Path:
    p1 = Path("models/weights/realsign_26class_mlp.pth")
    if p1.exists():
        return p1
    return Path("models/realsign_26class_mlp.pth")

DEFAULT_REALSIGN_MODEL_PATH = _resolve_realsign_path()

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


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

    @property
    def label(self) -> str:
        return self.letter



class RealSignMLP(nn.Module):
    """PyTorch MLP Classifier for 26-class ISL Hand Alphabet Recognition."""

    def __init__(
        self,
        input_dim: int = 89,
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
        return self.net(x)


class RealSignClassifier:
    """
    Adapter and inference engine for RealSign 26-Class ISL Alphabet recognition.
    Consumes 21 3D hand landmarks (x, y, z), applies wrist-centering and palm span
    normalization, and classifies into letters A to Z using the PyTorch MLP.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.40

    def __init__(self, weights_path: Path | str = DEFAULT_REALSIGN_MODEL_PATH) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"RealSign model weights not found at {self.weights_path.resolve()}")

        checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self.class_names: List[str] = checkpoint.get(
            "classes", [chr(ord("A") + i) for i in range(26)]
        )
        self.num_classes: int = len(self.class_names)
        arch = checkpoint.get("architecture", {}) if isinstance(checkpoint, dict) else {}

        # Auto-detect batchnorm and input dimensions from weights
        first_layer_weight = state_dict.get("net.0.weight")
        inferred_in_dim = first_layer_weight.shape[1] if first_layer_weight is not None else 63
        has_bn = any("running_mean" in k for k in state_dict.keys())

        self.input_dim = arch.get("input_dim", inferred_in_dim)
        self.hidden1 = arch.get("hidden1", 128)
        self.hidden2 = arch.get("hidden2", 64)
        self.dropout = arch.get("dropout", 0.2)
        self.use_batchnorm = arch.get("use_batchnorm", has_bn)
        self.test_accuracy = checkpoint.get("test_accuracy", None) if isinstance(checkpoint, dict) else None
        self.macro_f1 = checkpoint.get("macro_f1", None) if isinstance(checkpoint, dict) else None

        self.model = RealSignMLP(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            num_classes=self.num_classes,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Exponential Moving Average (EMA) smoothing state: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
        self.ema_alpha = 0.40
        self._ema_probs: Optional[np.ndarray] = None
        self._last_mtime: float = self.weights_path.stat().st_mtime if self.weights_path.exists() else 0.0

        logger.info(
            "RealSignClassifier loaded from %s (Dim: %d, BN: %s). Supported classes: %d (%s..%s). Accuracy: %s",
            self.weights_path,
            self.input_dim,
            self.use_batchnorm,
            self.num_classes,
            self.class_names[0],
            self.class_names[-1],
            f"{self.test_accuracy * 100:.2f}%" if self.test_accuracy else "N/A",
        )

    def check_and_reload(self) -> None:
        """Hot-reloads updated model weights if checkpoint file changed on disk."""
        try:
            if not self.weights_path.exists():
                return
            current_mtime = self.weights_path.stat().st_mtime
            if current_mtime > self._last_mtime:
                logger.info("Detected newly trained RealSign weights at %s. Hot-reloading model in-memory...", self.weights_path)
                checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                arch = checkpoint.get("architecture", {}) if isinstance(checkpoint, dict) else {}
                first_layer_weight = state_dict.get("net.0.weight")
                inferred_in_dim = first_layer_weight.shape[1] if first_layer_weight is not None else 63
                has_bn = any("running_mean" in k for k in state_dict.keys())

                self.input_dim = arch.get("input_dim", inferred_in_dim)
                self.hidden1 = arch.get("hidden1", 128)
                self.hidden2 = arch.get("hidden2", 64)
                self.dropout = arch.get("dropout", 0.2)
                self.use_batchnorm = arch.get("use_batchnorm", has_bn)
                self.class_names = checkpoint.get("classes", [chr(ord("A") + i) for i in range(26)]) if isinstance(checkpoint, dict) else [chr(ord("A") + i) for i in range(26)]
                self.num_classes = len(self.class_names)
                self.test_accuracy = checkpoint.get("test_accuracy", None) if isinstance(checkpoint, dict) else None
                self.macro_f1 = checkpoint.get("macro_f1", None) if isinstance(checkpoint, dict) else None

                self.model = RealSignMLP(
                    input_dim=self.input_dim,
                    hidden1=self.hidden1,
                    hidden2=self.hidden2,
                    num_classes=self.num_classes,
                    dropout=self.dropout,
                    use_batchnorm=self.use_batchnorm,
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self._last_mtime = current_mtime
                self.reset_smoothing()
                logger.info("RealSign model successfully hot-reloaded! (Dim: %d, Accuracy: %s)", self.input_dim, f"{self.test_accuracy*100:.2f}%" if self.test_accuracy else "N/A")
        except Exception as e:
            logger.warning("Error checking/reloading RealSign weights: %s", e)

    def reset_smoothing(self) -> None:
        """Resets the EMA smoothed probability filter state."""
        self._ema_probs = None

    def predict(
        self,
        features_63d: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        apply_smoothing: bool = False,
        apply_heuristic_boost: bool = False,
        raw_landmarks_21x3: Optional[np.ndarray] = None,
    ) -> RealSignPrediction:
        """
        Accepts a 63-D, 89-D, or 126-D feature vector, or a (21, 3) raw landmark matrix directly and runs inference.
        Returns RealSignPrediction with raw class logits and softmax probabilities across 26 letters (A–Z).
        """
        features_arr = np.asarray(features_63d, dtype=np.float32)
        if features_arr.ndim == 2 and features_arr.shape == (21, 3):
            return self.predict_landmarks(
                features_arr,
                confidence_threshold=confidence_threshold,
                apply_smoothing=apply_smoothing,
                apply_heuristic_boost=apply_heuristic_boost,
            )

        return self.predict_vector(
            features_arr,
            confidence_threshold=confidence_threshold,
            apply_smoothing=apply_smoothing,
            apply_heuristic_boost=apply_heuristic_boost,
            raw_landmarks_21x3=raw_landmarks_21x3,
        )

    @torch.inference_mode()
    def predict_vector(
        self,
        features_63d: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        apply_smoothing: bool = False,
        apply_heuristic_boost: bool = False,
        raw_landmarks_21x3: Optional[np.ndarray] = None,
    ) -> RealSignPrediction:
        """
        Run inference on a single 63-D normalized feature vector or (21, 3) landmark matrix.
        Uses @torch.inference_mode() to prevent autograd graph memory leaks
        and applies EMA probability smoothing to eliminate 1-frame jitter flickering.
        By default, apply_heuristic_boost is False, trusting the pure trained neural network.
        """
        self.check_and_reload()

        original_63d: Optional[np.ndarray] = None
        feat_vec = np.asarray(features_63d, dtype=np.float32)

        if feat_vec.ndim == 2 and feat_vec.shape == (21, 3):
            norm_pts, _ = normalize_wrist_coordinates(feat_vec)
            feat_vec = norm_pts.ravel().astype(np.float32)

        if feat_vec.ndim == 1 and feat_vec.shape[0] == 126:
            # Dual-hand 126-D input
            if self.input_dim == 89 or self.input_dim == 63:
                # Use primary active hand (right hand slots 63:126 or left hand slots 0:63)
                rh_part = feat_vec[63:126]
                lh_part = feat_vec[0:63]
                primary_hand = rh_part if float(np.linalg.norm(rh_part)) > 1e-3 else lh_part
                feat_vec = primary_hand
                original_63d = feat_vec.copy()
                if self.input_dim == 89:
                    from training.features.geometry import expand_63d_to_89d
                    feat_vec = expand_63d_to_89d(feat_vec)
        elif feat_vec.ndim == 1 and feat_vec.shape[0] == 63:
            original_63d = feat_vec.copy()
            if self.input_dim == 89:
                from training.features.geometry import expand_63d_to_89d
                feat_vec = expand_63d_to_89d(feat_vec)
            elif self.input_dim == 126:
                # Pad left hand with zeros when given single right hand
                feat_vec = np.concatenate([np.zeros(63, dtype=np.float32), feat_vec])

        if feat_vec.ndim != 1 or feat_vec.shape[0] != self.input_dim:
            raise ValueError(f"Expected feature vector shape ({self.input_dim},), got {feat_vec.shape}")

        if not np.all(np.isfinite(feat_vec)):
            raise ValueError("Feature vector contains NaN or Inf values")

        # Guard against zero or degenerate landmark vectors (e.g. absent or uninitialized hand)
        raw_energy = float(np.linalg.norm(features_63d))
        total_energy = float(np.linalg.norm(feat_vec))
        if confidence_threshold > 0.0 and (raw_energy < 1e-3 or total_energy < 1e-3):
            return RealSignPrediction(
                letter="Awaiting Sign Input",
                confidence=0.0,
                is_known_sign=False,
                probabilities={name: 0.0 for name in self.class_names},
                top_3=[("Awaiting Sign Input", 0.0)],
                raw_class_idx=-1,
                logits={name: 0.0 for name in self.class_names},
            )

        with torch.inference_mode():
            inp_tensor = torch.from_numpy(feat_vec.astype(np.float32)).unsqueeze(0)
            logits_tensor = self.model(inp_tensor)
            logits_arr = logits_tensor.numpy()[0].copy()

        # Optional geometric rule validation (only if explicitly enabled; disabled by default to avoid hijacking)
        if apply_heuristic_boost:
            boosted_letter: Optional[str] = None
            try:
                pts_to_eval = None
                if raw_landmarks_21x3 is not None:
                    pts_to_eval = raw_landmarks_21x3
                elif original_63d is not None:
                    pts_to_eval = original_63d.reshape(21, 3)
                else:
                    from training.features.geometry import ACTIVE_COORDINATE_INDICES_57
                    recon_63 = np.zeros(63, dtype=np.float32)
                    recon_63[ACTIVE_COORDINATE_INDICES_57] = feat_vec[:57]
                    recon_63[28] = -1.0
                    pts_to_eval = recon_63.reshape(21, 3)

                if pts_to_eval is not None:
                    boosted_letter = evaluate_alphabet_heuristics(pts_to_eval)
            except Exception as e:
                logger.debug("Failed evaluating alphabet heuristics: %s", e)

            if boosted_letter and boosted_letter in self.class_names:
                target_idx = self.class_names.index(boosted_letter)
                other_mask = np.ones(self.num_classes, dtype=bool)
                other_mask[target_idx] = False
                other_logits = logits_arr[other_mask]
                max_other = float(np.max(other_logits))
                logsumexp_other = float(max_other + np.log(np.sum(np.exp(other_logits - max_other))))
                logits_arr[target_idx] = max(float(logits_arr[target_idx]), logsumexp_other + 2.0)

        # Calibrated Softmax with Temperature Scaling (T=0.75) for high-confidence discriminability
        temperature = 0.75
        scaled_logits = (logits_arr - np.max(logits_arr)) / temperature
        exp_logits = np.exp(scaled_logits)
        raw_probs = (exp_logits / np.sum(exp_logits)).astype(np.float32)

        # Exponential Moving Average (EMA) Smoothing: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
        if apply_smoothing:
            if self._ema_probs is None or self._ema_probs.shape != raw_probs.shape:
                self._ema_probs = raw_probs.copy()
            else:
                self._ema_probs = (self.ema_alpha * raw_probs + (1.0 - self.ema_alpha) * self._ema_probs).astype(np.float32)
            probs = self._ema_probs
        else:
            probs = raw_probs

        sorted_indices = np.argsort(probs)[::-1]
        best_idx = int(sorted_indices[0])
        best_letter = self.class_names[best_idx]
        p_top1 = float(probs[best_idx])
        p_top2 = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = p_top1 - p_top2

        # Confidence & Margin Gate:
        # Respect user/client-configured confidence_threshold without forcing a 0.50 clamp,
        # with a gentle 0.05 margin requirement to filter true noise while retaining responsiveness.
        effective_min_conf = float(confidence_threshold) if confidence_threshold > 0.0 else 0.0
        effective_min_margin = 0.05 if confidence_threshold > 0.0 else 0.0

        if p_top1 < effective_min_conf or margin < effective_min_margin:
            is_known = False
            final_letter = "Awaiting Sign Input"
            final_conf = 0.0
        else:
            is_known = True
            final_letter = best_letter
            final_conf = p_top1

        top_3 = [(self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices[:3]]
        probs_dict = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        logits_dict = {name: float(logits_arr[i]) for i, name in enumerate(self.class_names)}

        return RealSignPrediction(
            letter=final_letter,
            confidence=final_conf,
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
        apply_smoothing: bool = False,
        apply_heuristic_boost: bool = False,
    ) -> RealSignPrediction:
        """
        Preprocess 21 3D landmarks (wrist-centering and palm scale) and run inference.
        """
        if landmarks_21x3 is None or len(landmarks_21x3) != 21:
            return self._empty_prediction()

        pts = np.asarray(landmarks_21x3, dtype=np.float32)
        if pts.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {pts.shape}")

        if self.input_dim == 89:
            from training.features.geometry import extract_realsign_89d_features
            feat_vec = extract_realsign_89d_features(pts)
        else:
            norm_pts, _ = normalize_hand_landmarks(pts)
            feat_vec = norm_pts.flatten()

        return self.predict_vector(
            feat_vec,
            confidence_threshold=confidence_threshold,
            apply_smoothing=apply_smoothing,
            apply_heuristic_boost=apply_heuristic_boost,
            raw_landmarks_21x3=pts,
        )

    def predict_hand_points(
        self,
        raw_landmarks: Any,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        apply_smoothing: bool = False,
        apply_heuristic_boost: bool = False,
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
        return self.predict_landmarks(
            pts,
            confidence_threshold=confidence_threshold,
            apply_smoothing=apply_smoothing,
            apply_heuristic_boost=apply_heuristic_boost,
        )

    def _empty_prediction(self) -> RealSignPrediction:
        self._ema_probs = None
        probs = {c: 0.0 for c in self.class_names}
        probs["no_sign"] = 1.0
        logits = {c: 0.0 for c in self.class_names}
        return RealSignPrediction(
            letter="Awaiting Sign Input",
            confidence=0.0,
            is_known_sign=False,
            probabilities=probs,
            top_3=[("no_sign", 1.0)],
            raw_class_idx=-1,
            logits=logits,
        )
