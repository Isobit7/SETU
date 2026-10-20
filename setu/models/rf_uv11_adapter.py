"""
SETU - RF-UV-11 Pretrained Model Adapter
Provides a lightweight, zero-dependency NumPy inference engine for the 63-feature
single-hand gesture classification model from RF-UV-11 (Apache-2.0).

Contract:
- Input: Exactly 63 features (21 landmarks * 3 [x, y, z]) in normalized MediaPipe coordinates.
- Output: 26 alphabet classes ('A' through 'Z') with confidence scores.
- Temporal smoothing: Majority voting window for jitter-free real-time prediction.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List, Any
from collections import deque, Counter
import numpy as np
import h5py

from setu.exceptions import ModelArtifactError, LandmarkExtractionError, InferenceError

DEFAULT_PRETRAINED_PATH = Path("models/pretrained/rf_uv_11_model.h5")


class RFUV11Adapter:
    """
    Adapter for the RF-UV-11 pretrained sign language classification model.
    Directly evaluates weights using pure NumPy matrix operations to ensure sub-millisecond
    latency without requiring heavy TensorFlow/Keras runtime overhead.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        window_size: int = 5,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_PRETRAINED_PATH
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.labels: List[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.prediction_window: deque = deque(maxlen=window_size)
        self._is_loaded: bool = False

        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.W3: Optional[np.ndarray] = None
        self.b3: Optional[np.ndarray] = None

        self._load_weights()

    def _load_weights(self) -> None:
        """Loads and verifies HDF5 model weights."""
        if not self.model_path.exists():
            raise ModelArtifactError(
                f"RF-UV-11 pretrained model weights not found at: {self.model_path}. "
                "Ensure 'rf_uv_11_model.h5' exists in 'models/pretrained/'."
            )

        try:
            with h5py.File(str(self.model_path), "r") as f:
                if "model_weights" not in f:
                    raise ModelArtifactError(
                        f"Invalid HDF5 structure in {self.model_path}: missing 'model_weights' root."
                    )
                mw = f["model_weights"]
                # Layer 1: Dense 63 -> 128
                self.W1 = np.array(mw["dense"]["sequential"]["dense"]["kernel"], dtype=np.float32)
                self.b1 = np.array(mw["dense"]["sequential"]["dense"]["bias"], dtype=np.float32)
                # Layer 2: Dense 128 -> 64
                self.W2 = np.array(mw["dense_1"]["sequential"]["dense_1"]["kernel"], dtype=np.float32)
                self.b2 = np.array(mw["dense_1"]["sequential"]["dense_1"]["bias"], dtype=np.float32)
                # Layer 3: Dense 64 -> 26
                self.W3 = np.array(mw["dense_2"]["sequential"]["dense_2"]["kernel"], dtype=np.float32)
                self.b3 = np.array(mw["dense_2"]["sequential"]["dense_2"]["bias"], dtype=np.float32)

            # Validate parameter dimensions
            if self.W1.shape != (63, 128) or self.b1.shape != (128,):
                raise ModelArtifactError(f"Unexpected Layer 1 shape: {self.W1.shape}, bias: {self.b1.shape}")
            if self.W2.shape != (128, 64) or self.b2.shape != (64,):
                raise ModelArtifactError(f"Unexpected Layer 2 shape: {self.W2.shape}, bias: {self.b2.shape}")
            if self.W3.shape != (64, 26) or self.b3.shape != (26,):
                raise ModelArtifactError(f"Unexpected Layer 3 shape: {self.W3.shape}, bias: {self.b3.shape}")

            self._is_loaded = True
        except Exception as e:
            if isinstance(e, ModelArtifactError):
                raise
            raise ModelArtifactError(
                f"Failed to read HDF5 weights from {self.model_path}: {e}"
            ) from e

    @staticmethod
    def extract_63_features(hand_landmarks: Any) -> np.ndarray:
        """
        Converts 21 MediaPipe hand landmarks into the 63-feature flat vector
        expected by the RF-UV-11 model.

        Parameters:
            hand_landmarks: Iterable of 21 landmark objects with .x, .y, .z attributes.

        Returns:
            np.ndarray of shape (63,) with dtype float32.
        """
        if hand_landmarks is None:
            raise LandmarkExtractionError("Cannot extract 63 features: hand_landmarks is None.")

        if len(hand_landmarks) != 21:
            raise LandmarkExtractionError(
                f"Expected exactly 21 hand landmarks, got {len(hand_landmarks)}."
            )

        coords = []
        for lm in hand_landmarks:
            if not hasattr(lm, "x") or not hasattr(lm, "y") or not hasattr(lm, "z"):
                raise LandmarkExtractionError(
                    "Each landmark item must have 'x', 'y', and 'z' attributes."
                )
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])

        arr = np.array(coords, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise LandmarkExtractionError("Extracted landmarks contain non-finite (NaN or Inf) values.")

        return arr

    def predict_raw(
        self, landmarks_63: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Executes a single feedforward inference pass.

        Parameters:
            landmarks_63: 1D array of shape (63,).

        Returns:
            Tuple of:
            - predicted_label: str (e.g., 'A')
            - confidence: float in [0.0, 1.0]
            - top3: List of (label, confidence) sorted in descending order
        """
        if not self._is_loaded:
            raise InferenceError("Model weights are not loaded.")

        if not isinstance(landmarks_63, np.ndarray):
            landmarks_63 = np.asarray(landmarks_63, dtype=np.float32)

        if landmarks_63.shape != (63,):
            raise InferenceError(
                f"Invalid input shape {landmarks_63.shape}: RF-UV-11 model expects shape (63,)."
            )

        if not np.all(np.isfinite(landmarks_63)):
            raise InferenceError("Input features contain non-finite (NaN or Inf) values.")

        # Dense 1: 63 -> 128 (ReLU)
        h1 = np.maximum(0.0, np.dot(landmarks_63, self.W1) + self.b1)

        # Dense 2: 128 -> 64 (ReLU)
        h2 = np.maximum(0.0, np.dot(h1, self.W2) + self.b2)

        # Dense 3: 64 -> 26 (Softmax)
        logits = np.dot(h2, self.W3) + self.b3
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        pred_label = self.labels[best_idx]

        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [(self.labels[i], float(probs[i])) for i in top3_indices]

        return pred_label, confidence, top3

    def predict_smoothed(
        self, landmarks_63: np.ndarray
    ) -> Tuple[Optional[str], float, str, float]:
        """
        Executes inference with temporal majority-vote smoothing.

        Returns:
            Tuple of:
            - smoothed_label: str or None if no consensus above threshold
            - smoothed_confidence: float
            - raw_label: str (instantaneous frame prediction)
            - raw_confidence: float
        """
        raw_label, raw_conf, _ = self.predict_raw(landmarks_63)
        self.prediction_window.append((raw_label, raw_conf))

        valid_votes = [lbl for lbl, c in self.prediction_window if c >= self.confidence_threshold]
        if valid_votes:
            most_common_label, count = Counter(valid_votes).most_common(1)[0]
            # Require at least majority of recent valid votes
            if count >= max(1, len(valid_votes) // 2):
                matching_confs = [c for lbl, c in self.prediction_window if lbl == most_common_label]
                avg_conf = float(np.mean(matching_confs)) if matching_confs else raw_conf
                return most_common_label, avg_conf, raw_label, raw_conf

        return None, 0.0, raw_label, raw_conf

    def reset_window(self) -> None:
        """Clears the temporal smoothing window (called when hands leave the frame)."""
        self.prediction_window.clear()
