"""AI4Bharat INCLUDE Pretrained ISL Recognition Service.

Implements the official AI4Bharat INCLUDE 5-layer bidirectional LSTM architecture
(263 classes, 134-D keypoints input) and inference service for SETU.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.services.vision.landmarks import HolisticLandmarkerWrapper

logger = logging.getLogger("setu.models.include")


@dataclass
class INCLUDELstmConfig:
    input_size: int = 134
    hidden_size: int = 256
    num_layers: int = 5
    batch_first: bool = True
    bidirectional: bool = True
    dropout: float = 0.2


class INCLUDELstmModel(nn.Module):
    """Official AI4Bharat INCLUDE LSTM classifier."""

    def __init__(self, config: INCLUDELstmConfig = INCLUDELstmConfig(), n_classes: int = 263) -> None:
        super().__init__()
        self.config = config
        self.n_classes = n_classes
        self.lstm = nn.LSTM(**asdict(config))
        in_features = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching AI4Bharat models/lstm.py:
        x: (batch_size, seq_len, 134)
        Returns: logits (batch_size, n_classes)
        """
        out, _ = self.lstm(x)
        out = torch.max(out, dim=1).values
        if self.training:
            out = F.dropout(out, p=0.3)
        return self.l1(out)


def interp_channel(arr: np.ndarray, scale: float) -> np.ndarray:
    """Linearly interpolates missing landmark points (NaNs) across time frames, matching AI4Bharat dataset.py."""
    out = np.zeros_like(arr)
    n_frames, n_landmarks = arr.shape
    all_idx = np.arange(n_frames)
    for col in range(n_landmarks):
        col_vals = arr[:, col]
        nans = np.isnan(col_vals)
        if np.all(nans):
            out[:, col] = 0.0
        elif not np.any(nans):
            out[:, col] = col_vals * scale
        else:
            valid_idx = np.where(~nans)[0]
            interpolated = np.interp(all_idx, valid_idx, col_vals[valid_idx])
            out[:, col] = interpolated * scale
    return out


def extract_include_134_features(
    video_path: Path,
    landmarker: HolisticLandmarkerWrapper,
    max_frame_len: int = 200,
    target_width: float = 1920.0,
    target_height: float = 1080.0,
) -> np.ndarray:
    """Extracts 134-D keypoints from video frames matching AI4Bharat INCLUDE specifications:
    - 25 upper-body pose landmarks (x, y) = 50 features
    - 21 left hand landmarks (x, y) = 42 features
    - 21 right hand landmarks (x, y) = 42 features
    Total: 134 features per frame.
    Scaled to (target_width, target_height) pixel space and linearly interpolated across time frames.
    Padded or truncated to max_frame_len frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    pose_x_list: List[List[float]] = []
    pose_y_list: List[List[float]] = []
    hand1_x_list: List[List[float]] = []
    hand1_y_list: List[List[float]] = []
    hand2_x_list: List[List[float]] = []
    hand2_y_list: List[List[float]] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hl_result = landmarker.detect(frame)

        # 1. Pose: Upper 25 landmarks (0 to 24)
        if hl_result.pose_landmarks and len(hl_result.pose_landmarks) > 0:
            first_pose = hl_result.pose_landmarks[0] if isinstance(hl_result.pose_landmarks[0], list) else hl_result.pose_landmarks
            px = [float(lm.x) for lm in first_pose[:25]]
            py = [float(lm.y) for lm in first_pose[:25]]
            if len(px) < 25:
                px.extend([np.nan] * (25 - len(px)))
                py.extend([np.nan] * (25 - len(py)))
        else:
            px = [np.nan] * 25
            py = [np.nan] * 25
        pose_x_list.append(px)
        pose_y_list.append(py)

        # 2. Hand 1 (Left hand): 21 landmarks
        if hl_result.left_hand_landmarks and len(hl_result.left_hand_landmarks) > 0:
            first_lh = hl_result.left_hand_landmarks[0] if isinstance(hl_result.left_hand_landmarks[0], list) else hl_result.left_hand_landmarks
            h1x = [float(lm.x) for lm in first_lh[:21]]
            h1y = [float(lm.y) for lm in first_lh[:21]]
            if len(h1x) < 21:
                h1x.extend([np.nan] * (21 - len(h1x)))
                h1y.extend([np.nan] * (21 - len(h1y)))
        else:
            h1x = [np.nan] * 21
            h1y = [np.nan] * 21
        hand1_x_list.append(h1x)
        hand1_y_list.append(h1y)

        # 3. Hand 2 (Right hand): 21 landmarks
        if hl_result.right_hand_landmarks and len(hl_result.right_hand_landmarks) > 0:
            first_rh = hl_result.right_hand_landmarks[0] if isinstance(hl_result.right_hand_landmarks[0], list) else hl_result.right_hand_landmarks
            h2x = [float(lm.x) for lm in first_rh[:21]]
            h2y = [float(lm.y) for lm in first_rh[:21]]
            if len(h2x) < 21:
                h2x.extend([np.nan] * (21 - len(h2x)))
                h2y.extend([np.nan] * (21 - len(h2y)))
        else:
            h2x = [np.nan] * 21
            h2y = [np.nan] * 21
        hand2_x_list.append(h2x)
        hand2_y_list.append(h2y)

    cap.release()

    n_frames = len(pose_x_list)
    if n_frames == 0:
        raise ValueError(f"Video {video_path} yielded 0 frames")

    # Convert to numpy arrays of shape (T, N)
    pose_x = np.array(pose_x_list, dtype=np.float32)
    pose_y = np.array(pose_y_list, dtype=np.float32)
    h1_x = np.array(hand1_x_list, dtype=np.float32)
    h1_y = np.array(hand1_y_list, dtype=np.float32)
    h2_x = np.array(hand2_x_list, dtype=np.float32)
    h2_y = np.array(hand2_y_list, dtype=np.float32)

    # Interpolate across frames and scale to pixel coordinates
    pose_x_interp = interp_channel(pose_x, target_width)
    pose_y_interp = interp_channel(pose_y, target_height)
    h1_x_interp = interp_channel(h1_x, target_width)
    h1_y_interp = interp_channel(h1_y, target_height)
    h2_x_interp = interp_channel(h2_x, target_width)
    h2_y_interp = interp_channel(h2_y, target_height)

    # Interleave x and y coordinates: [x0, y0, x1, y1, ...]
    pose_xy = np.empty((n_frames, 50), dtype=np.float32)
    pose_xy[:, 0::2] = pose_x_interp
    pose_xy[:, 1::2] = pose_y_interp

    h1_xy = np.empty((n_frames, 42), dtype=np.float32)
    h1_xy[:, 0::2] = h1_x_interp
    h1_xy[:, 1::2] = h1_y_interp

    h2_xy = np.empty((n_frames, 42), dtype=np.float32)
    h2_xy[:, 0::2] = h2_x_interp
    h2_xy[:, 1::2] = h2_y_interp

    # Concatenate into (T, 134)
    features_134 = np.concatenate([pose_xy, h1_xy, h2_xy], axis=-1)

    # Pad or truncate to max_frame_len
    if features_134.shape[0] < max_frame_len:
        pad_width = ((0, max_frame_len - features_134.shape[0]), (0, 0))
        features_134 = np.pad(features_134, pad_width, mode="constant", constant_values=0.0)
    else:
        features_134 = features_134[:max_frame_len]

    return features_134


def format_display_name(label: str) -> str:
    """Formats canonical class keys into human-readable sign titles."""
    custom_names: Dict[str, str] = {
        "thankyou": "THANK YOU",
        "goodmorning": "GOOD MORNING",
        "goodevening": "GOOD EVENING",
        "goodafternoon": "GOOD AFTERNOON",
        "goodnight": "GOOD NIGHT",
        "howareyou": "HOW ARE YOU",
        "biglarge": "BIG / LARGE",
        "smalllittle": "SMALL / LITTLE",
        "storeorshop": "STORE / SHOP",
        "streetorroad": "STREET / ROAD",
        "raceethnicity": "RACE / ETHNICITY",
        "trainticket": "TRAIN TICKET",
        "trainstation": "TRAIN STATION",
        "secondnumber": "SECOND (NUMBER)",
        "youplural": "YOU (PLURAL)",
    }
    if label in custom_names:
        return custom_names[label]
    return label.upper().replace("_", " ")


@dataclass
class CandidatePrediction:
    label: str
    display_name: str
    confidence: float


@dataclass
class INCLUDEResult:
    label: str
    display_name: str
    confidence: float
    top5_candidates: List[CandidatePrediction]
    sequence_length: int
    feature_dimension: int
    inference_time_ms: float
    model_name: str = "AI4Bharat-INCLUDE-263-LSTM"
    is_known_sign: bool = True
    entropy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 4),
            "is_known_sign": self.is_known_sign,
            "entropy": round(self.entropy, 4),
            "top5_candidates": [
                {
                    "label": c.label,
                    "display_name": c.display_name,
                    "confidence": round(c.confidence, 4),
                }
                for c in self.top5_candidates
            ],
            "sequence_length": self.sequence_length,
            "feature_dimension": self.feature_dimension,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model_name": self.model_name,
        }


class INCLUDEInferenceService:
    """Singleton service for loading and executing AI4Bharat INCLUDE inference."""

    _instance: Optional[INCLUDEInferenceService] = None

    def __init__(
        self,
        checkpoint_path: Path = Path("models/pretrained/include_no_cnn_lstm.pth"),
        label_map_path: Path = Path("models/pretrained/label_map_include.json"),
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.label_map_path = label_map_path
        self.model: Optional[INCLUDELstmModel] = None
        self.label_map: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.is_loaded: bool = False
        self.checkpoint_score: Optional[float] = None

    @classmethod
    def get_instance(cls) -> INCLUDEInferenceService:
        if cls._instance is None:
            cls._instance = INCLUDEInferenceService()
        return cls._instance

    def load(self) -> None:
        """Loads official checkpoint and label map with verification."""
        if self.is_loaded:
            return

        if not self.label_map_path.exists():
            raise FileNotFoundError(f"INCLUDE label map not found at {self.label_map_path.resolve()}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"INCLUDE checkpoint not found at {self.checkpoint_path.resolve()}")

        logger.info("Loading AI4Bharat INCLUDE label mapping from %s...", self.label_map_path)
        with open(self.label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        num_classes = len(self.label_map)

        logger.info("Loading AI4Bharat INCLUDE checkpoint from %s...", self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.checkpoint_score = float(ckpt.get("score", 0.0))

        self.model = INCLUDELstmModel(INCLUDELstmConfig(), n_classes=num_classes)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()
        self.is_loaded = True
        logger.info(
            "INCLUDE model loaded successfully with %d classes (checkpoint score: %.4f).",
            num_classes,
            self.checkpoint_score,
        )

    def predict_features(self, features_134: np.ndarray) -> INCLUDEResult:
        """Runs inference given a (seq_len, 134) or (batch, seq_len, 134) feature array."""
        if not self.is_loaded or self.model is None:
            self.load()

        if features_134.ndim == 2:
            if features_134.shape[1] != 134:
                raise ValueError(
                    f"INCLUDE model expects 134 features per frame, got {features_134.shape[1]}. "
                    "Do not pass 158-D features to the INCLUDE model."
                )
            seq_len = features_134.shape[0]
            # Pad if shorter than 200, or truncate
            if seq_len < 200:
                features_134 = np.pad(features_134, ((0, 200 - seq_len), (0, 0)), "constant")
            elif seq_len > 200:
                features_134 = features_134[:200]
            tensor_in = torch.from_numpy(features_134).unsqueeze(0).float()
        elif features_134.ndim == 3:
            if features_134.shape[2] != 134:
                raise ValueError(f"INCLUDE model expects 134 features per frame, got {features_134.shape[2]}.")
            tensor_in = torch.from_numpy(features_134).float()
            seq_len = features_134.shape[1]
        else:
            raise ValueError(f"Invalid features array shape: {features_134.shape}")

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        top5_indices = np.argsort(probs)[::-1][:5]
        top1_idx = int(top5_indices[0])
        top1_label = self.idx_to_label[top1_idx]
        top1_conf = float(probs[top1_idx])

        top5 = [
            CandidatePrediction(
                label=self.idx_to_label[int(idx)],
                display_name=format_display_name(self.idx_to_label[int(idx)]),
                confidence=float(probs[int(idx)]),
            )
            for idx in top5_indices
        ]

        return INCLUDEResult(
            label=top1_label,
            display_name=format_display_name(top1_label),
            confidence=top1_conf,
            top5_candidates=top5,
            sequence_length=seq_len,
            feature_dimension=134,
            inference_time_ms=inference_time_ms,
            is_known_sign=True,
        )

    def predict_video(self, video_path: Path, landmarker: HolisticLandmarkerWrapper) -> INCLUDEResult:
        """Extracts 134-D keypoints from video and runs inference."""
        feats = extract_include_134_features(video_path, landmarker)
        return self.predict_features(feats)


class StreamingIncludeAdapter:
    """
    Maintains a 30-frame rolling FIFO landmark queue for real-time live webcam streaming inference
    with the official AI4Bharat INCLUDE 263-class LSTM model.
    """

    def __init__(
        self,
        service: Optional[INCLUDEInferenceService] = None,
        buffer_size: int = 30,
        eval_every_n_frames: int = 3,
        min_frames: int = 12,
        entropy_threshold: float = 4.8,
        target_width: float = 1920.0,
        target_height: float = 1080.0,
        window_size: Optional[int] = None,
    ) -> None:
        from collections import deque
        self.service = service or INCLUDEInferenceService.get_instance()
        self.buffer_size = window_size if window_size is not None else buffer_size
        self.eval_every_n_frames = eval_every_n_frames
        self.min_frames = min_frames
        self.entropy_threshold = entropy_threshold
        self.target_width = target_width
        self.target_height = target_height
        self.frame_buffer: deque = deque(maxlen=self.buffer_size)
        self.frame_counter: int = 0
        self.last_result: Optional[INCLUDEResult] = None

    def load(self) -> None:
        """Loads underlying INCLUDE inference service model and labels."""
        self.service.load()

    def reset(self) -> None:
        """Clears the rolling buffer and resets frame counter."""
        self.frame_buffer.clear()
        self.frame_counter = 0
        self.last_result = None

    def extract_frame_134d(
        self,
        pose_landmarks: Optional[List[Any]],
        left_hand_landmarks: Optional[List[Any]],
        right_hand_landmarks: Optional[List[Any]],
    ) -> np.ndarray:
        """
        Constructs a 134-D vector from raw or serialized landmarks:
        - 25 pose landmarks (x, y) * (target_width, target_height) -> 50 features
        - 21 left hand landmarks (x, y) * (target_width, target_height) -> 42 features
        - 21 right hand landmarks (x, y) * (target_width, target_height) -> 42 features
        Missing landmarks are padded with 0.0.
        """
        vec = np.zeros(134, dtype=np.float32)

        def get_xy(lm: Any) -> Tuple[float, float]:
            if isinstance(lm, dict):
                return float(lm.get("x", 0.0)), float(lm.get("y", 0.0))
            return float(getattr(lm, "x", 0.0)), float(getattr(lm, "y", 0.0))

        # 1. Pose: 25 landmarks (0 to 24) -> 50 coords
        if pose_landmarks:
            first_pose = pose_landmarks[0] if (len(pose_landmarks) > 0 and isinstance(pose_landmarks[0], list)) else pose_landmarks
            for i, lm in enumerate(first_pose[:25]):
                x, y = get_xy(lm)
                vec[i * 2] = x * self.target_width
                vec[i * 2 + 1] = y * self.target_height

        # 2. Left Hand: 21 landmarks -> 42 coords (offset 50)
        if left_hand_landmarks:
            first_lh = left_hand_landmarks[0] if (len(left_hand_landmarks) > 0 and isinstance(left_hand_landmarks[0], list)) else left_hand_landmarks
            for i, lm in enumerate(first_lh[:21]):
                x, y = get_xy(lm)
                vec[50 + i * 2] = x * self.target_width
                vec[50 + i * 2 + 1] = y * self.target_height

        # 3. Right Hand: 21 landmarks -> 42 coords (offset 92)
        if right_hand_landmarks:
            first_rh = right_hand_landmarks[0] if (len(right_hand_landmarks) > 0 and isinstance(right_hand_landmarks[0], list)) else right_hand_landmarks
            for i, lm in enumerate(first_rh[:21]):
                x, y = get_xy(lm)
                vec[92 + i * 2] = x * self.target_width
                vec[92 + i * 2 + 1] = y * self.target_height

        return vec

    def append_frame(
        self,
        pose_landmarks: Optional[List[Any]],
        left_hand_landmarks: Optional[List[Any]],
        right_hand_landmarks: Optional[List[Any]],
    ) -> None:
        """Appends landmarks of current frame into rolling FIFO queue."""
        feat_134 = self.extract_frame_134d(pose_landmarks, left_hand_landmarks, right_hand_landmarks)
        self.frame_buffer.append(feat_134)
        self.frame_counter += 1

    def append_vector(self, feat_134: np.ndarray) -> None:
        """Appends precomputed 134-D vector into rolling FIFO queue."""
        if feat_134.shape != (134,):
            raise ValueError(f"Expected 134-D vector, got {feat_134.shape}")
        self.frame_buffer.append(feat_134.astype(np.float32))
        self.frame_counter += 1

    def should_evaluate(self) -> bool:
        """Evaluates every eval_every_n_frames once min_frames have accumulated."""
        return len(self.frame_buffer) >= self.min_frames and (self.frame_counter % self.eval_every_n_frames == 0)

    def predict_window(self, confidence_threshold: float = 0.35) -> INCLUDEResult:
        """
        Runs sliding-window LSTM inference over the buffered sequence.
        Gated by confidence, top1-top2 margin, and softmax entropy.
        """
        if len(self.frame_buffer) < self.min_frames:
            return self._empty_result()

        arr = np.array(self.frame_buffer, dtype=np.float32)
        res = self.service.predict_features(arr)

        if res.top5_candidates:
            top_probs = np.array([c.confidence for c in res.top5_candidates], dtype=np.float32)
            entropy = float(-np.sum(top_probs * np.log(np.clip(top_probs, 1e-9, 1.0))))
        else:
            entropy = 10.0

        res.entropy = entropy
        margin = (res.top5_candidates[0].confidence - res.top5_candidates[1].confidence) if len(res.top5_candidates) > 1 else res.confidence

        is_valid = (
            res.confidence >= confidence_threshold
            and margin >= 0.04
            and entropy <= self.entropy_threshold
        )
        res.is_known_sign = is_valid
        if not is_valid:
            res.label = "awaiting_sign"
            res.display_name = "Awaiting Sign..."

        self.last_result = res
        return res

    def _empty_result(self) -> INCLUDEResult:
        return INCLUDEResult(
            label="awaiting_sign",
            display_name="Awaiting Sign...",
            confidence=0.0,
            top5_candidates=[],
            sequence_length=len(self.frame_buffer),
            feature_dimension=134,
            inference_time_ms=0.0,
            is_known_sign=False,
            entropy=10.0,
        )


# Compatibility aliases
IncludeClassifier = INCLUDEInferenceService
IncludePrediction = INCLUDEResult

__all__ = [
    "INCLUDELstmConfig",
    "INCLUDELstmModel",
    "CandidatePrediction",
    "INCLUDEResult",
    "IncludePrediction",
    "INCLUDEInferenceService",
    "IncludeClassifier",
    "StreamingIncludeAdapter",
    "extract_include_134_features",
]

