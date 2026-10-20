"""Test inference with official AI4Bharat INCLUDE pretrained checkpoint on a real INCLUDE video."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from setu.vision.landmarks import HolisticLandmarkerWrapper


@dataclass
class LstmConfig:
    input_size: int = 134
    hidden_size: int = 256
    num_layers: int = 5
    batch_first: bool = True
    bidirectional: bool = True
    dropout: float = 0.2


class INCLUDELstmModel(nn.Module):
    def __init__(self, config: LstmConfig = LstmConfig(), n_classes: int = 263):
        super().__init__()
        self.config = config
        self.n_classes = n_classes
        self.lstm = nn.LSTM(**asdict(config))
        in_features = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, 134)
        out, _ = self.lstm(x)
        out = torch.max(out, dim=1).values
        if self.training:
            out = F.dropout(out, p=0.3)
        return self.l1(out)


def interp_channel(arr: np.ndarray, scale: float) -> np.ndarray:
    """Linearly interpolate missing landmark points (NaNs) across time frames, matching AI4Bharat dataset.py."""
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
    Scaled to (1920, 1080) pixel space and linearly interpolated across time frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    pose_x_list, pose_y_list = [], []
    hand1_x_list, hand1_y_list = [], []
    hand2_x_list, hand2_y_list = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hl_result = landmarker.detect(frame)

        # 1. Pose landmarks: 25 upper body landmarks
        if hl_result.pose_landmarks and len(hl_result.pose_landmarks) > 0:
            first_pose = hl_result.pose_landmarks[0] if isinstance(hl_result.pose_landmarks[0], list) else hl_result.pose_landmarks
            px = [lm.x for lm in first_pose[:25]]
            py = [lm.y for lm in first_pose[:25]]
            # Pad if fewer than 25
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
            h1x = [lm.x for lm in first_lh[:21]]
            h1y = [lm.y for lm in first_lh[:21]]
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
            h2x = [lm.x for lm in first_rh[:21]]
            h2y = [lm.y for lm in first_rh[:21]]
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


def main():
    checkpoint_path = Path("models/pretrained/include_no_cnn_lstm.pth")
    label_map_path = Path("models/pretrained/label_map_include.json")
    task_path = Path("models/tasks/holistic_landmarker.task")

    print(f"Loading label map from {label_map_path}...")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"Loaded {len(label_map)} classes.")

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = INCLUDELstmModel(LstmConfig(), n_classes=len(label_map))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print("Model initialized and weights loaded.")

    print("Initializing MediaPipe HolisticLandmarkerWrapper...")
    landmarker = HolisticLandmarkerWrapper(task_model_path=task_path)
    landmarker.initialize()

    test_videos = [
        Path("data/raw/include/hello_s01_mvi_0029.mov"),
        Path("data/raw/include/thank_you_s04_mvi_0006.mov"),
        Path("data/raw/include/good_morning_s07_mvi_0047.mov"),
        Path("data/raw/include/i_s10_mvi_0013.mov"),
        Path("data/raw/include/you_s13_mvi_0022.mov"),
    ]

    for video_path in test_videos:
        if not video_path.exists():
            print(f"Skipping {video_path} (not found)")
            continue

        print(f"\nProcessing {video_path.name}...")
        feats = extract_include_134_features(video_path, landmarker)
        print(f"Extracted features shape: {feats.shape}")

        tensor_in = torch.from_numpy(feats).unsqueeze(0)  # (1, 200, 134)
        with torch.no_grad():
            logits = model(tensor_in)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        top5_indices = np.argsort(probs)[::-1][:5]
        top1_idx = top5_indices[0]
        top1_label = idx_to_label[top1_idx]
        top1_conf = probs[top1_idx]

        print(f"Top-1 Prediction: {top1_label} ({top1_conf:.4f})")
        print("Top-5 Candidates:")
        for idx in top5_indices:
            print(f"  - {idx_to_label[idx]}: {probs[idx]:.4f}")

    landmarker.close()


if __name__ == "__main__":
    main()
