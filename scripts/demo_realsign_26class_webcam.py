"""
SETU — Live Real-Time Webcam Demonstration for 26-Class RealSign PyTorch MLP

Features:
- Captures live webcam stream via OpenCV cv2.VideoCapture(0).
- Extracts 21 3D hand landmarks in real-time with MediaPipe HandLandmarker.
- Normalizes landmarks: wrist-centering (P_i - P_0) and palm span scaling (||P_9 - P_0||_2).
- Classifies 63-D features into 26 ISL alphabet letters (A–Z) using trained PyTorch MLP.
- Renders hand skeleton, bounding box, top-left FPS HUD, and bottom movie-style subtitle:
    * If confidence >= 0.60: displays "[Letter] (Confidence %)"
    * Else: displays "Detecting..."
- Graceful shutdown when user presses 'q' or 'ESC'.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.feature_extractor import normalize_hand_landmarks

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "realsign_26class_mlp.pth"
DEFAULT_TASK_PATH = ROOT_DIR / "models" / "tasks" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class RealSignMLP(nn.Module):
    """PyTorch MLP Classifier for 26-class ISL Hand Alphabet Recognition."""

    def __init__(
        self,
        input_dim: int = 63,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 26,
        dropout: float = 0.2,
    ):
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


def load_classifier(model_path: Path) -> Tuple[RealSignMLP, List[str]]:
    """Loads trained RealSign 26-class MLP model and verified class list."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading PyTorch model checkpoint: {model_path}...")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    classes = checkpoint.get("classes", [chr(ord("A") + i) for i in range(26)])
    arch = checkpoint.get("architecture", {})
    input_dim = arch.get("input_dim", 63)
    hidden1 = arch.get("hidden1", 128)
    hidden2 = arch.get("hidden2", 64)
    num_classes = arch.get("num_classes", len(classes))
    dropout = arch.get("dropout", 0.2)

    model = RealSignMLP(
        input_dim=input_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f" Model loaded successfully! Classes: {len(classes)} ({classes[0]}..{classes[-1]})")
    return model, classes


def create_hand_detector(task_path: Path) -> vision.HandLandmarker:
    """Initializes MediaPipe HandLandmarker in IMAGE running mode."""
    if not task_path.exists():
        raise FileNotFoundError(f"Hand landmarker task asset not found: {task_path}")

    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_annotations(
    frame: np.ndarray,
    landmarks: Any,
    color_joint: Tuple[int, int, int] = (0, 240, 180),  # emerald
    color_bone: Tuple[int, int, int] = (255, 190, 0),   # amber
) -> Tuple[int, int, int, int]:
    """
    Renders skeleton connections, joints, and bounding box for detected hand.
    Returns: (x_min, y_min, x_max, y_max)
    """
    h, w = frame.shape[:2]
    coords: List[Tuple[int, int]] = []

    for lm in landmarks:
        px = int(np.clip(lm.x * w, 0, w - 1))
        py = int(np.clip(lm.y * h, 0, h - 1))
        coords.append((px, py))

    # Draw skeleton bones
    for s_idx, e_idx in HAND_CONNECTIONS:
        if s_idx < len(coords) and e_idx < len(coords):
            cv2.line(frame, coords[s_idx], coords[e_idx], color_bone, 2, cv2.LINE_AA)

    # Draw joint nodes
    for idx, (px, py) in enumerate(coords):
        radius = 5 if idx in (4, 8, 12, 16, 20) else 3  # larger fingertip joints
        cv2.circle(frame, (px, py), radius, color_joint, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

    # Calculate bounding box
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    x_min = max(0, min(xs) - 18)
    y_min = max(0, min(ys) - 18)
    x_max = min(w - 1, max(xs) + 18)
    y_max = min(h - 1, max(ys) + 18)

    # Draw styled bounding box corners
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 220, 160), 2)
    return x_min, y_min, x_max, y_max


def render_ui_overlay(
    frame: np.ndarray,
    fps: float,
    pred_letter: Optional[str],
    confidence: float,
    top_3: List[Tuple[str, float]],
    hand_detected: bool,
) -> None:
    """
    Renders the movie-style subtitle banner and HUD overlays on preview frame.
    """
    h, w = frame.shape[:2]

    # --- TOP HUD HEADER ---
    top_bar = frame.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 55), (18, 22, 28), -1)
    cv2.addWeighted(top_bar, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame, "SETU — RealSign 26-Class Alphabet Recognizer",
        (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:4.1f} | Hand Tracking: {'Active' if hand_detected else 'Searching'} | Press 'q' or ESC to exit",
        (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 220), 1, cv2.LINE_AA
    )

    # --- TOP RIGHT PROBABILITY CARD (IF DETECTED) ---
    if hand_detected and top_3:
        card_w, card_h = 190, 80
        card_x = w - card_w - 15
        card_y = 65
        card_bg = frame.copy()
        cv2.rectangle(card_bg, (card_x, card_y), (w - 15, card_y + card_h), (20, 25, 32), -1)
        cv2.addWeighted(card_bg, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (w - 15, card_y + card_h), (60, 75, 90), 1)

        cv2.putText(
            frame, "TOP PREDICTIONS",
            (card_x + 10, card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 180, 210), 1, cv2.LINE_AA
        )
        for r_idx, (let, prob) in enumerate(top_3):
            line_y = card_y + 36 + (r_idx * 18)
            cv2.putText(
                frame, f"{r_idx+1}. Letter {let}: {prob*100:4.1f}%",
                (card_x + 12, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 180) if r_idx == 0 and prob >= 0.60 else (220, 220, 220), 1, cv2.LINE_AA
            )

    # --- MOVIE-STYLE SUBTITLE OVERLAY AT BOTTOM ---
    sub_h = 70
    sub_y = h - sub_h - 15
    sub_margin = 40
    sub_w = w - (2 * sub_margin)

    # Translucent dark band
    sub_bg = frame.copy()
    cv2.rectangle(sub_bg, (sub_margin, sub_y), (w - sub_margin, sub_y + sub_h), (12, 16, 22), -1)
    cv2.addWeighted(sub_bg, 0.85, frame, 0.15, 0, frame)

    # Border glow
    border_color = (0, 220, 140) if (hand_detected and confidence >= 0.60) else (70, 80, 95)
    cv2.rectangle(frame, (sub_margin, sub_y), (w - sub_margin, sub_y + sub_h), border_color, 2)

    # Subtitle Text logic
    if hand_detected and confidence >= 0.60 and pred_letter:
        subtitle_text = f"[{pred_letter}] ({confidence * 100:.1f}%)"
        text_color = (0, 255, 170)
        font_scale = 1.15
        thickness = 2
    else:
        subtitle_text = "Detecting..."
        text_color = (180, 190, 200)
        font_scale = 0.85
        thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = sub_margin + (sub_w - text_w) // 2
    text_y = sub_y + (sub_h + text_h) // 2 - 2

    cv2.putText(
        frame, subtitle_text,
        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
    )


def run_webcam_demo(
    camera_index: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    task_path: Path = DEFAULT_TASK_PATH,
    max_frames: Optional[int] = None,
    headless: bool = False,
) -> None:
    """Main loop capturing webcam, predicting letter, and rendering HUD overlay."""
    print("=" * 75)
    print("SETU — RealSign 26-Class Alphabet Real-Time Webcam Demo")
    print("=" * 75)

    # 1. Load ML Model
    model, classes = load_classifier(model_path)

    # 2. Initialize MediaPipe Detector
    print(f"Initializing MediaPipe HandLandmarker from: {task_path}...")
    detector = create_hand_detector(task_path)
    print(" HandLandmarker initialized successfully!")

    # 3. Open Webcam
    print(f"Opening webcam at device index: {camera_index}...", flush=True)
    # On Windows, cv2.CAP_DSHOW opens in ~0.5s instead of negotiating for 15s
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        err_msg = (
            f"\n[ERROR] CameraError: Failed to open webcam index {camera_index}.\n"
            "Possible causes:\n"
            "  1. No webcam is connected to this device.\n"
            "  2. Another application (e.g. Teams, Zoom, browser) is currently holding the camera.\n"
            "  3. Camera permissions are blocked in Windows Settings (Privacy -> Camera).\n"
            "Try passing an alternative index via: --camera 1 (or verify hardware connection)."
        )
        print(err_msg, file=sys.stderr)
        sys.exit(1)

    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Webcam feed active: {actual_w}x{actual_h} resolution.")
    print("\n[CONTROLS]:")
    print("  - Position hand clearly inside the camera frame.")
    print("  - Perform an ISL alphabet sign (A to Z).")
    print("  - Press 'q' or 'ESC' on the video window to quit gracefully.")
    print("-" * 75)

    window_name = "SETU — RealSign ISL Alphabet Recognizer"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 30.0
    frame_count = 0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("\n[WARNING] Failed to grab frame from camera stream. Exiting loop.")
                break

            frame_count += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Flip horizontally for natural mirror interaction
            preview_frame = cv2.flip(raw_frame, 1)

            # Convert to MediaPipe Image
            img_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Detect hand
            detection_res = detector.detect(mp_img)

            hand_detected = False
            pred_letter: Optional[str] = None
            pred_conf = 0.0
            top_3: List[Tuple[str, float]] = []

            if detection_res and detection_res.hand_landmarks and len(detection_res.hand_landmarks) > 0:
                hand_detected = True
                lms = detection_res.hand_landmarks[0]

                # Draw skeleton and get bounding box
                draw_hand_annotations(preview_frame, lms)

                # Extract 21 3D points
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                # Normalize by wrist-centering and palm span distance
                norm_pts, _ = normalize_hand_landmarks(pts)
                feat_vec = norm_pts.flatten()  # shape (63,)

                # PyTorch Inference
                with torch.no_grad():
                    input_tensor = torch.from_numpy(feat_vec).unsqueeze(0).float()
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).numpy()[0]

                # Top predictions
                sorted_indices = np.argsort(probabilities)[::-1]
                pred_idx = sorted_indices[0]
                pred_letter = classes[pred_idx]
                pred_conf = float(probabilities[pred_idx])

                top_3 = [(classes[i], float(probabilities[i])) for i in sorted_indices[:3]]

            # Render movie-style subtitle and HUD
            render_ui_overlay(
                frame=preview_frame,
                fps=fps,
                pred_letter=pred_letter,
                confidence=pred_conf,
                top_3=top_3,
                hand_detected=hand_detected,
            )

            # Display frame
            if not headless:
                cv2.imshow(window_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    print("\nUser requested shutdown ('q' / ESC key pressed).")
                    break

            if max_frames and frame_count >= max_frames:
                print(f"\nReached max frames limit ({max_frames}). Stopping demo.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down gracefully...")

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("Camera hardware safely released. Demo terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Live Webcam Demo for RealSign 26-Class Alphabet Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to realsign_26class_mlp.pth")
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK_PATH, help="Path to hand_landmarker.task")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process before auto-exit")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI window")
    args = parser.parse_args()

    run_webcam_demo(
        camera_index=args.camera,
        model_path=args.model,
        task_path=args.task,
        max_frames=args.max_frames,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
