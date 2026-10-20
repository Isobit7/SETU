"""
SETU - Live OpenCV Demonstration for ISL 5-Sign MVP
Reuses existing CameraReader, HolisticLandmarkerWrapper, and FeatureExtractor158.

Features:
- Live camera capture (or offline video fallback).
- Multi-hand & pose landmark tracking with bone overlays.
- Rolling 30-frame temporal buffer fed into FeatureExtractor158.
- Pure NumPy ISLMVPClassifier inference (<0.2 ms).
- Recognizes 5 authentic ISL signs:
    1. HELLO
    2. THANK YOU
    3. GOOD MORNING
    4. I
    5. YOU
- Explicit NO SIGN / RESTING and UNKNOWN state handling.
- Real-time HUD showing active prediction, confidence %, and class probability breakdown.
- CLI flags: --camera <idx>, --video <path>, --max-frames <N>, --headless.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from setu.vision.camera import CameraReader
from setu.vision.landmarks import HolisticLandmarkerWrapper
from setu.features import FeatureExtractor158
from setu.models.isl_mvp_classifier import ISLMVPClassifier, ISLPrediction
from setu.exceptions import CameraError, ModelArtifactError, SETUError

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

DISPLAY_NAMES = {
    "hello": "HELLO",
    "thank_you": "THANK YOU",
    "good_morning": "GOOD MORNING",
    "i": "I (ME)",
    "you": "YOU",
    "no_sign": "NO SIGN / RESTING",
    "unknown": "UNKNOWN / UNCLEAR",
}


def draw_styled_hand(
    image: np.ndarray,
    landmarks: Any,
    mirrored: bool = True,
    color_joint: Tuple[int, int, int] = (0, 255, 200),
    color_bone: Tuple[int, int, int] = (0, 180, 255)
) -> None:
    """Draws hand skeleton connections and joints on an image."""
    h, w = image.shape[:2]
    coords = []
    for lm in landmarks:
        norm_x = (1.0 - lm.x) if mirrored else lm.x
        px = int(np.clip(norm_x * w, 0, w - 1))
        py = int(np.clip(lm.y * h, 0, h - 1))
        coords.append((px, py))

    # Draw bones
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(image, coords[start_idx], coords[end_idx], color_bone, 2, cv2.LINE_AA)

    # Draw joints
    for idx, (px, py) in enumerate(coords):
        radius = 5 if idx in (4, 8, 12, 16, 20) else 3
        cv2.circle(image, (px, py), radius, color_joint, -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    pred: ISLPrediction,
    hand_detected: bool,
    buffer_fill: float,
    fps: float,
) -> None:
    """Renders high-contrast, clean HUD overlay on preview frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 30), -1)

    # Status card on top-right
    card_w, card_h = 280, 185
    card_x = w - card_w - 15
    cv2.rectangle(overlay, (card_x, 95), (w - 15, 95 + card_h), (25, 28, 36), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Header title
    cv2.putText(
        frame, "SETU - Real ISL 5-Sign Recognition MVP",
        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f} | Buffer: {int(buffer_fill*100)}% | Model: 158-D Holistic -> Linear Softmax",
        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 190, 210), 1, cv2.LINE_AA
    )

    # Card Title
    cv2.putText(
        frame, "ISL RECOGNITION STATUS",
        (card_x + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 165, 190), 1, cv2.LINE_AA
    )

    display_label = DISPLAY_NAMES.get(pred.label, pred.label.upper())

    if pred.is_known_sign:
        # Green / cyan active badge
        box_color = (0, 230, 120) if pred.confidence >= 0.75 else (0, 190, 240)
        cv2.rectangle(frame, (20, 100), (320, 195), (25, 28, 36), -1)
        cv2.rectangle(frame, (20, 100), (320, 195), box_color, 2)

        cv2.putText(
            frame, display_label,
            (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Confidence: {pred.confidence * 100:.1f}%",
            (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA
        )
    else:
        # Neutral / warning badge for resting / unknown
        badge_color = (90, 120, 240) if pred.label == "unknown" else (120, 130, 140)
        cv2.rectangle(frame, (20, 100), (320, 195), (25, 28, 36), -1)
        cv2.rectangle(frame, (20, 100), (320, 195), badge_color, 1)

        cv2.putText(
            frame, display_label,
            (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 210, 220), 2, cv2.LINE_AA
        )
        status_msg = "Hands moving/unclear" if hand_detected else "No active hands detected"
        cv2.putText(
            frame, status_msg,
            (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 160, 170), 1, cv2.LINE_AA
        )

    # Probability ranking in card
    sorted_probs = sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    y_offset = 145
    for sign_name, prob in sorted_probs[:4]:
        bar_len = int(prob * 90)
        sign_disp = DISPLAY_NAMES.get(sign_name, sign_name)[:12]
        cv2.putText(
            frame, f"{sign_disp:12s}:",
            (card_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 200, 210), 1, cv2.LINE_AA
        )
        cv2.rectangle(frame, (card_x + 120, y_offset - 9), (card_x + 120 + bar_len, y_offset - 2), (0, 180, 240), -1)
        cv2.putText(
            frame, f"{prob*100:4.1f}%",
            (card_x + 220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA
        )
        y_offset += 25


def run_isl_mvp_demo(
    camera_index: int = 0,
    video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    model_path: Optional[str] = None,
    headless: bool = False,
    confidence_threshold: float = 0.50,
) -> None:
    """Executes the live ISL MVP demo loop."""
    print("=" * 70)
    print("SETU — Real ISL 5-Sign MVP Live Demonstration")
    print("Vocabulary: [HELLO, THANK YOU, GOOD MORNING, I, YOU] + [NO SIGN / RESTING]")
    print("=" * 70)

    # 1. Initialize Model
    m_path = Path(model_path) if model_path else Path("models/isl_mvp_model.npz")
    try:
        classifier = ISLMVPClassifier(m_path)
        print(f"[OK] ISL MVP Classifier loaded from: {m_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load ISL model: {e}", file=sys.stderr)
        return

    # 2. Initialize Landmarker & Feature Extractor
    try:
        landmarker = HolisticLandmarkerWrapper()
        landmarker.initialize()
        extractor = FeatureExtractor158()
        print(f"[OK] MediaPipe Holistic Landmarker & 158-D Extractor ready")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Vision pipeline: {e}", file=sys.stderr)
        return

    # 3. Open Video Source
    is_webcam = video_path is None
    cam: Optional[CameraReader] = None
    cap: Optional[cv2.VideoCapture] = None

    if video_path is not None:
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"[ERROR] Video file not found: {video_file}", file=sys.stderr)
            return
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_file}", file=sys.stderr)
            return
        print(f"[OK] Video file opened: {video_file.name}")
    else:
        try:
            cam = CameraReader(camera_index=camera_index, mirror_preview=False)
            cam.open()
            print(f"[OK] Camera index {camera_index} opened successfully.")
        except CameraError as e:
            print(f"[ERROR] Camera open failed: {e}", file=sys.stderr)
            return

    # 4. State buffers
    rolling_features: deque = deque(maxlen=30)
    current_pred = ISLPrediction(
        label="no_sign",
        confidence=1.0,
        is_known_sign=False,
        probabilities={c: 0.0 for c in classifier.class_names},
        raw_class_idx=classifier.class_names.index("no_sign"),
    )

    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()
    fps_frame_count = 0

    print("\n[INFO] Starting recognition loop. Press 'q' or ESC in preview window to exit.\n")

    try:
        while True:
            if is_webcam and cam is not None:
                success, frame_raw, preview_frame = cam.read_frame()
                if not success or frame_raw is None:
                    print("[WARN] Camera returned empty frame. Retrying...")
                    time.sleep(0.05)
                    continue
            elif cap is not None:
                ret, frame_raw = cap.read()
                if not ret or frame_raw is None:
                    print("[INFO] Video feed reached end.")
                    break
                preview_frame = cv2.flip(frame_raw, 1)
            else:
                break

            frame_count += 1
            fps_frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = fps_frame_count / (now - last_fps_time)
                last_fps_time = now
                fps_frame_count = 0

            # Extract holistic landmarks
            hl_result = landmarker.detect(frame_raw)

            # Extract 158-D single-frame feature vector
            feat_158 = FeatureExtractor158.extract_from_mediapipe_result(hl_result)
            rolling_features.append(feat_158)

            has_left = bool(hl_result.left_hand_landmarks and len(hl_result.left_hand_landmarks) > 0)
            has_right = bool(hl_result.right_hand_landmarks and len(hl_result.right_hand_landmarks) > 0)
            hand_detected = has_left or has_right

            # Inference triggered when we have at least 15 frames
            if len(rolling_features) >= 15:
                # If buffer is not full 30, repeat latest frame to fill 30
                feat_array = np.array(rolling_features, dtype=np.float32)
                if feat_array.shape[0] < 30:
                    pad = np.repeat(feat_array[-1:], 30 - feat_array.shape[0], axis=0)
                    feat_array = np.concatenate([feat_array, pad], axis=0)

                current_pred = classifier.predict_features(
                    feat_array, confidence_threshold=confidence_threshold
                )

            # Render overlay
            display_frame = cv2.flip(frame_raw, 1)  # Mirror preview for user naturalness
            if hl_result.left_hand_landmarks:
                draw_styled_hand(display_frame, hl_result.left_hand_landmarks, mirrored=True)
            if hl_result.right_hand_landmarks:
                draw_styled_hand(display_frame, hl_result.right_hand_landmarks, mirrored=True)

            buffer_fill = len(rolling_features) / 30.0
            draw_hud(display_frame, current_pred, hand_detected, buffer_fill, fps)

            if not headless:
                cv2.imshow("SETU - Real ISL Recognition Demo", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("[INFO] User quit requested.")
                    break

            if max_frames and frame_count >= max_frames:
                print(f"[INFO] Reached max frames ({max_frames}). Stopping.")
                break

    finally:
        if cam is not None:
            cam.release()
        if cap is not None:
            cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[OK] Demonstration session ended cleanly.")


def main():
    parser = argparse.ArgumentParser(description="SETU Real ISL Recognition Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of live camera")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .npz")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to run")
    parser.add_argument("--headless", action="store_true", help="Run without GUI preview window")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold")
    args = parser.parse_args()

    run_isl_mvp_demo(
        camera_index=args.camera,
        video_path=args.video,
        max_frames=args.max_frames,
        model_path=args.model,
        headless=args.headless,
        confidence_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
