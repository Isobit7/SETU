"""
SETU - Minimal OpenCV Demonstration for Pretrained RF-UV-11 Model
Reuses existing CameraReader and HolisticLandmarkerWrapper.

Features:
- Live webcam feed with natural mirrored preview.
- Hand landmark visualization with bone connections.
- Real-time predicted letter and confidence display.
- Clear 'NO HAND DETECTED' state when hands are absent.
- Temporal smoothing window to eliminate frame jitter.
- Graceful error handling for missing hardware or missing model artifacts.
- CLI flags: --camera <index>, --video <path>, --max-frames <N>, --headless.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Any

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from backend.app.services.vision import CameraReader
from backend.app.services.vision import HolisticLandmarkerWrapper
from models.adapters.rf_uv11 import RFUV11Adapter
from config.exceptions import CameraError, ModelArtifactError, LandmarkExtractionError, SETUError

# MediaPipe 21 hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


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
    hand_detected: bool,
    letter: Optional[str] = None,
    confidence: float = 0.0,
    raw_letter: Optional[str] = None,
    raw_conf: float = 0.0,
    top3: Optional[List[Tuple[str, float]]] = None,
    fps: float = 0.0,
    hand_side: str = "Right"
) -> None:
    """Renders high-contrast, clean HUD overlay on preview frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 30), -1)
    # Status card on right
    card_w, card_h = 240, 150
    cv2.rectangle(overlay, (w - card_w - 15, 100), (w - 15, 100 + card_h), (25, 28, 36), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Header title
    cv2.putText(
        frame, "SETU - RF-UV-11 Pretrained Recognition Demo",
        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"FPS: {fps:.1f} | Hand: {hand_side} | Model: 63-D MLP (Apache-2.0)",
        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1, cv2.LINE_AA
    )

    card_x = w - card_w - 5
    if hand_detected and letter is not None:
        # Display Prediction Badge
        box_color = (0, 200, 100) if confidence >= 0.70 else (0, 180, 240)
        cv2.rectangle(frame, (20, 100), (160, 230), (25, 28, 36), -1)
        cv2.rectangle(frame, (20, 100), (160, 230), box_color, 2)
        
        cv2.putText(
            frame, letter,
            (45, 195), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"{confidence * 100:.1f}%",
            (48, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA
        )

        # Details in card
        cv2.putText(
            frame, "RECOGNITION STATUS",
            (card_x, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Smoothed: {letter} ({confidence*100:.1f}%)",
            (card_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"Raw Frame: {raw_letter} ({raw_conf*100:.1f}%)",
            (card_x, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
        )
        if top3:
            top_str = " | ".join([f"{l}:{c*100:.0f}%" for l, c in top3[:3]])
            cv2.putText(
                frame, f"Top3: {top_str}",
                (card_x, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 220, 180), 1, cv2.LINE_AA
            )
    else:
        # No Hand Detected
        cv2.putText(
            frame, "NO HAND DETECTED",
            (card_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 100, 240), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Place hand in frame",
            (card_x, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Predictions paused",
            (card_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA
        )


def run_demo(
    camera_index: int = 0,
    video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    model_path: Optional[str] = None,
    headless: bool = False
) -> None:
    """Runs the live OpenCV camera gesture recognition pipeline."""
    print("=" * 70)
    print("SETU ΓÇö Pretrained RF-UV-11 Gesture Recognition Live Demonstration")
    print("=" * 70)

    # 1. Initialize Model Adapter
    try:
        adapter = RFUV11Adapter(model_path=model_path, window_size=5, confidence_threshold=0.45)
        print(f"[OK] RF-UV-11 Model Adapter initialized from: {adapter.model_path}")
    except ModelArtifactError as e:
        print(f"\n[ERROR] Model Artifact Failure:\n  {e}\n", file=sys.stderr)
        return

    # 2. Initialize Holistic Landmarker
    try:
        landmarker = HolisticLandmarkerWrapper()
        landmarker.initialize()
        print(f"[OK] MediaPipe Tasks Landmarker initialized: {landmarker.task_model_path}")
    except ModelArtifactError as e:
        print(f"\n[ERROR] Landmark Model Failure:\n  {e}\n", file=sys.stderr)
        return

    # 3. Initialize Video Feed
    if video_path is not None:
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"[ERROR] Video file not found: {video_file}", file=sys.stderr)
            return
        cap = cv2.VideoCapture(str(video_file))
        is_webcam = False
        print(f"[OK] Video file opened: {video_file.name}")
    else:
        try:
            camera = CameraReader(camera_index=camera_index, mirror_preview=True)
            camera.open()
            is_webcam = True
            print(f"[OK] Camera index {camera_index} opened successfully.")
        except CameraError as e:
            print(f"\n[ERROR] Camera Hardware Failure:\n  {e}\n", file=sys.stderr)
            return

    print("\nStarting live recognition loop. Press 'q' to quit, 's' to save snapshot.")
    print("-" * 70)

    frame_count = 0
    t_start = time.perf_counter()
    fps = 0.0

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                print(f"\nReached maximum requested frames ({max_frames}). Stopping.")
                break

            # Read frame
            if is_webcam:
                success, raw_frame, preview_frame = camera.read_frame()
                if not success or raw_frame is None or preview_frame is None:
                    print("[WARN] Camera returned empty frame. Retrying...")
                    time.sleep(0.05)
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\nEnd of video stream reached.")
                    break
                raw_frame = frame
                preview_frame = frame.copy()

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.perf_counter() - t_start
                fps = frame_count / max(elapsed, 0.001)

            # Detect landmarks
            try:
                res = landmarker.detect(raw_frame)
            except LandmarkExtractionError as e:
                print(f"[WARN] Landmark extraction error: {e}")
                continue

            # Prioritize right hand, fallback to left hand
            hand_landmarks = None
            hand_side = "None"
            if res.right_hand_landmarks and len(res.right_hand_landmarks) == 21:
                hand_landmarks = res.right_hand_landmarks
                hand_side = "Right"
            elif res.left_hand_landmarks and len(res.left_hand_landmarks) == 21:
                hand_landmarks = res.left_hand_landmarks
                hand_side = "Left"

            # Predict if hand detected
            if hand_landmarks is not None:
                # Draw skeleton
                draw_styled_hand(preview_frame, hand_landmarks, mirrored=is_webcam)

                # Extract features & predict
                try:
                    feat63 = adapter.extract_63_features(hand_landmarks)
                    smoothed_lbl, s_conf, raw_lbl, r_conf = adapter.predict_smoothed(feat63)
                    _, _, top3 = adapter.predict_raw(feat63)

                    # Draw HUD with predictions
                    draw_hud(
                        preview_frame,
                        hand_detected=True,
                        letter=smoothed_lbl or raw_lbl,
                        confidence=s_conf if smoothed_lbl else r_conf,
                        raw_letter=raw_lbl,
                        raw_conf=r_conf,
                        top3=top3,
                        fps=fps,
                        hand_side=hand_side
                    )

                    if frame_count % 30 == 0 or frame_count == 1:
                        top_repr = " ".join([f"{l}:{c*100:.0f}%" for l, c in top3[:2]])
                        print(f"Frame {frame_count:04d} | Hand: {hand_side} | Pred: '{raw_lbl}' ({r_conf*100:.1f}%) | Smooth: '{smoothed_lbl}' | Top: [{top_repr}]")

                except (InferenceError, LandmarkExtractionError) as e:
                    cv2.putText(preview_frame, f"Inference Error: {e}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # No hand present
                adapter.reset_window()
                draw_hud(preview_frame, hand_detected=False, fps=fps, hand_side="None")

            # Display window unless running in headless mode
            if not headless:
                cv2.imshow("SETU - Pretrained RF-UV-11 Demonstration", preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\nUser requested exit ('q').")
                    break
                elif key == ord("s"):
                    out_snap = Path(f"snapshot_frame_{frame_count:04d}.png")
                    cv2.imwrite(str(out_snap), preview_frame)
                    print(f"[SNAPSHOT] Saved preview frame to {out_snap}")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        if is_webcam:
            camera.release()
        else:
            cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print(f"[DONE] Demo finished. Total frames processed: {frame_count}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SETU Pretrained RF-UV-11 Gesture Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file path for offline testing")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--model", type=str, default=None, help="Custom path to model weights")
    parser.add_argument("--headless", action="store_true", help="Run without creating an OpenCV GUI window")

    args = parser.parse_args()
    run_demo(
        camera_index=args.camera,
        video_path=args.video,
        max_frames=args.max_frames,
        model_path=args.model,
        headless=args.headless
    )
