"""
SETU — Custom Sign Dataset Collection Utility
Interactive CLI tool for capturing 30-frame sequence recordings of custom ISL signs
with 164-D Holistic landmark coordinates and 6-D scale-invariant facial expression ratios.

Saves captured samples directly to data/processed/custom_<label>_<timestamp>.npz
for dynamic ISL Sequence LSTM model training.

Usage:
    python scripts/record_custom_signs.py --label namaste --num-samples 10
    python scripts/record_custom_signs.py --mock --num-samples 2 --headless
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from setu.features.extractor import FeatureExtractor158
from setu.features.schema import EXPANDED_FEATURE_DIM
from setu.vision.landmarks import HolisticLandmarkerWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SETU.Recorder")

DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def sanitize_label(label: str) -> str:
    """Sanitizes user input into a lowercase, filesystem-safe identifier."""
    cleaned = label.strip().lower().replace(" ", "_")
    alphanumeric = "".join(c for c in cleaned if c.isalnum() or c in ("_", "-"))
    return alphanumeric or "custom_sign"


def draw_hud(
    frame: np.ndarray,
    label: str,
    take_idx: int,
    total_takes: int,
    status_text: str,
    progress: float,
    status_color: Tuple[int, int, int] = (0, 255, 0),
    hand_detected: bool = False,
) -> np.ndarray:
    """Renders a clean HUD overlay with progress bar, sign label, and detection indicators."""
    h, w, _ = frame.shape
    vis = frame.copy()

    # Semi-transparent top bar
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    # Top border line
    cv2.line(vis, (0, 85), (w, 85), (40, 50, 70), 1)

    # Title & label
    title_text = f"SETU DATASET RECORDER: '{label.upper()}'"
    cv2.putText(vis, title_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Take counter and status
    take_info = f"Take: {take_idx}/{total_takes} | Status: {status_text}"
    cv2.putText(vis, take_info, (20, 65), cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # Hand detection badge
    badge_color = (40, 200, 100) if hand_detected else (80, 80, 220)
    badge_text = "HANDS: DETECTED" if hand_detected else "HANDS: NONE"
    cv2.putText(vis, badge_text, (w - 200, 48), cv2.FONT_HERSHEY_DUPLEX, 0.55, badge_color, 1, cv2.LINE_AA)

    # Bottom progress bar (if recording)
    if progress > 0.0:
        bar_y = h - 25
        bar_w = int((w - 40) * np.clip(progress, 0.0, 1.0))
        cv2.rectangle(vis, (20, bar_y), (w - 20, bar_y + 12), (30, 40, 50), -1)
        cv2.rectangle(vis, (20, bar_y), (20 + bar_w, bar_y + 12), status_color, -1)

    return vis


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    font_scale: float = 1.4,
    color: Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 3,
) -> None:
    """Draws centered text with a dark drop shadow for high readability."""
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def record_sign_samples(
    label: str,
    num_samples: int = 10,
    camera_idx: int = 0,
    countdown_sec: int = 3,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    headless: bool = False,
    mock: bool = False,
) -> List[Path]:
    """
    Executes the interactive sign recording loop:
    1. Initializes MediaPipe Holistic landmarker.
    2. Runs countdown before each take.
    3. Captures exactly 30 frames of 164-D features per sample.
    4. Saves compressed .npz archives to output_dir.
    """
    clean_label = sanitize_label(label)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []

    logger.info("=" * 60)
    logger.info(f"SETU Custom Sign Recorder: '{clean_label}'")
    logger.info(f"Target Takes: {num_samples} | Destination: {output_dir}")
    logger.info("=" * 60)

    landmarker: Optional[HolisticLandmarkerWrapper] = None
    if not mock:
        try:
            landmarker = HolisticLandmarkerWrapper()
            landmarker.initialize()
            logger.info("MediaPipe Holistic landmarker initialized.")
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed ({e}). Defaulting to mock frames.")
            mock = True

    cap: Optional[cv2.VideoCapture] = None
    if not mock:
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            logger.warning(f"Failed to open camera index {camera_idx}. Falling back to mock generator.")
            mock = True
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = f"SETU Dataset Recorder — '{clean_label}'"
    if not headless and not mock:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        for take_idx in range(1, num_samples + 1):
            logger.info(f"\n--- Preparing Take {take_idx}/{num_samples} for '{clean_label}' ---")

            # -----------------------------------------------------------
            # Phase 1: Visual Countdown
            # -----------------------------------------------------------
            countdown_start = time.time()
            while time.time() - countdown_start < countdown_sec:
                remaining = int(np.ceil(countdown_sec - (time.time() - countdown_start)))

                if mock:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    assert cap is not None
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        frame = cv2.flip(raw_frame, 1)  # mirror for user feedback

                hud_frame = draw_hud(
                    frame,
                    clean_label,
                    take_idx,
                    num_samples,
                    f"COUNTDOWN: {remaining}s",
                    0.0,
                    status_color=(0, 215, 255),
                    hand_detected=False,
                )
                draw_centered_text(hud_frame, f"Ready... {remaining}", font_scale=1.8, color=(0, 215, 255))

                if not headless and not mock:
                    cv2.imshow(window_name, hud_frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        logger.info("Recording cancelled by user.")
                        return saved_files
                else:
                    time.sleep(0.05)

            # -----------------------------------------------------------
            # Phase 2: Capture 30 Frames Sequence (164-D per frame)
            # -----------------------------------------------------------
            logger.info(f"Recording Take {take_idx}/{num_samples}... Perform sign now!")
            sequence_164d: List[np.ndarray] = []

            for frame_num in range(1, 31):
                if mock:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Synthetic 164-D vector with presence flags set
                    feat = np.zeros(EXPANDED_FEATURE_DIM, dtype=np.float32)
                    feat[146] = 1.0  # right hand present
                    feat[147] = 1.0  # left hand present
                    feat[:63] = np.random.randn(63).astype(np.float32) * 0.05
                    feat[73:136] = np.random.randn(63).astype(np.float32) * 0.05
                    # 6 facial ratios
                    feat[158:164] = [0.25, 0.12, 0.12, 0.12, 0.35, 0.35]
                    hand_detected = True
                else:
                    assert cap is not None
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        feat = np.zeros(EXPANDED_FEATURE_DIM, dtype=np.float32)
                        hand_detected = False
                    else:
                        frame = cv2.flip(raw_frame, 1)
                        try:
                            assert landmarker is not None
                            hl_res = landmarker.detect(raw_frame)
                            feat = FeatureExtractor158.extract_from_mediapipe_result(hl_res, include_facial=True)
                            has_lh = bool(hl_res.left_hand_landmarks)
                            has_rh = bool(hl_res.right_hand_landmarks)
                            hand_detected = has_lh or has_rh
                        except Exception as e:
                            logger.debug(f"Landmark error: {e}")
                            feat = np.zeros(EXPANDED_FEATURE_DIM, dtype=np.float32)
                            hand_detected = False

                sequence_164d.append(feat)

                progress = frame_num / 30.0
                hud_frame = draw_hud(
                    frame,
                    clean_label,
                    take_idx,
                    num_samples,
                    f"RECORDING ({frame_num}/30)",
                    progress,
                    status_color=(0, 255, 120),
                    hand_detected=hand_detected,
                )

                if not headless and not mock:
                    cv2.imshow(window_name, hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        logger.info("Recording aborted by user.")
                        return saved_files
                elif mock:
                    time.sleep(0.01)

            # -----------------------------------------------------------
            # Phase 3: Save Sequence NPZ
            # -----------------------------------------------------------
            seq_array = np.array(sequence_164d, dtype=np.float32)  # Shape: (30, 164)
            timestamp = int(time.time() * 1000)
            file_name = f"custom_{clean_label}_{timestamp}.npz"
            file_path = output_dir / file_name

            np.savez_compressed(
                file_path,
                features=seq_array,
                label=clean_label,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            saved_files.append(file_path)
            logger.info(f"\u2713 Take {take_idx}/{num_samples} saved -> {file_path.name} (Shape: {seq_array.shape})")

            # -----------------------------------------------------------
            # Phase 4: Inter-Take Cooldown (1.2 seconds)
            # -----------------------------------------------------------
            if take_idx < num_samples:
                pause_start = time.time()
                while time.time() - pause_start < 1.2:
                    if mock:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        assert cap is not None
                        ret, raw_frame = cap.read()
                        frame = cv2.flip(raw_frame, 1) if (ret and raw_frame is not None) else np.zeros((480, 640, 3), dtype=np.uint8)

                    hud_frame = draw_hud(
                        frame,
                        clean_label,
                        take_idx,
                        num_samples,
                        "SAVED! RESET HANDS...",
                        1.0,
                        status_color=(100, 200, 255),
                        hand_detected=False,
                    )
                    draw_centered_text(hud_frame, "Saved! Reset Hands", font_scale=1.0, color=(100, 200, 255))

                    if not headless and not mock:
                        cv2.imshow(window_name, hud_frame)
                        key = cv2.waitKey(30) & 0xFF
                        if key in (27, ord("q")):
                            return saved_files
                    else:
                        time.sleep(0.05)

    finally:
        if cap is not None:
            cap.release()
        if not headless and not mock:
            cv2.destroyAllWindows()

    logger.info("=" * 60)
    logger.info(f"Done! Successfully captured {len(saved_files)} takes for '{clean_label}'.")
    logger.info("=" * 60)
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SETU Custom Sign Dataset Recorder (OpenCV + MediaPipe 164-D Features)"
    )
    parser.add_argument("--label", type=str, default="", help="Sign label (e.g. 'namaste', 'help')")
    parser.add_argument("--num-samples", "--takes", dest="num_samples", type=int, default=10, help="Number of samples/takes to record")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Output folder")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock generator (for testing without webcam)")

    args = parser.parse_args()

    label = args.label
    if not label:
        if args.headless or args.mock:
            label = "custom_gesture"
        else:
            try:
                label = input("\nEnter sign label to record (e.g., 'namaste', 'help'): ").strip()
            except EOFError:
                label = "custom_gesture"

    record_sign_samples(
        label=label,
        num_samples=args.num_samples,
        camera_idx=args.camera,
        countdown_sec=args.countdown,
        output_dir=Path(args.output_dir),
        headless=args.headless,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
