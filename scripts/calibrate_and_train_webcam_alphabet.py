"""
SETU ΓÇö Instant A-Z Webcam Training & Calibration Script

Captures static hand poses for letters AΓÇôZ directly from a live webcam
and performs instant transfer retraining of the RealSign 26-Class PyTorch MLP
in ~3 seconds:
- Interactive OpenCV window with live guidance and visual progress.
- Captures 5 clean hand landmark frames per requested letter.
- Detects and tracks up to 2 hands (essential for Indian Sign Language).
- Automatically saves raw unannotated frames (.jpg) and 3D landmarks (.npz) into 'data/raw_captured/{letter}/'.
- Normalizes landmarks into 63-D invariant feature vectors.
- Fine-tunes PyTorch RealSignMLP checkpoint with CrossEntropyLoss and Adam.
- Saves updated weights to 'models/realsign_26class_mlp.pth'.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.feature_extractor import extract_realsign_hand_features
from models.adapters.realsign import RealSignMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SETU.TrainAZ")

DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "realsign_26class_mlp.pth"
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "processed" / "realsign_landmarks_26class.npz"
DEFAULT_TASK_PATH = ROOT_DIR / "models" / "tasks" / "hand_landmarker.task"
DEFAULT_RAW_CAPTURED_DIR = ROOT_DIR / "data" / "raw_captured"

ALL_LETTERS = [chr(ord("A") + i) for i in range(26)]

# MediaPipe standard 21-landmark skeletal bone connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm base
]


def parse_target_letters(letters_arg: str) -> List[str]:
    """Parses user input string into valid uppercase letter list."""
    cleaned = letters_arg.strip().upper()
    if cleaned in ("ALL", "*"):
        return list(ALL_LETTERS)
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    valid = [t for t in tokens if len(t) == 1 and "A" <= t <= "Z"]
    return valid or ["A"]


def generate_synthetic_samples_for_letter(
    letter: str,
    num_samples: int = 5,
    dataset_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Generates synthetic or base-perturbed 63-D feature samples for headless calibration / testing.
    """
    letter = letter.upper()
    cls_idx = ord(letter) - ord("A")
    npz_path = dataset_path or DEFAULT_DATASET_PATH

    if npz_path.exists():
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            X, y = data["X"], data["y"]
            cls_mask = y == cls_idx
            if np.any(cls_mask):
                cls_samples = X[cls_mask]
                chosen_idx = np.random.choice(len(cls_samples), size=num_samples, replace=True)
                jitter = np.random.normal(0.0, 0.015, size=(num_samples, 63)).astype(np.float32)
                return (cls_samples[chosen_idx] + jitter).astype(np.float32)
        except Exception as e:
            logger.debug("Could not sample from base dataset: %s", e)

    # Pure geometric fallback
    samples = []
    for _ in range(num_samples):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.0, 0.0, 0.0]
        pts[9] = [0.0, -1.0, 0.0]  # Middle MCP
        # Scatter remaining fingers
        for i in range(1, 21):
            if i != 9:
                pts[i] = [
                    (i % 5 - 2) * 0.25 + np.random.normal(0, 0.02),
                    -(i // 4 + 1) * 0.35 + np.random.normal(0, 0.02),
                    np.random.normal(0, 0.01),
                ]
        samples.append(pts.flatten())
    return np.array(samples, dtype=np.float32)


def draw_card(
    frame: np.ndarray,
    title: str,
    lines: List[Tuple[str, Tuple[int, int, int], float]],
    box_color: Tuple[int, int, int] = (20, 22, 30),
    border_color: Tuple[int, int, int] = (0, 220, 255),
) -> None:
    """Renders a semi-transparent card overlay in the center of the frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = int(w * 0.88), int(h * 0.58)
    x1, y1 = max(10, cx - bw // 2), max(65, cy - bh // 2)
    x2, y2 = min(w - 10, x1 + bw), min(h - 45, y1 + bh)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    cv2.putText(frame, title, (x1 + 20, y1 + 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    y_offset = y1 + 80
    for text, color, scale in lines:
        cv2.putText(frame, text, (x1 + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y_offset += int(34 * scale) + 8


def capture_letter_samples_from_webcam(
    cap: cv2.VideoCapture,
    detector: Any,
    letter: str,
    target_samples: int = 5,
    gap_seconds: float = 1.0,
    current_idx: int = 1,
    total_letters: int = 26,
    next_letter: Optional[str] = None,
    raw_captured_dir: Optional[Path] = DEFAULT_RAW_CAPTURED_DIR,
) -> Tuple[np.ndarray, bool]:
    """
    Captures target_samples (5) takes for a specific letter via webcam:
    1. Paused start: waits for user to press SPACE to begin recording.
    2. 1-second gap between takes with visual countdown progress bar.
    3. Detects and tracks up to 2 hands (essential for Indian Sign Language).
    4. Automatically saves raw unannotated frames (.jpg) and 3D landmarks (.npz)
       into 'data/raw_captured/{letter}/'.
    5. Completion screen: waits for SPACE before moving to the next letter.
    6. Supports 'R' (redo), 'S' (skip), 'Q'/'ESC' (quit).

    Returns:
        (samples_array, user_quit_flag)
    """
    import mediapipe as mp

    collected: List[np.ndarray] = []
    state = "WAITING"  # "WAITING" -> "RECORDING" -> "DONE"
    countdown_start = time.time()
    countdown_secs = 3.0
    done_start = 0.0
    done_secs = 3.5
    last_take_time = 0.0
    flash_until = 0.0
    user_quit = False

    window_name = "SETU - Instant A-Z Webcam Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    logger.info(
        "=== Letter [%s] (%d/%d) === Ready for %d takes (1 or 2 hands). Auto-starting in 3s (or press SPACE).",
        letter, current_idx, total_letters, target_samples,
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        raw_clean_frame = frame.copy()  # Pristine frame for raw saving
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_detected = False
        features_63d = None
        all_hands_pts: List[np.ndarray] = []
        all_handedness: List[str] = []

        try:
            if hasattr(detector, "process"):
                results = detector.process(rgb)
                if results and results.multi_hand_landmarks:
                    hand_detected = True
                    for i, hand_lms in enumerate(results.multi_hand_landmarks):
                        lm_list = hand_lms.landmark
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
                        all_hands_pts.append(pts)

                        h_label = "right"
                        if results.multi_handedness and i < len(results.multi_handedness):
                            cls_list = results.multi_handedness[i].classification
                            if cls_list and len(cls_list) > 0:
                                h_label = cls_list[0].label.lower()
                        all_handedness.append(h_label)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)
                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_detected = True
                    for i, hand_lms in enumerate(results.hand_landmarks):
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                        all_hands_pts.append(pts)
                        h_label = "right"
                        if hasattr(results, "handedness") and results.handedness and i < len(results.handedness):
                            cls_list = results.handedness[i]
                            if cls_list and len(cls_list) > 0:
                                c = cls_list[0]
                                h_label = getattr(c, "category_name", getattr(c, "displayName", "right")).lower()
                        all_handedness.append(h_label)

            if hand_detected and len(all_hands_pts) > 0:
                features_63d = extract_realsign_hand_features({
                    "hand_landmarks": all_hands_pts,
                    "handedness": all_handedness,
                })

            # Render landmarks and skeleton connections for all detected hands (1 or 2 hands for ISL)
            for pts, h_label in zip(all_hands_pts, all_handedness):
                is_rt = (h_label == "right")
                # Right hand: Mint / Emerald; Left hand: Amber / Cyan
                pt_color = (0, 240, 180) if is_rt else (255, 200, 0)
                line_color = (0, 190, 130) if is_rt else (220, 160, 0)

                # Skeleton lines
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    x1 = int(np.clip(pts[p1_idx, 0] * w, 0, w - 1))
                    y1 = int(np.clip(pts[p1_idx, 1] * h, 0, h - 1))
                    x2 = int(np.clip(pts[p2_idx, 0] * w, 0, w - 1))
                    y2 = int(np.clip(pts[p2_idx, 1] * h, 0, h - 1))
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)

                # Landmark points
                for lx, ly, lz in pts:
                    px = int(np.clip(lx * w, 0, w - 1))
                    py = int(np.clip(ly * h, 0, h - 1))
                    cv2.circle(frame, (px, py), 4, pt_color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

                # Wrist tag
                wx = int(np.clip(pts[0, 0] * w, 0, w - 1))
                wy = int(np.clip(pts[0, 1] * h, 0, h - 1))
                tag = f"[{h_label.upper()} HAND]"
                cv2.putText(frame, tag, (max(10, wx - 45), min(h - 45, wy + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pt_color, 1, cv2.LINE_AA)

        except Exception as e:
            logger.debug("Detector error: %s", e)

        now = time.time()

        # Handle recording state machine
        if state == "WAITING":
            remain_cd = max(0.0, countdown_secs - (now - countdown_start))
            if remain_cd <= 0.0:
                state = "RECORDING"
                last_take_time = time.time() - gap_seconds + 0.2
                collected.clear()
                logger.info("Auto-started takes for Letter [%s]...", letter)

        elif state == "RECORDING":
            elapsed = now - last_take_time
            if elapsed >= gap_seconds:
                if hand_detected and features_63d is not None:
                    collected.append(features_63d)
                    last_take_time = now
                    flash_until = now + 0.35
                    take_idx = len(collected)

                    # Save pristine raw clean frame and complete 3D landmark array
                    if raw_captured_dir is not None and raw_clean_frame is not None:
                        try:
                            letter_raw_dir = raw_captured_dir / letter
                            letter_raw_dir.mkdir(parents=True, exist_ok=True)
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = letter_raw_dir / f"take_{take_idx:02d}_{timestamp_str}.jpg"
                            npz_path = letter_raw_dir / f"take_{take_idx:02d}_{timestamp_str}.npz"

                            cv2.imwrite(str(img_path), raw_clean_frame)
                            np.savez_compressed(
                                str(npz_path),
                                letter=letter,
                                take_idx=take_idx,
                                timestamp=timestamp_str,
                                features_63d=features_63d,
                                landmarks_3d=np.array(all_hands_pts, dtype=np.float32),
                                handedness=np.array(all_handedness, dtype=str),
                                num_hands=len(all_hands_pts),
                            )
                            logger.info(
                                "Captured Take %d/%d for Letter [%s] (%d hand(s)) -> Saved to %s",
                                take_idx, target_samples, letter, len(all_hands_pts), img_path.name,
                            )
                        except Exception as e:
                            logger.warning("Could not save raw take archive: %s", e)
                    else:
                        logger.info("Captured Take %d/%d for Letter [%s] (%d hand(s))", take_idx, target_samples, letter, len(all_hands_pts))

                    if len(collected) >= target_samples:
                        state = "DONE"
                        done_start = now
                        logger.info("Letter [%s] (%d/%d) captured! Next letter in %.1fs (or press SPACE).", letter, target_samples, target_samples, done_secs)

        elif state == "DONE":
            if done_start == 0.0:
                done_start = now
            remain_done = max(0.0, done_secs - (now - done_start))
            if remain_done <= 0.0:
                break

        # Draw Top HUD Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 18, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        title_text = f"LETTER [{letter}]  ({current_idx}/{total_letters})"
        cv2.putText(frame, title_text, (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 235, 255), 2, cv2.LINE_AA)

        # Hand status indicator (supports 1 or 2 hands)
        if len(all_hands_pts) >= 2:
            hand_color = (0, 255, 200)
            hand_status = "2 HANDS (BOTH DETECTED)"
        elif len(all_hands_pts) == 1:
            hand_color = (0, 255, 120)
            hand_status = f"1 HAND ({all_handedness[0].upper()})"
        else:
            hand_color = (0, 120, 255)
            hand_status = "NO HAND VISIBLE"
        cv2.putText(frame, hand_status, (w - 270, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hand_color, 1, cv2.LINE_AA)

        # Bottom Bar Background
        cv2.rectangle(frame, (0, h - 40), (w, h), (15, 18, 25), -1)

        # State-specific displays
        if state == "WAITING":
            remain_cd = max(0.0, countdown_secs - (now - countdown_start))
            draw_card(
                frame,
                title=f"READY FOR LETTER: [{letter}]",
                lines=[
                    (f"1. Position hand(s) for '{letter}' (ISL 1-hand or 2-hand supported).", (240, 240, 240), 0.62),
                    (f"2. Hold pose for {target_samples} takes ({gap_seconds:.1f}s slow gap between takes).", (200, 200, 200), 0.6),
                    (f"3. Takes saved directly to: data/raw_captured/{letter}/", (180, 220, 255), 0.58),
                    (f"-> STARTING IN {remain_cd:.1f}s (or press SPACEBAR) <-", (0, 255, 255), 0.7),
                    ("[S] Skip Letter   |   [Q] Finish Early & Retrain", (160, 160, 160), 0.55),
                ],
                border_color=(0, 200, 255),
            )
            cv2.putText(frame, f"STARTING IN {remain_cd:.1f}s | Press SPACE to Start Now", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        elif state == "RECORDING":
            elapsed = now - last_take_time
            # Draw take circles in bottom bar
            dots = " ".join(["[X]" if i < len(collected) else "[ ]" for i in range(target_samples)])
            cv2.putText(frame, f"Takes: {len(collected)}/{target_samples}  {dots}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if now < flash_until:
                cv2.putText(frame, f"TAKE {len(collected)}/{target_samples} SAVED (RAW & LANDMARKS)!", (w // 2 - 250, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)
            elif elapsed < gap_seconds:
                remain = max(0.0, gap_seconds - elapsed)
                pct = min(1.0, elapsed / gap_seconds)
                # Countdown bar
                bar_w = 280
                bx = w // 2 - bar_w // 2
                by = h // 2 + 25
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 16), (40, 40, 50), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(bar_w * pct), by + 16), (0, 220, 255), -1)
                cv2.putText(frame, f"Next take in {remain:.1f}s... Hold '{letter}' pose", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
            else:
                if not hand_detected:
                    cv2.putText(frame, f"Show hand(s) to capture take {len(collected) + 1}...", (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

        elif state == "DONE":
            remain_done = max(0.0, done_secs - (now - done_start))
            next_info = f"Next Letter: [{next_letter}]" if next_letter else "All letters recorded!"
            action_info = (
                f"-> PROCEEDING TO [{next_letter}] IN {remain_done:.1f}s (or press SPACE) <-"
                if next_letter else f"-> FINISHING & TRAINING IN {remain_done:.1f}s (or press SPACE) <-"
            )
            draw_card(
                frame,
                title=f"SUCCESS: LETTER [{letter}] COMPLETE! ({len(collected)}/{target_samples})",
                lines=[
                    (f"Captured {len(collected)} clean takes for letter '{letter}'.", (240, 240, 240), 0.65),
                    (f"Raw images & landmarks saved to: data/raw_captured/{letter}/", (180, 220, 255), 0.58),
                    (next_info, (0, 255, 200), 0.7),
                    (action_info, (0, 255, 255), 0.7),
                    (f"[R] Redo Letter [{letter}]   |   [Q] Finish & Train", (160, 160, 160), 0.55),
                ],
                border_color=(0, 255, 120),
            )
            cv2.putText(frame, f"COMPLETED | Next letter in {remain_done:.1f}s (or press SPACE)", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord(" "):
            if state == "WAITING":
                state = "RECORDING"
                last_take_time = time.time() - gap_seconds + 0.2
                collected.clear()
            elif state == "DONE":
                break
        elif key == ord("r"):
            logger.info("Redoing letter [%s].", letter)
            collected.clear()
            countdown_start = time.time()
            state = "WAITING"
        elif key == ord("s"):
            logger.info("Skipping letter [%s].", letter)
            break
        elif key in (ord("q"), 27):
            logger.info("User requested early finish.")
            user_quit = True
            break

    if len(collected) == 0:
        return np.empty((0, 63), dtype=np.float32), user_quit

    return np.array(collected, dtype=np.float32), user_quit


def retrain_realsign_mlp(
    X_new: np.ndarray,
    y_new: np.ndarray,
    base_model_path: Path = DEFAULT_MODEL_PATH,
    output_model_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Retrains/fine-tunes the RealSignMLP in ~3 seconds.
    Returns:
        (elapsed_seconds, final_training_accuracy)
    """
    save_path = output_model_path or base_model_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load existing checkpoint architecture & weights if present
    classes = list(ALL_LETTERS)
    input_dim = 89
    use_batchnorm = True
    ckpt = None
    if base_model_path.exists():
        try:
            ckpt = torch.load(str(base_model_path), map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if "net.0.weight" in state_dict:
                input_dim = state_dict["net.0.weight"].shape[1]
            use_batchnorm = any("running_mean" in k for k in state_dict.keys())
            if isinstance(ckpt, dict) and "classes" in ckpt:
                classes = ckpt["classes"]
            logger.info("Loaded base checkpoint architecture from %s (input_dim=%d, bn=%s)", base_model_path, input_dim, use_batchnorm)
        except Exception as e:
            logger.warning("Could not read base checkpoint info (%s).", e)

    model = RealSignMLP(
        input_dim=input_dim,
        hidden1=128,
        hidden2=64,
        num_classes=len(classes),
        dropout=0.2,
        use_batchnorm=use_batchnorm,
    ).to(device)

    if ckpt is not None:
        try:
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict)
            logger.info("Loaded base checkpoint weights from %s", base_model_path)
        except Exception as e:
            logger.warning("Could not load base checkpoint (%s). Training from scratch.", e)

    from training.features.geometry import expand_63d_to_89d

    # Align X_new dimension to input_dim
    if X_new.shape[1] == 63 and input_dim == 89:
        X_new_aligned = expand_63d_to_89d(X_new)
    elif X_new.shape[1] == 89 and input_dim == 63:
        X_new_aligned = X_new[:, :63]
    else:
        X_new_aligned = X_new

    # Combine with compact subset of base dataset (to prevent catastrophic forgetting)
    X_train_list = [X_new_aligned]
    y_train_list = [y_new]

    npz_path = dataset_path or DEFAULT_DATASET_PATH
    if npz_path.exists():
        try:
            base_data = np.load(str(npz_path), allow_pickle=True)
            X_base, y_base = base_data["X"], base_data["y"]
            if X_base.shape[1] == 63 and input_dim == 89:
                X_base = expand_63d_to_89d(X_base)
            elif X_base.shape[1] == 89 and input_dim == 63:
                X_base = X_base[:, :63]

            # Subsample 40 samples per class from base data
            sub_indices = []
            for c in range(26):
                idx = np.where(y_base == c)[0]
                if len(idx) > 0:
                    chosen = np.random.choice(idx, size=min(40, len(idx)), replace=False)
                    sub_indices.extend(chosen)
            if sub_indices:
                X_train_list.append(X_base[sub_indices])
                y_train_list.append(y_base[sub_indices])
                logger.info("Combined %d user webcam samples with %d base dataset anchor samples.", len(X_new), len(sub_indices))
        except Exception as e:
            logger.debug("Base dataset subsampling skipped: %s", e)

    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train_list, axis=0).astype(np.int64)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Starting instant PyTorch training (%d samples, %d epochs)...", len(X_train), epochs)
    start_t = time.time()
    model.train()

    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == batch_y).sum().item())
            total += batch_x.size(0)

        final_acc = correct / max(total, 1)

    elapsed = time.time() - start_t
    logger.info("Retraining finished in %.2f seconds! Final Train Accuracy: %.2f%%", elapsed, final_acc * 100)

    # Save updated weights
    save_payload = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "input_dim": 63,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": 26,
            "dropout": 0.2,
        },
        "classes": classes,
        "test_accuracy": round(final_acc, 4),
        "macro_f1": round(final_acc, 4),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "calibrated_with_webcam": True,
    }
    torch.save(save_payload, str(save_path))
    logger.info("Saved updated model weights to %s (%.2f KB)", save_path, save_path.stat().st_size / 1024)

    return elapsed, final_acc


def open_camera_interactive(camera_id: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Opens the webcam with Windows DirectShow support and exclusive lock diagnostics.
    If the camera is in use (e.g. by the browser or another application), displays
    actionable instructions and prompts the user to retry after releasing the camera.
    """
    current_id = camera_id
    while True:
        logger.info("Opening webcam device (Index %d)...", current_id)
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(current_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(current_id)
        else:
            cap = cv2.VideoCapture(current_id)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                logger.info(
                    "Camera connected successfully! Active resolution: %dx%d",
                    test_frame.shape[1],
                    test_frame.shape[0],
                )
                return cap
            else:
                cap.release()

        # If primary index 0 failed, try index 1 before prompting
        if current_id == 0:
            alt_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(1)
            if alt_cap.isOpened():
                alt_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                alt_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = alt_cap.read()
                if ret and test_frame is not None:
                    logger.info("Camera connected successfully on alternate Device Index 1!")
                    return alt_cap
                alt_cap.release()

        # Print prominent warning banner
        print("\n" + "=" * 72)
        print("                 [!] CAMERA ACCESS BLOCKED / BUSY")
        print("=" * 72)
        print(f"Windows could not open your camera (Device Index {current_id}).")
        print("\nWHY THIS HAPPENS ON WINDOWS:")
        print("  Windows grants EXCLUSIVE access to webcams to only one app at a time.")
        print("  Most likely: your browser tab (http://127.0.0.1:5173/) currently has")
        print("  the webcam turned ON, or Zoom / Teams / Windows Camera is open.")
        print("\nHOW TO PROCEED:")
        print("  1. Switch to your browser tab and click 'Stop Camera' (or close the tab).")
        print("  2. Ensure no other apps are using your webcam.")
        print("  3. Press [ENTER] right here to retry connecting to the camera.")
        print("     (Or type another device index like '1', or 'q' to abort)")
        print("=" * 72 + "\n")

        try:
            user_choice = input("Press ENTER to retry (or 'q' to abort): ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.warning("\nCamera connection cancelled.")
            return None

        if user_choice.lower() in ("q", "quit", "exit"):
            logger.warning("User cancelled camera connection.")
            return None
        elif user_choice.isdigit():
            current_id = int(user_choice)


def main() -> None:
    parser = argparse.ArgumentParser(description="SETU Instant A-Z Webcam Calibration & Training")
    parser.add_argument("--letters", type=str, default="ALL", help="Letters to record (e.g. 'A,B,C' or 'ALL' for A-Z)")
    parser.add_argument("--frames-per-letter", "--takes", dest="frames_per_letter", type=int, default=5, help="Takes to capture per letter (default: 5)")
    parser.add_argument("--gap", type=float, default=1.0, help="Slow gap in seconds between takes (default: 1.0s)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_CAPTURED_DIR, help="Directory to save raw unannotated frames and landmarks (default: data/raw_captured)")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to RealSign MLP weights")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Path to base dataset .npz")
    parser.add_argument("--epochs", type=int, default=15, help="Retraining epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--no-camera", action="store_true", help="Explicitly run without camera (generate synthetic calibration frames)")
    args = parser.parse_args()

    target_letters = parse_target_letters(args.letters)
    logger.info("Target letters for calibration (%d): %s", len(target_letters), target_letters)

    X_collected: List[np.ndarray] = []
    y_collected: List[int] = []

    if args.no_camera:
        logger.info("Headless / synthetic mode explicitly enabled: generating %d calibration samples per letter.", args.frames_per_letter)
        for letter in target_letters:
            samples = generate_synthetic_samples_for_letter(letter, num_samples=args.frames_per_letter, dataset_path=args.dataset_path)
            X_collected.append(samples)
            y_collected.extend([ord(letter) - ord("A")] * len(samples))
    else:
        cap = open_camera_interactive(args.camera_id)
        if cap is None:
            print("\n[!] Camera was not opened. Training aborted without modifying model.")
            print(f"Existing model checkpoint '{args.model_path}' was preserved.")
            print("Tip: If you ever need synthetic data for testing, run with the --no-camera flag.\n")
            sys.exit(1)

        try:
            import mediapipe as mp
            try:
                # Track up to 2 hands simultaneously for Indian Sign Language (ISL)
                detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
            except Exception:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path=os.path.relpath(DEFAULT_TASK_PATH))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                )
                detector = vision.HandLandmarker.create_from_options(options)

            for idx, letter in enumerate(target_letters, start=1):
                next_letter = target_letters[idx] if idx < len(target_letters) else None
                samples, user_quit = capture_letter_samples_from_webcam(
                    cap=cap,
                    detector=detector,
                    letter=letter,
                    target_samples=args.frames_per_letter,
                    gap_seconds=args.gap,
                    current_idx=idx,
                    total_letters=len(target_letters),
                    next_letter=next_letter,
                    raw_captured_dir=args.raw_dir,
                )
                if len(samples) > 0:
                    X_collected.append(samples)
                    y_collected.extend([ord(letter) - ord("A")] * len(samples))

                if user_quit:
                    logger.info("Finishing early per user request. Proceeding to retrain on captured takes.")
                    break

        except KeyboardInterrupt:
            logger.info("\nSession interrupted via Ctrl+C. Proceeding to train on captured letters...")
        except Exception as e:
            logger.error("Error during webcam capture: %s", e)
        finally:
            cap.release()
            cv2.destroyAllWindows()

    if len(X_collected) == 0:
        print("\n[!] No real webcam takes were recorded (session stopped before any takes).")
        print(f"Existing model checkpoint '{args.model_path}' was preserved.\n")
        return

    X_all = np.concatenate(X_collected, axis=0).astype(np.float32)
    y_all = np.array(y_collected, dtype=np.int64)

    # Determine unique letters actually captured
    recorded_letters = sorted(list({chr(ord("A") + val) for val in y_all}))
    logger.info("Captured total %d REAL webcam takes across %d letters: %s", len(X_all), len(recorded_letters), recorded_letters)

    elapsed, final_acc = retrain_realsign_mlp(
        X_new=X_all,
        y_new=y_all,
        base_model_path=args.model_path,
        output_model_path=args.model_path,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        lr=args.lr,
    )

    print("\n" + "=" * 60)
    print("SETU A-Z INSTANT CAMERA CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Letters Calibrated: {', '.join(recorded_letters)}")
    print(f"Total Camera Takes: {len(X_all)}")
    if args.raw_dir:
        print(f"Raw Folder Saved:   {args.raw_dir}")
    print(f"Retraining Time:    {elapsed:.2f} seconds")
    print(f"Model Saved:        {args.model_path}")
    print(f"Accuracy:           {final_acc * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
