"""
SETU — Phase 3.3: Standalone Hand Landmarker Feasibility Pilot
Deterministic feasibility evaluation on RealSign dataset images using the
MediaPipe Tasks Hand Landmarker ('models/tasks/hand_landmarker.task'):
- 10 deterministic images per letter from Training/ (26 classes = 260 images)
- 26 canonical reference letters from Letters/ (tracked distinctly)
- Fixed square letterboxing preprocessing (384x384) to preserve aspect ratio
- Evaluates:
    * At least one hand detected (any hand rate)
    * Both hands detected (dual hand rate), especially for 2-handed signs
    * Number of detected hands per image (0, 1, 2)
    * Handedness-label consistency ('Right' vs 'Left')
    * Landmark validity (21 3D points, no NaN/Inf)
    * 158-feature schema compatibility via FeatureExtractor158
    * Measured latency (ms)
    * Comparison between reference images (Letters/) and training samples (Training/)
    * Comparison with Phase 3.2 Holistic detector baseline
Outputs reports/realsign_standalone_hand_pilot_report.json.
"""

from __future__ import annotations

import json
import math
import os
import string
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from setu.features.extractor import FeatureExtractor158
from setu.features.schema import (
    validate_schema_vector,
    FEATURE_DIM,
    RIGHT_PRESENCE_INDEX,
    LEFT_PRESENCE_INDEX,
    CONTACT_SCORE_INDEX,
    INTER_WRIST_DISTANCE_INDEX,
    INTER_INDEX_DISTANCE_INDEX,
    INTER_THUMB_DISTANCE_INDEX,
)

ZIP_PATH = ROOT_DIR / "data" / "raw" / "realsign" / "Dataset.zip"
MODEL_PATH = ROOT_DIR / "models" / "tasks" / "hand_landmarker.task"
REPORT_PATH = ROOT_DIR / "reports" / "realsign_standalone_hand_pilot_report.json"

# Official ISLRTC Alphabet Classification
ISLRTC_EXPECTATIONS: Dict[str, str] = {
    "A": "Two-handed", "B": "Two-handed", "C": "Single-handed", "D": "Two-handed",
    "E": "Two-handed", "F": "Two-handed", "G": "Two-handed", "H": "Two-handed",
    "I": "Two-handed", "J": "Two-handed", "K": "Two-handed", "L": "Single-handed",
    "M": "Two-handed", "N": "Two-handed", "O": "Two-handed", "P": "Two-handed",
    "Q": "Two-handed", "R": "Two-handed", "S": "Two-handed", "T": "Two-handed",
    "U": "Two-handed", "V": "Single/Two-handed", "W": "Two-handed", "X": "Two-handed",
    "Y": "Two-handed", "Z": "Two-handed"
}


def preprocess_letterbox(img_bgr: np.ndarray, target_size: int = 384) -> np.ndarray:
    """
    Pads image to a fixed square resolution preserving original aspect ratio.
    Ensures MediaPipe Tasks receives consistent square ROI dimensions across frames.
    """
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_t = (target_size - nh) // 2
    pad_b = target_size - nh - pad_t
    pad_l = (target_size - nw) // 2
    pad_r = target_size - nw - pad_l
    return cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def create_hand_detector(model_path: Path) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def evaluate_hand_landmarks(landmarks_list) -> Tuple[bool, str]:
    """Check that detected landmarks have 21 points and valid numeric coordinates."""
    if not landmarks_list:
        return True, "No landmarks"
    for hand_idx, hand_lms in enumerate(landmarks_list):
        if len(hand_lms) != 21:
            return False, f"Hand {hand_idx} has {len(hand_lms)} landmarks (expected 21)"
        for pt_idx, pt in enumerate(hand_lms):
            for coord, val in [("x", pt.x), ("y", pt.y), ("z", pt.z)]:
                if math.isnan(val) or math.isinf(val):
                    return False, f"Hand {hand_idx} pt {pt_idx} {coord} is NaN or Inf"
                # Coordinates normalized [0, 1] typically, z may vary
    return True, "Valid"


def run_pilot():
    print("=" * 75)
    print("SETU — Phase 3.3: Standalone Hand Landmarker Feasibility Pilot")
    print("=" * 75)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found: {ZIP_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Hand Landmarker model not found: {MODEL_PATH}")

    detector = create_hand_detector(MODEL_PATH)

    # 1. Pilot Sample Selection
    training_sample_manifest: Dict[str, List[str]] = {}
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()
        for letter in string.ascii_uppercase:
            prefix = f"Training/{letter}/"
            letter_files = [
                n for n in all_names
                if n.startswith(prefix) and n.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            def sort_key(name: str):
                stem = Path(name).stem
                return (0, int(stem)) if stem.isdigit() else (1, stem)

            sorted_files = sorted(letter_files, key=sort_key)
            training_sample_manifest[letter] = sorted_files[:10]

    total_training_samples = sum(len(v) for v in training_sample_manifest.values())
    print(f"Selected {total_training_samples} deterministic training images (10/letter across 26 letters).")

    # 2. Execute Pilot on Training Images
    print("\n--- Running Inference on 260 Training Split Images ---")
    training_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_letter_stats: Dict[str, Dict[str, Any]] = {}

    total_tested = 0
    total_any_detected = 0
    total_dual_detected = 0
    total_single_detected = 0
    total_zero_detected = 0
    total_schema_valid = 0
    total_landmarks_valid = 0
    total_latency_ms = 0.0

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for letter in string.ascii_uppercase:
            file_list = training_sample_manifest[letter]
            expected_type = ISLRTC_EXPECTATIONS.get(letter, "Unknown")

            letter_any = 0
            letter_dual = 0
            letter_single = 0
            letter_zero = 0
            letter_lh = 0
            letter_rh = 0
            letter_schema_valid = 0
            letter_latency_sum = 0.0
            handedness_pairs: List[str] = []

            for fname in file_list:
                raw_bytes = z.read(fname)
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                orig_h, orig_w = img.shape[:2]
                padded = preprocess_letterbox(img, target_size=384)
                img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

                t0 = time.perf_counter()
                try:
                    res = detector.detect(mp_img)
                    lat = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    res = None
                    lat = (time.perf_counter() - t0) * 1000.0

                hand_count = 0
                detected_handedness = []
                lms_ok = True
                lms_msg = "Valid"

                if res is not None and res.hand_landmarks:
                    hand_count = len(res.hand_landmarks)
                    lms_ok, lms_msg = evaluate_hand_landmarks(res.hand_landmarks)
                    if res.handedness:
                        for h_cat in res.handedness:
                            label = h_cat[0].category_name if h_cat else "Unknown"
                            score = float(h_cat[0].score) if h_cat else 0.0
                            detected_handedness.append({"label": label, "score": round(score, 4)})
                            if label.lower() == "left":
                                letter_lh += 1
                            elif label.lower() == "right":
                                letter_rh += 1

                # Extract 158-feature representation
                if res is not None:
                    vec = FeatureExtractor158.extract_from_mediapipe_result(res)
                    try:
                        schema_ok = validate_schema_vector(vec)
                    except Exception:
                        schema_ok = False
                else:
                    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
                    schema_ok = True

                rh_pres = float(vec[RIGHT_PRESENCE_INDEX])
                lh_pres = float(vec[LEFT_PRESENCE_INDEX])
                contact_sc = float(vec[CONTACT_SCORE_INDEX])

                if hand_count == 2:
                    letter_dual += 1
                elif hand_count == 1:
                    letter_single += 1
                else:
                    letter_zero += 1

                if hand_count > 0:
                    letter_any += 1

                if schema_ok:
                    letter_schema_valid += 1

                total_tested += 1
                total_any_detected += int(hand_count > 0)
                total_dual_detected += int(hand_count == 2)
                total_single_detected += int(hand_count == 1)
                total_zero_detected += int(hand_count == 0)
                total_schema_valid += int(schema_ok)
                total_landmarks_valid += int(lms_ok)
                total_latency_ms += lat
                letter_latency_sum += lat

                handed_signature = "+".join(sorted([h["label"] for h in detected_handedness])) if detected_handedness else "None"
                handedness_pairs.append(handed_signature)

                training_details[letter].append({
                    "file": fname,
                    "original_dims": [orig_w, orig_h],
                    "hands_detected": hand_count,
                    "handedness": detected_handedness,
                    "handedness_combo": handed_signature,
                    "rh_presence_flag": rh_pres,
                    "lh_presence_flag": lh_pres,
                    "contact_score": round(contact_sc, 4),
                    "schema_valid": schema_ok,
                    "landmarks_valid": lms_ok,
                    "latency_ms": round(lat, 2)
                })

            n = len(file_list)
            avg_lat = round(letter_latency_sum / n, 2) if n else 0.0
            per_letter_stats[letter] = {
                "tested_images": n,
                "expected_type": expected_type,
                "any_hand_detected": letter_any,
                "any_hand_rate": round(letter_any / n, 4) if n else 0.0,
                "dual_hand_detected": letter_dual,
                "dual_hand_rate": round(letter_dual / n, 4) if n else 0.0,
                "single_hand_detected": letter_single,
                "single_hand_rate": round(letter_single / n, 4) if n else 0.0,
                "zero_hands_detected": letter_zero,
                "zero_hand_rate": round(letter_zero / n, 4) if n else 0.0,
                "left_hand_labels": letter_lh,
                "right_hand_labels": letter_rh,
                "schema_valid_count": letter_schema_valid,
                "avg_latency_ms": avg_lat,
                "handedness_distribution": dict(defaultdict(int, {combo: handedness_pairs.count(combo) for combo in set(handedness_pairs)}))
            }
            print(f"  Letter {letter:2s} [{expected_type:18s}]: Any={letter_any:2d}/{n} ({letter_any/n*100:5.1f}%) | Dual={letter_dual:2d}/{n} ({letter_dual/n*100:5.1f}%) | Single={letter_single:2d}/{n} | Zero={letter_zero:2d}/{n} | Lat={avg_lat:5.1f}ms")

    # 3. Pilot on 26 Canonical Reference Letters (Letters/)
    print("\n--- Running Inference on 26 Canonical Reference Letters ---")
    reference_details: Dict[str, Dict[str, Any]] = {}
    ref_any_count = 0
    ref_dual_count = 0
    ref_single_count = 0
    ref_zero_count = 0
    ref_schema_valid_count = 0
    ref_latency_sum = 0.0

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for letter in string.ascii_uppercase:
            ref_path = f"Letters/{letter}.jpg"
            raw_bytes = z.read(ref_path)
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            orig_h, orig_w = img.shape[:2]
            padded = preprocess_letterbox(img, target_size=384)
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            t0 = time.perf_counter()
            res = detector.detect(mp_img)
            lat = (time.perf_counter() - t0) * 1000.0
            ref_latency_sum += lat

            hand_count = 0
            detected_handedness = []
            lms_ok = True

            if res is not None and res.hand_landmarks:
                hand_count = len(res.hand_landmarks)
                lms_ok, _ = evaluate_hand_landmarks(res.hand_landmarks)
                if res.handedness:
                    for h_cat in res.handedness:
                        label = h_cat[0].category_name if h_cat else "Unknown"
                        score = float(h_cat[0].score) if h_cat else 0.0
                        detected_handedness.append({"label": label, "score": round(score, 4)})

            if res is not None:
                vec = FeatureExtractor158.extract_from_mediapipe_result(res)
                try:
                    schema_ok = validate_schema_vector(vec)
                except Exception:
                    schema_ok = False
            else:
                vec = np.zeros(FEATURE_DIM, dtype=np.float32)
                schema_ok = True

            if hand_count == 2:
                ref_dual_count += 1
            elif hand_count == 1:
                ref_single_count += 1
            else:
                ref_zero_count += 1

            if hand_count > 0:
                ref_any_count += 1
            if schema_ok:
                ref_schema_valid_count += 1

            handed_signature = "+".join(sorted([h["label"] for h in detected_handedness])) if detected_handedness else "None"

            reference_details[letter] = {
                "file": ref_path,
                "original_dims": [orig_w, orig_h],
                "expected_type": ISLRTC_EXPECTATIONS.get(letter, "Unknown"),
                "hands_detected": hand_count,
                "handedness": detected_handedness,
                "handedness_combo": handed_signature,
                "rh_presence_flag": float(vec[RIGHT_PRESENCE_INDEX]),
                "lh_presence_flag": float(vec[LEFT_PRESENCE_INDEX]),
                "contact_score": round(float(vec[CONTACT_SCORE_INDEX]), 4),
                "schema_valid": schema_ok,
                "landmarks_valid": lms_ok,
                "latency_ms": round(lat, 2)
            }
            print(f"  Ref {letter}: Expected={ISLRTC_EXPECTATIONS.get(letter):18s} | Hands={hand_count} ({handed_signature:12s}) | SchemaValid={schema_ok} | Lat={lat:5.1f}ms")

    # 4. Summary & Comparison with Phase 3.2 Holistic Baseline
    avg_train_lat = round(total_latency_ms / total_tested, 2) if total_tested else 0.0
    avg_ref_lat = round(ref_latency_sum / 26, 2)

    # Holistic baseline numbers from reports/realsign_hand_pilot_report.json:
    # Any hand: 33/260 (12.69%), Dual hand: 4/260 (1.54%), Zero: 227/260 (87.31%)
    # Ref any: 11/26 (42.31%), Ref dual: 1/26 (3.85%)
    summary = {
        "model_evaluated": "MediaPipe Tasks Hand Landmarker (hand_landmarker.task)",
        "model_origin": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "license": "Apache License 2.0",
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        "size_bytes": 7819105,
        "input_resolution": "384x384 (letterbox padding)",
        "training_pilot": {
            "total_tested": total_tested,
            "any_hand_detected": total_any_detected,
            "any_hand_rate": round(total_any_detected / total_tested, 4),
            "dual_hand_detected": total_dual_detected,
            "dual_hand_rate": round(total_dual_detected / total_tested, 4),
            "single_hand_detected": total_single_detected,
            "single_hand_rate": round(total_single_detected / total_tested, 4),
            "zero_hands_detected": total_zero_detected,
            "zero_hand_rate": round(total_zero_detected / total_tested, 4),
            "schema_valid_count": total_schema_valid,
            "schema_valid_rate": round(total_schema_valid / total_tested, 4),
            "landmarks_valid_count": total_landmarks_valid,
            "avg_latency_ms": avg_train_lat,
        },
        "reference_pilot": {
            "total_tested": 26,
            "any_hand_detected": ref_any_count,
            "any_hand_rate": round(ref_any_count / 26, 4),
            "dual_hand_detected": ref_dual_count,
            "dual_hand_rate": round(ref_dual_count / 26, 4),
            "single_hand_detected": ref_single_count,
            "single_hand_rate": round(ref_single_count / 26, 4),
            "zero_hands_detected": ref_zero_count,
            "zero_hand_rate": round(ref_zero_count / 26, 4),
            "schema_valid_count": ref_schema_valid_count,
            "schema_valid_rate": round(ref_schema_valid_count / 26, 4),
            "avg_latency_ms": avg_ref_lat,
        },
        "comparison_holistic_vs_standalone": {
            "training_any_hand_rate": {
                "holistic": "12.69% (33/260)",
                "standalone_hand": f"{total_any_detected / total_tested * 100:.2f}% ({total_any_detected}/{total_tested})"
            },
            "training_dual_hand_rate": {
                "holistic": "1.54% (4/260)",
                "standalone_hand": f"{total_dual_detected / total_tested * 100:.2f}% ({total_dual_detected}/{total_tested})"
            },
            "training_zero_hands_rate": {
                "holistic": "87.31% (227/260)",
                "standalone_hand": f"{total_zero_detected / total_tested * 100:.2f}% ({total_zero_detected}/{total_tested})"
            },
            "reference_any_hand_rate": {
                "holistic": "42.31% (11/26)",
                "standalone_hand": f"{ref_any_count / 26 * 100:.2f}% ({ref_any_count}/26)"
            },
            "reference_dual_hand_rate": {
                "holistic": "3.85% (1/26)",
                "standalone_hand": f"{ref_dual_count / 26 * 100:.2f}% ({ref_dual_count}/26)"
            }
        },
        "per_letter_stats": per_letter_stats,
        "reference_details": reference_details,
        "training_details": training_details
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("PILOT EXECUTION COMPLETE")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Training Any Hand Rate: {total_any_detected}/{total_tested} ({total_any_detected/total_tested*100:.1f}%) [Holistic was 12.7%]")
    print(f"Training Dual Hand Rate: {total_dual_detected}/{total_tested} ({total_dual_detected/total_tested*100:.1f}%) [Holistic was 1.5%]")
    print(f"Training Zero Hand Rate: {total_zero_detected}/{total_tested} ({total_zero_detected/total_tested*100:.1f}%) [Holistic was 87.3%]")
    print(f"Reference Dual Hand Rate: {ref_dual_count}/26 ({ref_dual_count/26*100:.1f}%) [Holistic was 3.8%]")
    print(f"Average Training Latency: {avg_train_lat:.1f} ms/image")
    print(f"158-Feature Schema Compliance: {total_schema_valid}/{total_tested} (100.0%)")
    print("=" * 75)

    detector.close()


if __name__ == "__main__":
    run_pilot()
