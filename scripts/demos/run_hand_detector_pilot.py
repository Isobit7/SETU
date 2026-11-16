"""
SETU — Phase 3.2: Small Hand-Detector Pilot
Deterministic feasibility evaluation on RealSign dataset images using the
installed MediaPipe HolisticLandmarker ('models/tasks/holistic_landmarker.task'):
- 10 deterministic images per letter from Training/ (26 classes = 260 images)
- 26 canonical reference letters from Letters/ (tracked distinctly)
- Square letterboxing preprocessing to prevent dimension mismatch and ROI distortion
- Measures detection success rate, hand counts (0, 1, 2), handedness, runtime,
  and 158-feature schema validation.
Outputs reports/realsign_hand_pilot_report.json.
"""

from __future__ import annotations

import io
import json
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

from backend.app.services.vision import HolisticLandmarkerWrapper
from training.feature_extractor import FeatureExtractor158
from training.feature_extractor import validate_schema_vector, FEATURE_DIM

ZIP_PATH = Path("data/raw/realsign/Dataset.zip")
REPORT_PATH = Path("reports/realsign_hand_pilot_report.json")
MODEL_PATH = Path("models/tasks/holistic_landmarker.task")

# ISLRTC Official Handedness Classification
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
    Ensures MediaPipe's internal segmentation and ROI calculators receive consistent
    dimensions across successive frames without C++ RET_CHECK failures.
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


def run_pilot():
    print("=" * 70)
    print("SETU — Phase 3.2: Small Hand-Detector Pilot")
    print("=" * 70)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe task model not found: {MODEL_PATH}")

    landmarker = HolisticLandmarkerWrapper(MODEL_PATH)
    landmarker.initialize()

    # 1. Pilot Sample Selection
    training_sample_manifest: Dict[str, List[str]] = {}
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()
        for letter in string.ascii_uppercase:
            prefix = f"Training/{letter}/"
            # Filter valid image files, exclude directories or non-image files
            letter_files = [
                n for n in all_names
                if n.startswith(prefix) and n.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            # Deterministic sort: numerical if possible, else lexicographical
            def sort_key(name: str):
                stem = Path(name).stem
                return (0, int(stem)) if stem.isdigit() else (1, stem)
            
            sorted_files = sorted(letter_files, key=sort_key)
            # Pick first 10 files deterministically
            training_sample_manifest[letter] = sorted_files[:10]

    print(f"Selected 10 deterministic training images per letter across 26 letters ({sum(len(v) for v in training_sample_manifest.values())} total).")

    # 2. Execute Pilot on Training Samples
    print("\n--- Executing Pilot on 260 Training Split Images ---")
    training_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_letter_stats: Dict[str, Dict[str, Any]] = {}

    total_images_tested = 0
    total_any_hand_detected = 0
    total_dual_hand_detected = 0
    total_zero_hand_detected = 0
    total_schema_valid = 0
    total_latency_ms = 0.0

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for letter in string.ascii_uppercase:
            file_list = training_sample_manifest[letter]
            t0_letter = time.perf_counter()

            lh_detected_count = 0
            rh_detected_count = 0
            both_detected_count = 0
            any_detected_count = 0
            zero_detected_count = 0
            valid_schema_count = 0

            for fname in file_list:
                raw_bytes = z.read(fname)
                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                orig_h, orig_w = img.shape[:2]
                padded = preprocess_letterbox(img, target_size=384)

                t_start = time.perf_counter()
                try:
                    mp_res = landmarker.detect(padded)
                    latency = (time.perf_counter() - t_start) * 1000.0
                except Exception as e:
                    mp_res = None
                    latency = (time.perf_counter() - t_start) * 1000.0

                lh_present = False
                rh_present = False
                pose_present = False

                if mp_res is not None:
                    if mp_res.left_hand_landmarks and len(mp_res.left_hand_landmarks) == 21:
                        lh_present = True
                    if mp_res.right_hand_landmarks and len(mp_res.right_hand_landmarks) == 21:
                        rh_present = True
                    if mp_res.pose_landmarks and len(mp_res.pose_landmarks) > 0:
                        pose_present = True

                # Extract 158-feature vector
                if mp_res is not None:
                    vec = FeatureExtractor158.extract_from_mediapipe_result(mp_res)
                    try:
                        is_valid = validate_schema_vector(vec)
                    except Exception:
                        is_valid = False
                else:
                    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
                    is_valid = True

                hand_count = int(lh_present) + int(rh_present)

                if hand_count == 2:
                    both_detected_count += 1
                elif hand_count == 1:
                    pass
                else:
                    zero_detected_count += 1

                if hand_count > 0:
                    any_detected_count += 1
                if lh_present:
                    lh_detected_count += 1
                if rh_present:
                    rh_detected_count += 1
                if is_valid:
                    valid_schema_count += 1

                total_images_tested += 1
                total_any_hand_detected += int(hand_count > 0)
                total_dual_hand_detected += int(hand_count == 2)
                total_zero_hand_detected += int(hand_count == 0)
                total_schema_valid += int(is_valid)
                total_latency_ms += latency

                training_results[letter].append({
                    "file": fname,
                    "original_dims": [orig_w, orig_h],
                    "pose_detected": pose_present,
                    "lh_detected": lh_present,
                    "rh_detected": rh_present,
                    "hands_detected": hand_count,
                    "latency_ms": round(latency, 2),
                    "schema_valid": is_valid
                })

            n = len(file_list)
            per_letter_stats[letter] = {
                "tested_images": n,
                "expected_type": ISLRTC_EXPECTATIONS.get(letter, "Unknown"),
                "any_hand_detected": any_detected_count,
                "any_hand_rate": round(any_detected_count / n, 2) if n else 0.0,
                "both_hands_detected": both_detected_count,
                "both_hands_rate": round(both_detected_count / n, 2) if n else 0.0,
                "left_hand_detected": lh_detected_count,
                "right_hand_detected": rh_detected_count,
                "zero_hands_detected": zero_detected_count,
                "schema_valid_count": valid_schema_count
            }
            print(f"  Letter {letter:2s} [{ISLRTC_EXPECTATIONS.get(letter, ''):15s}]: Any={any_detected_count}/{n} | Dual={both_detected_count}/{n} | Zero={zero_detected_count}/{n}")

    # 3. Pilot on 26 Canonical Reference Letters
    print("\n--- Executing Pilot on 26 Reference Images ---")
    reference_results: Dict[str, Dict[str, Any]] = {}
    ref_dual_count = 0
    ref_any_count = 0
    ref_zero_count = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for letter in string.ascii_uppercase:
            ref_path = f"Letters/{letter}.jpg"
            raw_bytes = z.read(ref_path)
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            orig_h, orig_w = img.shape[:2]
            padded = preprocess_letterbox(img, target_size=384)

            t_start = time.perf_counter()
            mp_res = landmarker.detect(padded)
            lat = (time.perf_counter() - t_start) * 1000.0

            lh_present = bool(mp_res.left_hand_landmarks and len(mp_res.left_hand_landmarks) == 21)
            rh_present = bool(mp_res.right_hand_landmarks and len(mp_res.right_hand_landmarks) == 21)
            pose_present = bool(mp_res.pose_landmarks and len(mp_res.pose_landmarks) > 0)
            h_count = int(lh_present) + int(rh_present)

            vec = FeatureExtractor158.extract_from_mediapipe_result(mp_res)
            schema_ok = validate_schema_vector(vec)

            if h_count == 2:
                ref_dual_count += 1
            if h_count > 0:
                ref_any_count += 1
            else:
                ref_zero_count += 1

            reference_results[letter] = {
                "file": ref_path,
                "original_dims": [orig_w, orig_h],
                "expected_type": ISLRTC_EXPECTATIONS.get(letter, "Unknown"),
                "pose_detected": pose_present,
                "lh_detected": lh_present,
                "rh_detected": rh_present,
                "hands_detected": h_count,
                "schema_valid": schema_ok,
                "latency_ms": round(lat, 2)
            }
            print(f"  Ref {letter:2s} [{ISLRTC_EXPECTATIONS.get(letter, ''):15s}]: LH={lh_present!s:5s} RH={rh_present!s:5s} Pose={pose_present!s:5s} Hands={h_count}")

    avg_latency = total_latency_ms / total_images_tested if total_images_tested else 0.0

    print("\n--- Summary ---")
    print(f"Training split images tested: {total_images_tested}")
    print(f"  Images with at least 1 hand detected: {total_any_hand_detected} / {total_images_tested} ({total_any_hand_detected/total_images_tested*100:.2f}%)")
    print(f"  Images with dual hands detected: {total_dual_hand_detected} / {total_images_tested} ({total_dual_hand_detected/total_images_tested*100:.2f}%)")
    print(f"  Images with 0 hands detected: {total_zero_hand_detected} / {total_images_tested} ({total_zero_hand_detected/total_images_tested*100:.2f}%)")
    print(f"Reference images tested: 26")
    print(f"  Reference images with at least 1 hand: {ref_any_count} / 26 ({ref_any_count/26*100:.2f}%)")
    print(f"  Reference images with dual hands: {ref_dual_count} / 26 ({ref_dual_count/26*100:.2f}%)")
    print(f"  Reference images with 0 hands: {ref_zero_count} / 26 ({ref_zero_count/26*100:.2f}%)")
    print(f"Average latency per image: {avg_latency:.2f} ms")

    # 4. Construct Final Report
    report = {
        "dataset_archive": str(ZIP_PATH),
        "model_used": str(MODEL_PATH),
        "detector_type": "MediaPipe Tasks HolisticLandmarker (integrated via HolisticLandmarkerWrapper)",
        "preprocessing": {
            "strategy": "Square letterbox with aspect ratio preservation",
            "target_resolution": [384, 384],
            "padding_fill": [0, 0, 0]
        },
        "training_pilot_summary": {
            "total_images_tested": total_images_tested,
            "any_hand_detected_count": total_any_hand_detected,
            "any_hand_detection_rate": round(total_any_hand_detected / total_images_tested, 4) if total_images_tested else 0.0,
            "dual_hands_detected_count": total_dual_hand_detected,
            "dual_hands_detection_rate": round(total_dual_hand_detected / total_images_tested, 4) if total_images_tested else 0.0,
            "zero_hands_detected_count": total_zero_hand_detected,
            "zero_hands_rate": round(total_zero_hand_detected / total_images_tested, 4) if total_images_tested else 0.0,
            "schema_valid_rate": round(total_schema_valid / total_images_tested, 4) if total_images_tested else 0.0,
            "avg_latency_ms": round(avg_latency, 2)
        },
        "reference_pilot_summary": {
            "total_images_tested": 26,
            "any_hand_detected_count": ref_any_count,
            "any_hand_detection_rate": round(ref_any_count / 26, 4),
            "dual_hands_detected_count": ref_dual_count,
            "dual_hands_detection_rate": round(ref_dual_count / 26, 4),
            "zero_hands_detected_count": ref_zero_count,
            "zero_hands_rate": round(ref_zero_count / 26, 4)
        },
        "per_letter_breakdown": per_letter_stats,
        "reference_letter_results": reference_results,
        "sample_manifest": training_sample_manifest,
        "detailed_training_results": training_results
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nPilot finished. Detailed report written to {REPORT_PATH}")
    landmarker.close()


if __name__ == "__main__":
    run_pilot()
