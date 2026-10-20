"""
SETU - Controlled Alphabet Test Runner for RF-UV-11 Pretrained Model
Runs a structured evaluation of target alphabet letters across both:
1. Authentic two-handed Indian Sign Language (ISL) gestures
2. Single-handed gestures (ASL / RF-UV-11 training space)

Records: Expected Label, Predicted Label, Confidence, Hands Detected, Latency, and Outcome.
Saves structured results to reports/controlled_alphabet_test.json.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from setu.vision.camera import CameraReader
from setu.vision.landmarks import HolisticLandmarkerWrapper
from setu.models.rf_uv11_adapter import RFUV11Adapter

# Target test cases contrasting authentic ISL (two-handed) vs single-handed gestures
TARGET_ALPHABET_TESTS = [
    {
        "letter": "A",
        "isl_type": "Two-handed ISL",
        "description": "Dominant hand closed fist/thumb up, non-dominant hand index touches thumb."
    },
    {
        "letter": "B",
        "isl_type": "Two-handed ISL",
        "description": "Both hands form loops with index + thumb, touching together like 'B'."
    },
    {
        "letter": "C",
        "isl_type": "Single-handed ISL / ASL",
        "description": "Dominant hand curved in a C-shape."
    },
    {
        "letter": "D",
        "isl_type": "Two-handed ISL",
        "description": "Non-dominant index pointing up, dominant hand forms loop against it."
    },
    {
        "letter": "L",
        "isl_type": "Single-handed ISL / ASL",
        "description": "Index finger vertical, thumb extended horizontally at 90 degrees."
    },
    {
        "letter": "V",
        "isl_type": "Single-handed / ASL",
        "description": "Index and middle fingers extended in a V-shape."
    }
]


def run_controlled_test(
    camera_index: int = 0,
    seconds_per_sign: float = 3.5,
    output_report: Path = Path("reports/controlled_alphabet_test.json")
) -> Dict[str, Any]:
    print("=" * 70)
    print("SETU — Controlled Alphabet Authenticity & Recognition Test")
    print("=" * 70)
    print("This script evaluates whether the RF-UV-11 model accurately recognizes")
    print("authentic Indian Sign Language alphabet gestures versus single-handed signs.")
    print("-" * 70)

    adapter = RFUV11Adapter()
    landmarker = HolisticLandmarkerWrapper()
    landmarker.initialize()
    camera = CameraReader(camera_index=camera_index, mirror_preview=True)
    camera.open()

    results: List[Dict[str, Any]] = []

    try:
        for idx, test_case in enumerate(TARGET_ALPHABET_TESTS, 1):
            letter = test_case["letter"]
            isl_type = test_case["isl_type"]
            desc = test_case["description"]

            print(f"\n[{idx}/{len(TARGET_ALPHABET_TESTS)}] Target Letter: '{letter}' ({isl_type})")
            print(f"       Instructions: {desc}")
            print("       Preparing... Sign now! (Recording for {:.1f}s)".format(seconds_per_sign))

            # Warm-up countdown (1.5 seconds)
            t_warmup = time.perf_counter()
            while time.perf_counter() - t_warmup < 1.5:
                success, raw, preview = camera.read_frame()
                if preview is not None:
                    h, w = preview.shape[:2]
                    cv2.putText(
                        preview, f"Get Ready: Sign '{letter}' ({isl_type})",
                        (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2
                    )
                    cv2.imshow("SETU Controlled Alphabet Test", preview)
                    cv2.waitKey(10)

            # Sampling window
            t_start = time.perf_counter()
            frame_preds = []
            frame_confs = []
            detected_hands_count = []
            latencies = []

            while time.perf_counter() - t_start < seconds_per_sign:
                t0 = time.perf_counter()
                success, raw, preview = camera.read_frame()
                if not success or raw is None:
                    continue

                res = landmarker.detect(raw)
                t_detect = (time.perf_counter() - t0) * 1000

                # Count detected hands
                num_hands = 0
                active_hand = None
                if res.right_hand_landmarks:
                    num_hands += 1
                    active_hand = res.right_hand_landmarks
                if res.left_hand_landmarks:
                    num_hands += 1
                    if active_hand is None:
                        active_hand = res.left_hand_landmarks

                detected_hands_count.append(num_hands)

                pred_letter = None
                conf = 0.0
                if active_hand and len(active_hand) == 21:
                    t1 = time.perf_counter()
                    feat63 = adapter.extract_63_features(active_hand)
                    pred_letter, conf, _ = adapter.predict_raw(feat63)
                    t_infer = (time.perf_counter() - t1) * 1000
                    latencies.append(t_infer)
                    frame_preds.append(pred_letter)
                    frame_confs.append(conf)

                # Overlay feedback
                h, w = preview.shape[:2]
                cv2.rectangle(preview, (0, 0), (w, 80), (20, 25, 30), -1)
                cv2.putText(
                    preview, f"Testing Target: '{letter}' ({isl_type})",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                )
                cv2.putText(
                    preview, f"Hands in View: {num_hands} | Live Pred: {pred_letter or 'None'} ({conf*100:.1f}%)",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 180) if num_hands > 0 else (0, 100, 255), 2
                )

                cv2.imshow("SETU Controlled Alphabet Test", preview)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    print("Early exit requested by user.")
                    return {"aborted": True, "completed_tests": results}

            # Tally metrics for this letter
            if frame_preds:
                from collections import Counter
                pred_counts = Counter(frame_preds)
                top_pred, top_count = pred_counts.most_common(1)[0]
                matching_confs = [c for p, c in zip(frame_preds, frame_confs) if p == top_pred]
                avg_conf = float(np.mean(matching_confs))
                mean_hands = float(np.mean(detected_hands_count))
                correct = (top_pred == letter)
            else:
                top_pred = "NO_HAND_DETECTED"
                avg_conf = 0.0
                mean_hands = 0.0
                correct = False

            test_result = {
                "letter": letter,
                "isl_type": isl_type,
                "description": desc,
                "expected_label": letter,
                "predicted_label": top_pred,
                "confidence": round(avg_conf, 4),
                "correct": correct,
                "avg_hands_detected": round(mean_hands, 2),
                "total_frames_sampled": len(detected_hands_count),
                "hand_tracked_frames": len(frame_preds),
                "mean_infer_latency_ms": round(float(np.mean(latencies)), 3) if latencies else None
            }
            results.append(test_result)
            status_str = "PASS" if correct else "FAIL"
            print(f"       -> Result: {status_str} | Expected: '{letter}' | Predicted: '{top_pred}' | Conf: {avg_conf*100:.1f}% | Avg Hands: {mean_hands:.1f}")

    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

    output_report.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "RF-UV-11 Pretrained (Apache-2.0)",
        "model_architecture": "MLP (63 -> 128 -> 64 -> 26)",
        "results": results,
        "total_tests": len(results),
        "passed_tests": sum(1 for r in results if r.get("correct")),
        "failed_tests": sum(1 for r in results if not r.get("correct"))
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Test Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed.")
    print(f"Structured test report written to: {output_report}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SETU Controlled Alphabet Test Runner")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--duration", type=float, default=3.5, help="Duration in seconds per sign")
    args = parser.parse_args()

    run_controlled_test(camera_index=args.camera, seconds_per_sign=args.duration)
