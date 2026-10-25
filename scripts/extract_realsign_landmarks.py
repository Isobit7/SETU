"""
SETU — RealSign 26-Class Hand Landmark Extractor

Processes clean RealSign alphabet dataset from 'data/manifests/realsign_clean_manifest.json'
and 'data/raw/realsign/Dataset.zip':
1. Reads each valid image directly from the ZIP archive.
2. Runs MediaPipe Hand Landmarker (static_image_mode=True / RunningMode.IMAGE, max_num_hands=1).
3. If detected, extracts 21 3D landmarks (x, y, z), translates by wrist (landmark 0),
   and normalizes by palm span (distance from wrist 0 to middle MCP 9).
4. Flattens each hand into a 63-dimensional normalized feature vector.
5. If no hand is detected, records a failure and skips gracefully.
6. Saves extracted feature arrays, labels, split assignments, and audit stats into:
   'data/processed/realsign_landmarks_26class.npz'
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.feature_extractor import normalize_hand_landmarks

DEFAULT_MANIFEST_PATH = ROOT_DIR / "data" / "manifests" / "realsign_clean_manifest.json"
DEFAULT_ZIP_PATH = ROOT_DIR / "data" / "raw" / "realsign" / "Dataset.zip"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "tasks" / "hand_landmarker.task"
DEFAULT_OUTPUT_NPZ = ROOT_DIR / "data" / "processed" / "realsign_landmarks_26class.npz"

# Process-local state for worker pools
_worker_detector: Optional[vision.HandLandmarker] = None
_worker_zip: Optional[zipfile.ZipFile] = None


def init_worker(model_path_str: str, zip_path_str: str, min_confidence: float) -> None:
    """Initializes a process-local MediaPipe HandLandmarker and open ZipFile."""
    global _worker_detector, _worker_zip
    base_options = python.BaseOptions(model_asset_path=model_path_str)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_confidence,
    )
    _worker_detector = vision.HandLandmarker.create_from_options(options)
    _worker_zip = zipfile.ZipFile(zip_path_str, "r")


def process_image_chunk(chunk: List[Tuple[str, str, str, str]]) -> List[Dict[str, Any]]:
    """
    Processes a chunk of images in a worker process.
    Input chunk: [(sample_id, path, class_name, split), ...]
    Output: list of result dicts.
    """
    global _worker_detector, _worker_zip
    results: List[Dict[str, Any]] = []

    for sample_id, path, class_name, split in chunk:
        try:
            raw_bytes = _worker_zip.read(path)
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({
                    "sample_id": sample_id,
                    "path": path,
                    "class": class_name,
                    "split": split,
                    "detected": False,
                    "features": None,
                    "reason": "Image decode failed",
                })
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = _worker_detector.detect(mp_img)

            if res and res.hand_landmarks and len(res.hand_landmarks) > 0:
                lms = res.hand_landmarks[0]
                pts = np.array([[pt.x, pt.y, pt.z] for pt in lms], dtype=np.float32)
                norm_pts, scale = normalize_hand_landmarks(pts)
                features_63 = norm_pts.flatten()

                results.append({
                    "sample_id": sample_id,
                    "path": path,
                    "class": class_name,
                    "split": split,
                    "detected": True,
                    "features": features_63,
                    "palm_scale": scale,
                })
            else:
                results.append({
                    "sample_id": sample_id,
                    "path": path,
                    "class": class_name,
                    "split": split,
                    "detected": False,
                    "features": None,
                    "reason": "No hand detected",
                })
        except Exception as e:
            results.append({
                "sample_id": sample_id,
                "path": path,
                "class": class_name,
                "split": split,
                "detected": False,
                "features": None,
                "reason": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from clean RealSign dataset")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to clean manifest JSON")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH, help="Path to Dataset.zip")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to hand_landmarker.task")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_NPZ, help="Path to output NPZ file")
    parser.add_argument("--confidence", type=float, default=0.4, help="Min hand detection confidence")
    parser.add_argument("--workers", type=int, default=min(12, max(2, (os.cpu_count() or 4) - 2)), help="Worker processes")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size per worker task")
    args = parser.parse_args()

    print("=" * 75)
    print("SETU — RealSign 26-Class Hand Landmark Extractor (Phase 3.3)")
    print("=" * 75)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.zip.exists():
        raise FileNotFoundError(f"Dataset archive not found: {args.zip}")
    if not args.model.exists():
        raise FileNotFoundError(f"Hand landmarker task model not found: {args.model}")

    print(f"Loading manifest: {args.manifest}...")
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_doc = json.load(f)

    samples_by_split = manifest_doc["samples"]
    all_classes = manifest_doc["audit_summary"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    items_to_process: List[Tuple[str, str, str, str]] = []
    for split_name in ("Training", "Validation", "Testing"):
        split_samples = samples_by_split.get(split_name, [])
        for s in split_samples:
            items_to_process.append((
                s["sample_id"],
                s["path"],
                s["class"],
                split_name,
            ))

    total_items = len(items_to_process)
    print(f"Total clean images to process: {total_items:,}")
    print(f"Worker processes: {args.workers} (chunk size: {args.chunk_size})")
    print(f"Hand detection confidence threshold: {args.confidence}")

    chunks = [
        items_to_process[i: i + args.chunk_size]
        for i in range(0, total_items, args.chunk_size)
    ]
    print(f"Dispatching {len(chunks)} chunks across {args.workers} workers...")

    start_time = time.time()
    all_results: List[Dict[str, Any]] = []
    completed_items = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(args.model), str(args.zip), args.confidence),
    ) as executor:
        for chunk_res in executor.map(process_image_chunk, chunks):
            all_results.extend(chunk_res)
            completed_items += len(chunk_res)
            if completed_items % 2500 == 0 or completed_items == total_items:
                elapsed = time.time() - start_time
                rate = completed_items / max(elapsed, 0.001)
                remaining = (total_items - completed_items) / max(rate, 0.001)
                print(
                    f"  Processed {completed_items:,} / {total_items:,} images "
                    f"({completed_items/total_items*100:.1f}%) — "
                    f"{rate:.1f} img/s — ETA: {remaining:.0f}s"
                )

    total_duration = time.time() - start_time
    print(f"\nExtraction completed in {total_duration:.1f}s ({total_items/total_duration:.1f} img/s).")

    # Aggregate extracted data and compute statistics
    extracted_features: List[np.ndarray] = []
    labels_idx: List[int] = []
    split_tags: List[str] = []
    paths: List[str] = []
    sample_ids: List[str] = []

    split_stats = {
        sp: {"total": 0, "detected": 0, "failed": 0, "success_rate": 0.0}
        for sp in ("Training", "Validation", "Testing")
    }
    class_stats = {
        c: {"total": 0, "detected": 0, "failed": 0, "success_rate": 0.0}
        for c in all_classes
    }

    for res in all_results:
        sp = res["split"]
        c = res["class"]
        split_stats[sp]["total"] += 1
        class_stats[c]["total"] += 1

        if res["detected"]:
            split_stats[sp]["detected"] += 1
            class_stats[c]["detected"] += 1
            extracted_features.append(res["features"])
            labels_idx.append(class_to_idx[c])
            split_tags.append(sp)
            paths.append(res["path"])
            sample_ids.append(res["sample_id"])
        else:
            split_stats[sp]["failed"] += 1
            class_stats[c]["failed"] += 1

    for sp in split_stats:
        tot = split_stats[sp]["total"]
        det = split_stats[sp]["detected"]
        split_stats[sp]["success_rate"] = round((det / tot * 100.0) if tot > 0 else 0.0, 2)

    for c in class_stats:
        tot = class_stats[c]["total"]
        det = class_stats[c]["detected"]
        class_stats[c]["success_rate"] = round((det / tot * 100.0) if tot > 0 else 0.0, 2)

    total_detected = len(extracted_features)
    total_failed = total_items - total_detected
    overall_success_rate = round((total_detected / total_items * 100.0), 2)

    print("\n" + "=" * 75)
    print("Landmark Detection Statistics per Split")
    print("-" * 75)
    print(f"{'Split':<12} | {'Total':<8} | {'Detected':<10} | {'Failed':<8} | {'Success Rate':<12}")
    print("-" * 75)
    for sp in ("Training", "Validation", "Testing"):
        s = split_stats[sp]
        print(f"{sp:<12} | {s['total']:<8} | {s['detected']:<10} | {s['failed']:<8} | {s['success_rate']:>6.2f}%")
    print("-" * 75)
    print(f"{'OVERALL':<12} | {total_items:<8} | {total_detected:<10} | {total_failed:<8} | {overall_success_rate:>6.2f}%")
    print("=" * 75)

    # Convert to NumPy arrays
    X_arr = np.array(extracted_features, dtype=np.float32)
    y_arr = np.array(labels_idx, dtype=np.int64)
    splits_arr = np.array(split_tags, dtype=object)
    classes_arr = np.array(all_classes, dtype=object)
    paths_arr = np.array(paths, dtype=object)
    sample_ids_arr = np.array(sample_ids, dtype=object)

    stats_dict = {
        "dataset_name": "SETU-RealSign-26Class-Landmarks",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_scanned_images": total_items,
        "total_detected_landmarks": total_detected,
        "total_detection_failures": total_failed,
        "overall_detection_rate_pct": overall_success_rate,
        "feature_dim": 63,
        "num_classes": len(all_classes),
        "split_stats": split_stats,
        "class_stats": class_stats,
        "classes": all_classes,
        "execution_time_seconds": round(total_duration, 2),
    }

    # Ensure output parent directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving features to: {args.output}...")
    np.savez_compressed(
        args.output,
        X=X_arr,
        y=y_arr,
        splits=splits_arr,
        classes=classes_arr,
        paths=paths_arr,
        sample_ids=sample_ids_arr,
        stats_json=json.dumps(stats_dict, indent=2),
    )

    output_size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f" Features saved successfully ({output_size_mb:.2f} MB, {X_arr.shape[0]:,} valid feature vectors, shape: {X_arr.shape})")


if __name__ == "__main__":
    main()
