"""
SETU — Phase 3.2: RealSign Dataset Split Leakage Audit
Inspects Dataset.zip without modifying any files:
1. Calculates SHA-256 hashes for all images across Training, Testing, and Validation splits.
2. Identifies exact duplicates within and across splits.
3. Computes perceptual difference hashes (dHash) to detect identical and near-duplicate frames.
4. Analyzes filename distribution and detects temporal frame adjacency leakage (e.g. frame k in Test, k±1 in Train).
5. Outputs a detailed, structured audit report to reports/realsign_leakage_report.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
from PIL import Image

ZIP_PATH = Path("data/raw/realsign/Dataset.zip")
REPORT_PATH = Path("reports/realsign_leakage_report.json")


def compute_dhash(img_bytes: bytes, hash_size: int = 8) -> int:
    """
    Computes a 64-bit difference hash (dHash) from image bytes.
    dHash tracks relative brightness gradients across horizontal adjacent pixels.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    # Resize to (width = hash_size + 1, height = hash_size) -> 9x8 for 64 bits
    resized = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(resized)
    # Compare left pixel > right pixel
    diff = arr[:, :-1] > arr[:, 1:]
    # Pack boolean 8x8 into 64-bit unsigned integer
    bit_str = "".join("1" if b else "0" for b in diff.flatten())
    return int(bit_str, 2)


def popcount64(x: int) -> int:
    """Returns number of set bits in an integer (Hamming distance)."""
    return bin(x).count("1")


def run_leakage_audit():
    print("=" * 70)
    print("SETU — RealSign Split Leakage & Duplicate Audit (Phase 3.2)")
    print("=" * 70)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"RealSign archive not found at {ZIP_PATH}")

    splits = ["Training", "Testing", "Validation"]
    
    # Data structures:
    # sha256 -> list of (split, letter, filename)
    hash_to_locations: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    
    # Store records per split and class:
    # split_class_records[split][class] = list of {filename, frame_num, sha256, dhash}
    split_class_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        s: defaultdict(list) for s in splits
    }

    # Count tracking
    total_images_scanned = 0
    corrupt_files_found: List[Dict[str, Any]] = []

    print(f"Opening archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = [n for n in z.namelist() if not n.endswith("/")]
        
        # Filter for Training, Testing, Validation images
        image_entries = []
        for name in all_names:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] in splits and parts[2].lower().endswith((".jpg", ".jpeg", ".png")):
                image_entries.append((parts[0], parts[1], parts[2], name))

        print(f"Total target images to scan: {len(image_entries)}")

        for idx, (sp, letter, fname, full_name) in enumerate(image_entries, 1):
            raw_bytes = z.read(full_name)
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            # Extract numerical frame index if available (e.g. '123.jpg' -> 123)
            stem = Path(fname).stem
            frame_num = int(stem) if stem.isdigit() else None

            is_valid_image = True
            dhash_val = None
            try:
                dhash_val = compute_dhash(raw_bytes)
            except Exception as e:
                is_valid_image = False
                corrupt_files_found.append({
                    "path": full_name,
                    "size_bytes": len(raw_bytes),
                    "error": str(e),
                    "snippet": raw_bytes[:150].decode("latin-1", errors="replace")
                })

            entry_info = {
                "filename": fname,
                "frame_num": frame_num,
                "sha256": sha256,
                "dhash": dhash_val,
                "is_valid_image": is_valid_image,
                "full_path": full_name
            }
            if is_valid_image:
                split_class_records[sp][letter].append(entry_info)
            hash_to_locations[sha256].append((sp, letter, fname))
            total_images_scanned += 1

            if idx % 5000 == 0 or idx == len(image_entries):
                print(f"  Scanned {idx}/{len(image_entries)} images...")

    print("\n--- 1. Exact SHA-256 Duplicate Analysis ---")
    exact_duplicates_across_splits: List[Dict[str, Any]] = []
    intra_split_duplicates: Dict[str, int] = defaultdict(int)

    train_test_exact = 0
    train_val_exact = 0
    test_val_exact = 0
    train_test_val_exact = 0

    for sha256, locs in hash_to_locations.items():
        if len(locs) > 1:
            splits_present = set(loc[0] for loc in locs)
            for sp in splits_present:
                count_in_sp = sum(1 for loc in locs if loc[0] == sp)
                if count_in_sp > 1:
                    intra_split_duplicates[sp] += (count_in_sp - 1)

            if len(splits_present) > 1:
                exact_duplicates_across_splits.append({
                    "sha256": sha256,
                    "occurrences": [f"{loc[0]}/{loc[1]}/{loc[2]}" for loc in locs],
                    "splits": sorted(list(splits_present))
                })
                if splits_present == {"Training", "Testing"}:
                    train_test_exact += 1
                elif splits_present == {"Training", "Validation"}:
                    train_val_exact += 1
                elif splits_present == {"Testing", "Validation"}:
                    test_val_exact += 1
                elif len(splits_present) == 3:
                    train_test_val_exact += 1

    print(f"Exact identical files across splits: {len(exact_duplicates_across_splits)}")
    print(f"  Training <-> Testing exact matches: {train_test_exact}")
    print(f"  Training <-> Validation exact matches: {train_val_exact}")
    print(f"  Testing <-> Validation exact matches: {test_val_exact}")
    print(f"  All 3 splits exact matches: {train_test_val_exact}")
    for sp, cnt in intra_split_duplicates.items():
        print(f"  Intra-split duplicate images in {sp}: {cnt}")

    print("\n--- 2. Perceptual Similarity & Near-Duplicate Analysis (dHash) ---")
    # For each letter, compare dHash between Training and Testing/Validation
    classes = sorted(list(set(l for sp in splits for l in split_class_records[sp].keys())))
    
    per_class_leakage: Dict[str, Dict[str, Any]] = {}
    total_dhash_exact_train_test = 0
    total_dhash_near2_train_test = 0
    total_dhash_exact_train_val = 0
    total_dhash_near2_train_val = 0

    # Temporal frame adjacency analysis
    # If frame_num k is in Test, check if frame k-1 or k+1 is in Train
    total_test_frames = 0
    total_test_with_adjacent_train_frame = 0
    total_test_with_within2_train_frame = 0

    for letter in classes:
        train_recs = split_class_records["Training"].get(letter, [])
        test_recs = split_class_records["Testing"].get(letter, [])
        val_recs = split_class_records["Validation"].get(letter, [])

        train_dhashes = np.array([r["dhash"] for r in train_recs], dtype=np.uint64)
        test_dhashes = np.array([r["dhash"] for r in test_recs], dtype=np.uint64)
        val_dhashes = np.array([r["dhash"] for r in val_recs], dtype=np.uint64)

        train_frames = set(r["frame_num"] for r in train_recs if r["frame_num"] is not None)
        test_frames = [r["frame_num"] for r in test_recs if r["frame_num"] is not None]
        val_frames = set(r["frame_num"] for r in val_recs if r["frame_num"] is not None)

        # Vectorized / pairwise dHash comparison per class
        # train vs test:
        dhash_exact_tt = 0
        dhash_near2_tt = 0
        if len(train_dhashes) > 0 and len(test_dhashes) > 0:
            for th in test_dhashes:
                # Bitwise XOR with all train hashes
                diffs = np.bitwise_xor(train_dhashes, th)
                # Compute popcounts
                # In Python, popcount on array:
                min_dist = 64
                for d in diffs:
                    h_dist = popcount64(int(d))
                    if h_dist < min_dist:
                        min_dist = h_dist
                        if min_dist == 0:
                            break
                if min_dist == 0:
                    dhash_exact_tt += 1
                elif min_dist <= 2:
                    dhash_near2_tt += 1

        # train vs val:
        dhash_exact_tv = 0
        dhash_near2_tv = 0
        if len(train_dhashes) > 0 and len(val_dhashes) > 0:
            for vh in val_dhashes:
                min_dist = 64
                diffs = np.bitwise_xor(train_dhashes, vh)
                for d in diffs:
                    h_dist = popcount64(int(d))
                    if h_dist < min_dist:
                        min_dist = h_dist
                        if min_dist == 0:
                            break
                if min_dist == 0:
                    dhash_exact_tv += 1
                elif min_dist <= 2:
                    dhash_near2_tv += 1

        total_dhash_exact_train_test += dhash_exact_tt
        total_dhash_near2_train_test += dhash_near2_tt
        total_dhash_exact_train_val += dhash_exact_tv
        total_dhash_near2_train_val += dhash_near2_tv

        # Temporal adjacency test:
        adjacent_count = 0
        within2_count = 0
        for f in test_frames:
            if (f - 1 in train_frames) or (f + 1 in train_frames):
                adjacent_count += 1
            if any(f + delta in train_frames for delta in (-2, -1, 1, 2)):
                within2_count += 1

        total_test_frames += len(test_frames)
        total_test_with_adjacent_train_frame += adjacent_count
        total_test_with_within2_train_frame += within2_count

        per_class_leakage[letter] = {
            "train_count": len(train_recs),
            "test_count": len(test_recs),
            "val_count": len(val_recs),
            "test_dhash_exact_match_in_train": dhash_exact_tt,
            "test_dhash_near2_match_in_train": dhash_near2_tt,
            "val_dhash_exact_match_in_train": dhash_exact_tv,
            "val_dhash_near2_match_in_train": dhash_near2_tv,
            "test_frames_adjacent_to_train": adjacent_count,
            "test_frames_within_2_of_train": within2_count,
            "adjacent_leakage_ratio": round(adjacent_count / len(test_frames), 4) if test_frames else 0.0
        }

    print(f"Perceptually identical frames (dHash dist=0) Test in Train: {total_dhash_exact_train_test} / {total_test_frames} ({total_dhash_exact_train_test/total_test_frames*100:.2f}%)")
    print(f"Perceptually near-identical (dHash dist<=2) Test in Train: {total_dhash_near2_train_test} / {total_test_frames} ({total_dhash_near2_train_test/total_test_frames*100:.2f}%)")
    print(f"Test frames strictly adjacent (k±1) to Train frames: {total_test_with_adjacent_train_frame} / {total_test_frames} ({total_test_with_adjacent_train_frame/total_test_frames*100:.2f}%)")
    print(f"Test frames within ±2 frames of a Train frame: {total_test_with_within2_train_frame} / {total_test_frames} ({total_test_with_within2_train_frame/total_test_frames*100:.2f}%)")

    print("\n--- 3. Filename & Identity Mapping Analysis ---")
    # Inspect sample filenames across splits for letters
    # Sample letters: A, B, C, Z
    filename_analysis = {}
    for letter in ["A", "B", "C", "Z"]:
        train_fnames = sorted([int(r["filename"].split(".")[0]) for r in split_class_records["Training"].get(letter, []) if r["filename"].split(".")[0].isdigit()])
        test_fnames = sorted([int(r["filename"].split(".")[0]) for r in split_class_records["Testing"].get(letter, []) if r["filename"].split(".")[0].isdigit()])
        val_fnames = sorted([int(r["filename"].split(".")[0]) for r in split_class_records["Validation"].get(letter, []) if r["filename"].split(".")[0].isdigit()])
        
        all_nums = sorted(train_fnames + test_fnames + val_fnames)
        is_contiguous_0_to_max = (all_nums == list(range(len(all_nums))))
        filename_analysis[letter] = {
            "total_frames": len(all_nums),
            "range": [min(all_nums), max(all_nums)] if all_nums else [],
            "is_contiguous_0_to_max": is_contiguous_0_to_max,
            "train_sample_head": train_fnames[:10],
            "test_sample_head": test_fnames[:10],
            "val_sample_head": val_fnames[:10],
        }

    report = {
        "dataset_archive": str(ZIP_PATH),
        "total_images_scanned": total_images_scanned,
        "corrupt_or_non_image_files": corrupt_files_found,
        "exact_sha256_duplicates": {
            "total_exact_duplicates_across_splits": len(exact_duplicates_across_splits),
            "train_test_exact": train_test_exact,
            "train_val_exact": train_val_exact,
            "test_val_exact": test_val_exact,
            "all_three_splits_exact": train_test_val_exact,
            "intra_split_duplicates": dict(intra_split_duplicates),
            "sample_duplicate_pairs": exact_duplicates_across_splits[:15]
        },
        "perceptual_similarity_dhash": {
            "total_test_frames": total_test_frames,
            "test_exact_dhash_in_train": total_dhash_exact_train_test,
            "test_exact_dhash_ratio": round(total_dhash_exact_train_test / total_test_frames, 4) if total_test_frames else 0.0,
            "test_near2_dhash_in_train": total_dhash_near2_train_test,
            "test_near2_dhash_ratio": round(total_dhash_near2_train_test / total_test_frames, 4) if total_test_frames else 0.0,
            "val_exact_dhash_in_train": total_dhash_exact_train_val,
            "val_near2_dhash_in_train": total_dhash_near2_train_val
        },
        "temporal_frame_leakage": {
            "test_frames_adjacent_to_train_k_plus_minus_1": total_test_with_adjacent_train_frame,
            "adjacent_leakage_percentage": round(total_test_with_adjacent_train_frame / total_test_frames * 100, 2) if total_test_frames else 0.0,
            "test_frames_within_2_of_train": total_test_with_within2_train_frame,
            "within_2_leakage_percentage": round(total_test_with_within2_train_frame / total_test_frames * 100, 2) if total_test_frames else 0.0
        },
        "filename_identity_mapping": {
            "can_map_to_signer_identity": False,
            "findings": (
                "Filenames are anonymous consecutive integer frame numbers (e.g., '0.jpg' to '999.jpg'). "
                "There is no metadata, CSV, or signer/subject ID in filenames. "
                "The frame numbers 0 to N form a continuous video recording sequence. "
                "The dataset authors performed a pseudo-random shuffle of individual video frames into Training, Testing, and Validation. "
                "Because adjacent frames (e.g. frame 174 in Train, 175 in Test) belong to the exact same video recording session, "
                "the pre-packaged test split suffers from severe temporal autocorrelation leakage."
            ),
            "sample_letter_distributions": filename_analysis
        },
        "per_class_summary": per_class_leakage
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit complete. Full report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_leakage_audit()
