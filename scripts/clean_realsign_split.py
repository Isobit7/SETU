"""
SETU — Clean, Leakage-Aware Dataset Manifest Generator for RealSign (ISL Alphabet)

This script:
1. Traverses the RealSign dataset from 'data/raw/realsign/Dataset.zip' (or extracted directory).
2. Validates image integrity using PIL to filter out corrupt/non-image files (e.g. text/code files).
3. Computes SHA-256 content hashes to detect intra-split and cross-split duplicate images.
4. Enforces strict zero-leakage split deduplication using split precedence:
   Training > Validation > Testing.
   Ensures every unique content hash appears in EXACTLY ONE split.
5. Emits a structured JSON split manifest saved at:
   'data/manifests/realsign_clean_manifest.json'
6. Verifies via automated assertion checks that zero duplicate hashes exist across splits.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image


DEFAULT_ZIP_PATH = Path("data/raw/realsign/Dataset.zip")
DEFAULT_DIR_PATH = Path("data/raw/realsign/Dataset")
DEFAULT_OUTPUT_MANIFEST = Path("data/manifests/realsign_clean_manifest.json")

TARGET_SPLITS = ("Training", "Validation", "Testing")
VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def validate_image_bytes(raw_bytes: bytes) -> bool:
    """Verifies that bytes represent a valid, decodable image."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_image_file(file_path: Path) -> bool:
    """Verifies that file on disk represents a valid, decodable image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_samples_from_zip(
    zip_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scans and extracts metadata from all target split entries inside a ZIP archive."""
    print(f"Scanning archive: {zip_path} ({zip_path.stat().st_size:,} bytes)...")
    valid_records: List[Dict[str, Any]] = []
    corrupt_records: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = [n for n in z.namelist() if not n.endswith("/")]
        
        target_names = []
        for name in namelist:
            parts = name.replace("\\", "/").split("/")
            if len(parts) == 3 and parts[0] in TARGET_SPLITS:
                target_names.append((parts[0], parts[1], parts[2], name))

        print(f"Found {len(target_names):,} candidate files across {TARGET_SPLITS}.")

        for idx, (split, class_name, file_name, full_name) in enumerate(target_names, 1):
            raw_bytes = z.read(full_name)
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            size_bytes = len(raw_bytes)

            if not validate_image_bytes(raw_bytes):
                # Identify snippet of corrupt file for diagnostic reporting
                snippet = raw_bytes[:120].decode("latin-1", errors="replace")
                corrupt_records.append({
                    "path": full_name,
                    "split": split,
                    "class": class_name,
                    "filename": file_name,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "reason": "Image validation failed (non-image or corrupt binary)",
                    "snippet": snippet,
                })
                continue

            valid_records.append({
                "split_origin": split,
                "class": class_name,
                "filename": file_name,
                "archive_path": full_name,
                "source_type": "zip",
                "sha256": sha256,
                "size_bytes": size_bytes,
            })

            if idx % 5000 == 0 or idx == len(target_names):
                print(f"  Processed {idx:,} / {len(target_names):,} images...")

    return valid_records, corrupt_records


def collect_samples_from_dir(
    dir_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scans and extracts metadata from target split directories on disk."""
    print(f"Scanning directory: {dir_path}...")
    valid_records: List[Dict[str, Any]] = []
    corrupt_records: List[Dict[str, Any]] = []

    all_files: List[Tuple[str, str, str, Path]] = []
    for split in TARGET_SPLITS:
        split_dir = dir_path / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                for fpath in class_dir.iterdir():
                    if fpath.is_file() and fpath.suffix.lower() in VALID_IMAGE_EXTS:
                        all_files.append((split, class_name, fpath.name, fpath))

    print(f"Found {len(all_files):,} candidate files on disk across {TARGET_SPLITS}.")

    for idx, (split, class_name, file_name, fpath) in enumerate(all_files, 1):
        try:
            raw_bytes = fpath.read_bytes()
        except Exception as e:
            corrupt_records.append({
                "path": str(fpath),
                "split": split,
                "class": class_name,
                "filename": file_name,
                "size_bytes": 0,
                "sha256": "",
                "reason": f"File read error: {e}",
            })
            continue

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        size_bytes = len(raw_bytes)

        if not validate_image_bytes(raw_bytes):
            snippet = raw_bytes[:120].decode("latin-1", errors="replace")
            corrupt_records.append({
                "path": str(fpath),
                "split": split,
                "class": class_name,
                "filename": file_name,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "reason": "Image validation failed (non-image or corrupt binary)",
                "snippet": snippet,
            })
            continue

        valid_records.append({
            "split_origin": split,
            "class": class_name,
            "filename": file_name,
            "file_path": str(fpath),
            "source_type": "directory",
            "sha256": sha256,
            "size_bytes": size_bytes,
        })

        if idx % 5000 == 0 or idx == len(all_files):
            print(f"  Processed {idx:,} / {len(all_files):,} images...")

    return valid_records, corrupt_records


def deduplicate_and_assign_splits(
    valid_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Groups valid records by sha256 content hash and assigns each unique image
    to exactly one split according to deterministic precedence:
      Training > Validation > Testing
    
    Returns:
      clean_splits: dict mapping split name to list of unique sample dicts.
      dropped_duplicates: list of dropped duplicate occurrences with audit info.
      stats: summary counts.
    """
    hash_to_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in valid_records:
        hash_to_occurrences[rec["sha256"]].append(rec)

    clean_splits: Dict[str, List[Dict[str, Any]]] = {
        "Training": [],
        "Validation": [],
        "Testing": [],
    }

    dropped_duplicates: List[Dict[str, Any]] = []

    cross_split_duplicates_count = 0
    intra_split_duplicates_count = 0

    # Sort hashes for deterministic ordering
    sorted_hashes = sorted(hash_to_occurrences.keys())

    for sha256 in sorted_hashes:
        occurrences = hash_to_occurrences[sha256]
        
        # Check if multiple occurrences exist
        is_duplicate = len(occurrences) > 1
        splits_involved = set(occ["split_origin"] for occ in occurrences)

        if is_duplicate:
            if len(splits_involved) > 1:
                cross_split_duplicates_count += 1
            else:
                intra_split_duplicates_count += (len(occurrences) - 1)

        # Precedence policy: Training > Validation > Testing
        chosen_record = None
        for preferred_split in ("Training", "Validation", "Testing"):
            candidates = [occ for occ in occurrences if occ["split_origin"] == preferred_split]
            if candidates:
                # Deterministic selection within split: sort by path/filename
                candidates.sort(key=lambda r: r.get("archive_path") or r.get("file_path", ""))
                chosen_record = candidates[0]
                break

        if chosen_record is None:
            chosen_record = occurrences[0]

        assigned_split = chosen_record["split_origin"]

        # Track any other occurrences dropped
        other_occurrences = [occ for occ in occurrences if occ is not chosen_record]
        for dropped in other_occurrences:
            dropped_duplicates.append({
                "dropped_path": dropped.get("archive_path") or dropped.get("file_path"),
                "dropped_split": dropped["split_origin"],
                "class": dropped["class"],
                "sha256": sha256,
                "retained_split": assigned_split,
                "retained_path": chosen_record.get("archive_path") or chosen_record.get("file_path"),
                "leakage_type": "cross_split" if dropped["split_origin"] != assigned_split else "intra_split",
            })

        # Append to retained clean split
        sample_entry = {
            "sample_id": f"{assigned_split}_{chosen_record['class']}_{Path(chosen_record['filename']).stem}",
            "class": chosen_record["class"],
            "split": assigned_split,
            "filename": chosen_record["filename"],
            "path": chosen_record.get("archive_path") or chosen_record.get("file_path"),
            "sha256": sha256,
            "size_bytes": chosen_record["size_bytes"],
            "redundant_copies_dropped": len(other_occurrences),
        }
        clean_splits[assigned_split].append(sample_entry)

    # Sort each split by class then sample_id
    for sp in clean_splits:
        clean_splits[sp].sort(key=lambda x: (x["class"], x["sample_id"]))

    stats = {
        "unique_hashes": len(hash_to_occurrences),
        "total_dropped_duplicates": len(dropped_duplicates),
        "cross_split_duplicate_groups": cross_split_duplicates_count,
        "intra_split_duplicates_count": intra_split_duplicates_count,
    }
    return clean_splits, dropped_duplicates, stats


def build_manifest_document(
    clean_splits: Dict[str, List[Dict[str, Any]]],
    corrupt_records: List[Dict[str, Any]],
    dropped_duplicates: List[Dict[str, Any]],
    stats: Dict[str, Any],
    source_path: Path,
) -> Dict[str, Any]:
    """Assembles the formal manifest JSON document with comprehensive metadata."""
    classes = sorted(list(set(
        sample["class"]
        for sp in clean_splits.values()
        for sample in sp
    )))

    split_counts: Dict[str, int] = {sp: len(samples) for sp, samples in clean_splits.items()}
    total_samples = sum(split_counts.values())

    class_distribution: Dict[str, Dict[str, int]] = {
        sp: defaultdict(int) for sp in clean_splits
    }
    class_totals: Dict[str, int] = defaultdict(int)

    for sp, samples in clean_splits.items():
        for s in samples:
            class_distribution[sp][s["class"]] += 1
            class_totals[s["class"]] += 1

    manifest = {
        "name": "SETU-RealSign-Clean-Deduplicated-Manifest",
        "version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": str(source_path),
        "deduplication_policy": {
            "strategy": "leakage_free_strict_split",
            "precedence": "Training > Validation > Testing",
            "cross_split_overlap": 0,
            "description": (
                "Ensures each unique SHA-256 content hash appears in ONLY ONE split. "
                "Intra-split duplicate images and corrupt non-image files are excluded."
            ),
        },
        "audit_summary": {
            "total_images_scanned": total_samples + len(dropped_duplicates) + len(corrupt_records),
            "corrupt_files_excluded": len(corrupt_records),
            "total_valid_images": total_samples + len(dropped_duplicates),
            "duplicate_instances_dropped": len(dropped_duplicates),
            "cross_split_duplicate_groups": stats["cross_split_duplicate_groups"],
            "intra_split_duplicate_copies": stats["intra_split_duplicates_count"],
            "unique_valid_samples_retained": total_samples,
            "samples_per_split": split_counts,
            "num_classes": len(classes),
            "classes": classes,
            "class_distribution_by_split": {
                sp: dict(sorted(counts.items())) for sp, counts in class_distribution.items()
            },
            "class_totals": dict(sorted(class_totals.items())),
        },
        "corrupt_files": corrupt_records,
        "sample_dropped_duplicates": dropped_duplicates[:50],  # sample preview
        "samples": {
            sp: samples for sp, samples in clean_splits.items()
        },
    }
    return manifest


def verify_manifest_integrity(manifest: Dict[str, Any]) -> None:
    """Rigorous assertion checks to guarantee zero duplicates or cross-split leakage."""
    print("\nRunning automated integrity verification...")

    samples_by_split = manifest["samples"]
    split_hashes: Dict[str, Set[str]] = {sp: set() for sp in samples_by_split}

    for sp, samples in samples_by_split.items():
        for s in samples:
            h = s["sha256"]
            assert h not in split_hashes[sp], f"Intra-split duplicate found in {sp} for hash {h}!"
            split_hashes[sp].add(h)

    # Check pair-wise intersections across splits
    train_val_overlap = split_hashes["Training"].intersection(split_hashes["Validation"])
    train_test_overlap = split_hashes["Training"].intersection(split_hashes["Testing"])
    val_test_overlap = split_hashes["Validation"].intersection(split_hashes["Testing"])

    assert len(train_val_overlap) == 0, f"Leakage detected: {len(train_val_overlap)} hashes overlap Training & Validation!"
    assert len(train_test_overlap) == 0, f"Leakage detected: {len(train_test_overlap)} hashes overlap Training & Testing!"
    assert len(val_test_overlap) == 0, f"Leakage detected: {len(val_test_overlap)} hashes overlap Validation & Testing!"

    # Verify corrupt files are not present in manifest samples
    corrupt_paths = set(c["path"] for c in manifest["corrupt_files"])
    all_sample_paths = set(
        s["path"] for sp in samples_by_split.values() for s in samples
    )
    assert len(corrupt_paths.intersection(all_sample_paths)) == 0, "Corrupt files found in retained samples!"

    print(" Integrity verification PASSED:")
    print("   - Zero duplicate hashes within any split.")
    print("   - Zero cross-split hash overlap (Train n Val = 0, Train n Test = 0, Val n Test = 0).")
    print("   - Zero corrupt or unreadable files retained.")


def main():
    parser = argparse.ArgumentParser(description="RealSign Clean Split Manifest Generator")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH, help="Path to Dataset.zip")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR_PATH, help="Path to extracted Dataset folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MANIFEST, help="Output manifest JSON path")
    args = parser.parse_args()

    print("=" * 72)
    print("SETU — RealSign Clean & Leakage-Aware Split Manifest Generator")
    print("=" * 72)

    source_path: Path
    if args.dir.exists() and args.dir.is_dir():
        source_path = args.dir
        valid_records, corrupt_records = collect_samples_from_dir(source_path)
    elif args.zip.exists() and args.zip.is_file():
        source_path = args.zip
        valid_records, corrupt_records = collect_samples_from_zip(source_path)
    else:
        raise FileNotFoundError(
            f"Neither extracted dataset ({args.dir}) nor archive ({args.zip}) found."
        )

    print(f"\nTotal raw images scanned: {len(valid_records) + len(corrupt_records):,}")
    print(f"Corrupt or non-image files excluded: {len(corrupt_records)}")
    if corrupt_records:
        for c in corrupt_records:
            print(f"  [X] Excluded: {c['path']} ({c['size_bytes']} B) — {c['reason']}")

    print(f"Valid image records for deduplication: {len(valid_records):,}")

    # Run deduplication
    clean_splits, dropped_duplicates, stats = deduplicate_and_assign_splits(valid_records)

    print("\n--- Deduplication Summary ---")
    print(f"Unique image content hashes: {stats['unique_hashes']:,}")
    print(f"Total duplicate instances dropped: {stats['total_dropped_duplicates']:,}")
    print(f"  - Cross-split duplicate hash groups: {stats['cross_split_duplicate_groups']:,}")
    print(f"  - Redundant intra-split copies dropped: {stats['intra_split_duplicates_count']:,}")

    # Build manifest
    manifest = build_manifest_document(
        clean_splits=clean_splits,
        corrupt_records=corrupt_records,
        dropped_duplicates=dropped_duplicates,
        stats=stats,
        source_path=source_path,
    )

    # Integrity verification
    verify_manifest_integrity(manifest)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting clean manifest to: {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f" Manifest successfully saved ({file_size_mb:.2f} MB)")

    print("\n" + "=" * 72)
    print(f"{'Class':<6} | {'Training':<10} | {'Validation':<10} | {'Testing':<10} | {'Total':<10}")
    print("-" * 72)
    dist = manifest["audit_summary"]["class_distribution_by_split"]
    totals = manifest["audit_summary"]["class_totals"]
    for cls in manifest["audit_summary"]["classes"]:
        tr = dist["Training"].get(cls, 0)
        va = dist["Validation"].get(cls, 0)
        te = dist["Testing"].get(cls, 0)
        tot = totals.get(cls, 0)
        print(f"{cls:<6} | {tr:<10} | {va:<10} | {te:<10} | {tot:<10}")
    print("-" * 72)
    summary_splits = manifest["audit_summary"]["samples_per_split"]
    tot_all = sum(summary_splits.values())
    print(f"{'TOTAL':<6} | {summary_splits['Training']:<10} | {summary_splits['Validation']:<10} | {summary_splits['Testing']:<10} | {tot_all:<10}")
    print("=" * 72)


if __name__ == "__main__":
    main()
