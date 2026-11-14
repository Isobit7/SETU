"""
SETU — RealSign ISL Alphabet Dataset Deep Audit
Inspects Dataset.zip:
1. Enumerates exact class counts across Training, Testing, Validation splits.
2. Analyzes filename conventions to evaluate signer-isolation feasibility.
3. Extracts canonical reference letters to data/raw/realsign/letters_reference/.
4. Runs MediaPipe Holistic detection on reference letters to verify dual-hand detection feasibility.
5. Cross-checks against ISLRTC Indian Sign Language manual alphabet definitions.
6. Outputs a comprehensive audit report to reports/realsign_audit_report.json.
"""

import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.vision import HolisticLandmarkerWrapper

ZIP_PATH = Path("data/raw/realsign/Dataset.zip")
REFERENCE_DIR = Path("data/raw/realsign/letters_reference")
REPORT_PATH = Path("reports/realsign_audit_report.json")

# Official ISLRTC (Indian Sign Language Research and Training Centre) Alphabet Standard
ISLRTC_STANDARDS: Dict[str, Dict[str, Any]] = {
    "A": {"type": "Two-handed", "description": "Dominant hand index finger touches thumb tip of non-dominant open hand.", "movement": False},
    "B": {"type": "Two-handed", "description": "Both hands form circles with thumb and index fingers, touching side by side.", "movement": False},
    "C": {"type": "Single-handed", "description": "Dominant hand forms a curved 'C' shape (may be supported by non-dominant palm).", "movement": False},
    "D": {"type": "Two-handed", "description": "Non-dominant index points vertically; dominant index and thumb form an arch touching it.", "movement": False},
    "E": {"type": "Two-handed", "description": "Dominant index touches the index finger tip of non-dominant open hand.", "movement": False},
    "F": {"type": "Two-handed", "description": "Dominant index and middle finger cross over non-dominant index and middle finger.", "movement": False},
    "G": {"type": "Two-handed", "description": "Both hands form fists touching knuckles side by side.", "movement": False},
    "H": {"type": "Two-handed", "description": "Dominant flat palm sweeps across non-dominant flat palm horizontally.", "movement": False},
    "I": {"type": "Two-handed", "description": "Dominant index touches middle finger tip of non-dominant open hand (vowel contact).", "movement": False},
    "J": {"type": "Dynamic (Movement)", "description": "Dominant index draws a 'J' hook downwards across non-dominant palm. Requires motion tracking.", "movement": True},
    "K": {"type": "Two-handed", "description": "Non-dominant index points vertically; dominant index bends at middle joint against it.", "movement": False},
    "L": {"type": "Single-handed", "description": "Dominant index vertical, thumb extended horizontally at 90 degrees.", "movement": False},
    "M": {"type": "Two-handed", "description": "Three fingers (index, middle, ring) of dominant hand rest on non-dominant palm.", "movement": False},
    "N": {"type": "Two-handed", "description": "Two fingers (index, middle) of dominant hand rest on non-dominant palm.", "movement": False},
    "O": {"type": "Two-handed", "description": "Dominant index touches ring finger tip of non-dominant open hand (vowel contact).", "movement": False},
    "P": {"type": "Two-handed", "description": "Non-dominant index points vertically; dominant hand forms circle touching near top.", "movement": False},
    "Q": {"type": "Two-handed", "description": "Dominant thumb and index loop hooks over non-dominant thumb.", "movement": False},
    "R": {"type": "Two-handed", "description": "Dominant index hooked across non-dominant flat palm.", "movement": False},
    "S": {"type": "Two-handed", "description": "Pinky fingers of both hands hook together (or dominant pinky on palm).", "movement": False},
    "T": {"type": "Two-handed", "description": "Dominant index touches edge of non-dominant palm/index base.", "movement": False},
    "U": {"type": "Two-handed", "description": "Dominant index touches pinky finger tip of non-dominant open hand (vowel contact).", "movement": False},
    "V": {"type": "Single/Two-handed", "description": "Dominant index and middle extended in 'V' shape resting on non-dominant palm.", "movement": False},
    "W": {"type": "Two-handed", "description": "Fingers of both hands interlocked pointing up/forward in a 'W' crown.", "movement": False},
    "X": {"type": "Two-handed", "description": "Both index fingers crossed at right angles to form an 'X'.", "movement": False},
    "Y": {"type": "Two-handed", "description": "Dominant index placed into the 'V' groove between non-dominant thumb and index.", "movement": False},
    "Z": {"type": "Dynamic (Movement)", "description": "Dominant index traces a 'Z' zigzag path or dynamic diagonal palm contact.", "movement": True},
}


def audit_archive_structure(zip_path: Path) -> Dict[str, Any]:
    print(f"[INFO] Opening archive: {zip_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_names: Dict[str, List[str]] = defaultdict(list)
    total_images = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        infolist = z.infolist()
        print(f"[INFO] Total zip entries: {len(infolist):,}")

        for info in infolist:
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) >= 3 and parts[0] in ("Training", "Testing", "Validation"):
                split = parts[0]
                cls_name = parts[1]
                counts[split][cls_name] += 1
                total_images += 1
                if len(sample_names[f"{split}/{cls_name}"]) < 3:
                    sample_names[f"{split}/{cls_name}"].append(parts[-1])
            elif "Letters" in parts:
                counts["Reference_Letters"][parts[-1]] += 1

    return {
        "total_images": total_images,
        "counts": {k: dict(v) for k, v in counts.items()},
        "sample_filenames": dict(sample_names),
    }


def analyze_filename_conventions(zip_path: Path) -> Dict[str, Any]:
    """Examine image filenames across splits to determine if signer metadata exists."""
    print("[INFO] Analyzing filename patterns...")
    patterns = set()
    sample_files = []
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist()[:1000]:
            if not info.is_dir() and Path(info.filename).suffix.lower() in (".jpg", ".jpeg", ".png"):
                stem = Path(info.filename).stem
                patterns.add(stem.split("_")[0] if "_" in stem else "plain_numeric")
                if len(sample_files) < 20:
                    sample_files.append(Path(info.filename).name)

    return {
        "observed_prefixes": list(patterns),
        "sample_names": sample_files,
        "signer_id_in_filename": any("signer" in s.lower() or "s0" in s.lower() for s in sample_files),
    }


def extract_reference_images(zip_path: Path, dest_dir: Path) -> List[Path]:
    print(f"[INFO] Extracting reference letters to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if "Letters" in info.filename and not info.is_dir():
                filename = Path(info.filename).name
                out_path = dest_dir / filename
                with open(out_path, "wb") as f:
                    f.write(z.read(info))
                extracted_paths.append(out_path)
                print(f"  Extracted: {filename}")

    return sorted(extracted_paths)


def pad_to_square(img: np.ndarray, target_dim: int = 640) -> np.ndarray:
    """Pad image to fixed square canvas to satisfy MediaPipe static dimension requirements."""
    h, w = img.shape[:2]
    scale = min(target_dim / max(h, 1), target_dim / max(w, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_dim, target_dim, 3), dtype=np.uint8)
    y_off = (target_dim - new_h) // 2
    x_off = (target_dim - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def inspect_reference_letters_with_landmarker(reference_dir: Path) -> List[Dict[str, Any]]:
    print("[INFO] Initializing MediaPipe Holistic landmarker for reference audit...")
    landmarker = HolisticLandmarkerWrapper(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    
    results = []
    for letter_file in sorted(reference_dir.glob("*.*")):
        letter = letter_file.stem.upper()
        if len(letter) > 1 and not letter.isalpha():
            continue
        
        orig_img = cv2.imread(str(letter_file))
        if orig_img is None:
            continue
        
        h_orig, w_orig = orig_img.shape[:2]
        padded_img = pad_to_square(orig_img, 640)
        
        try:
            raw_lms = landmarker.detect(padded_img)
            has_lh = raw_lms.left_hand_landmarks is not None and len(raw_lms.left_hand_landmarks) > 0
            has_rh = raw_lms.right_hand_landmarks is not None and len(raw_lms.right_hand_landmarks) > 0
            has_pose = raw_lms.pose_landmarks is not None and len(raw_lms.pose_landmarks) > 0
            num_hands = (1 if has_lh else 0) + (1 if has_rh else 0)
        except Exception as ex:
            print(f"  [WARN] Detection error on '{letter}': {ex}")
            has_lh = False
            has_rh = False
            has_pose = False
            num_hands = 0
        
        islrtc_info = ISLRTC_STANDARDS.get(letter, {"type": "Unknown", "description": "N/A", "movement": False})
        
        results.append({
            "letter": letter,
            "filename": letter_file.name,
            "image_dimensions": f"{w_orig}x{h_orig}",
            "islrtc_expected_type": islrtc_info["type"],
            "islrtc_description": islrtc_info["description"],
            "requires_movement": islrtc_info["movement"],
            "mediapipe_left_hand_detected": has_lh,
            "mediapipe_right_hand_detected": has_rh,
            "mediapipe_hands_count": num_hands,
            "mediapipe_pose_detected": has_pose,
            "is_two_handed_in_image": num_hands == 2,
        })
        print(f"  Letter '{letter}': {num_hands} hand(s) detected (LH: {has_lh}, RH: {has_rh}) | ISLRTC: {islrtc_info['type']}")

    landmarker.close()
    return results


def main():
    if not ZIP_PATH.exists():
        print(f"[ERROR] {ZIP_PATH} does not exist yet. Please wait for download.", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("SETU — RealSign ISL Alphabet Dataset Audit")
    print("==================================================")
    
    structure_info = audit_archive_structure(ZIP_PATH)
    filename_info = analyze_filename_conventions(ZIP_PATH)
    ref_paths = extract_reference_images(ZIP_PATH, REFERENCE_DIR)
    ref_audit = inspect_reference_letters_with_landmarker(REFERENCE_DIR)

    report = {
        "dataset_name": "RealSign-Indian-Sign-Language-Dataset",
        "archive_path": str(ZIP_PATH.resolve()),
        "archive_size_bytes": ZIP_PATH.stat().st_size,
        "total_images": structure_info["total_images"],
        "class_counts_by_split": structure_info["counts"],
        "filename_analysis": filename_info,
        "reference_letters_count": len(ref_paths),
        "reference_letters_audit": ref_audit,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Audit complete. Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
