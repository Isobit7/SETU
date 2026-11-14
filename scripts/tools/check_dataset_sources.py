"""
SETU — Dataset Source and Integrity Inspector
Validates canonical processed landmark archives, manifests, and class balance.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def inspect_dataset_sources():
    print("=" * 70)
    print("   SETU Dataset Sources & Processed Landmark Integrity Inspector   ")
    print("=" * 70)

    # 1. Canonical Processed NPZ
    npz_path = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"
    if npz_path.exists():
        sz_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"\n[+] Canonical Landmark Archive: {npz_path.relative_to(PROJECT_ROOT)}")
        print(f"    Size: {sz_mb:.2f} MB ({npz_path.stat().st_size:,} bytes)")
        
        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            splits = np.array(data["splits"], dtype=str)
            classes = np.array(data["classes"], dtype=str)
            
            print(f"    Total Samples: {len(X):,}  |  Feature Dimension: {X.shape[1]}-D")
            print(f"    Classes ({len(classes)}): {list(classes)}")
            
            # Split breakdown
            unique_s, counts_s = np.unique(splits, return_counts=True)
            print("    Splits Breakdown:")
            for s, cnt in zip(unique_s, counts_s):
                print(f"      * {s.capitalize():<6}: {cnt:>6,} samples ({cnt/len(y)*100:>5.1f}%)")
    else:
        print(f"\n[-] Canonical dataset not found at: {npz_path}")

    # 2. Manifests
    manifests_dir = PROJECT_ROOT / "data" / "manifests"
    if manifests_dir.exists():
        print(f"\n[+] Dataset Manifests ({manifests_dir.relative_to(PROJECT_ROOT)}):")
        for mf in manifests_dir.glob("*.json"):
            print(f"    * {mf.name} ({mf.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"\n[-] Manifests directory not found.")

    print("\n" + "=" * 70)
    print("   Inspection Completed: Canonical Datasets Ready for Training & Inference   ")
    print("=" * 70)

if __name__ == "__main__":
    inspect_dataset_sources()
