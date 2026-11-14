"""
SETU — Project Packager for Gemini.zip
Generates a clean root-level zip archive containing all source code, configs,
models, training scripts, database files, and documentation while excluding
virtual environments, build directories, heavy third-party caches, and legacy archives.
"""

import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".agent",
    "setu.egg-info",
    "archive",      # in docs/archive/ (old backup zips)
    "NEW ZIP",      # in data/raw/NEW ZIP/ (redundant 2.4GB external image dump)
    "realsign",     # in data/raw/realsign/ (31k uncompressed raw images; features in data/processed)
}

EXCLUDE_FILES = {
    "Gemini.zip",
    ".coverage",
}


def should_exclude(file_path: Path) -> bool:
    rel_parts = file_path.relative_to(PROJECT_ROOT).parts
    for part in rel_parts:
        if part in EXCLUDE_DIRS:
            return True
    if file_path.name in EXCLUDE_FILES:
        return True
    if file_path.suffix == ".pyc":
        return True
    return False


def create_gemini_zip(output_path: Path) -> None:
    print(f"[INFO] Scanning {PROJECT_ROOT} for files to package...")
    files_to_pack = []
    total_bytes = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            full_path = Path(root) / f
            if not should_exclude(full_path):
                files_to_pack.append(full_path)
                total_bytes += full_path.stat().st_size

    print(f"[INFO] Total files to package: {len(files_to_pack)} ({total_bytes / (1024*1024):.2f} MB)")
    print(f"[INFO] Creating zip archive at: {output_path}")

    start_time = time.time()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for idx, file_path in enumerate(files_to_pack, 1):
            arcname = file_path.relative_to(PROJECT_ROOT)
            zipf.write(file_path, arcname=arcname)
            if idx % 50 == 0 or idx == len(files_to_pack):
                print(f"[PROGRESS] Added {idx}/{len(files_to_pack)} files ({arcname})")

    elapsed = time.time() - start_time
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully generated '{output_path.name}' in {elapsed:.1f}s!")
    print(f"[INFO] Final Archive Size: {zip_size_mb:.2f} MB")
    print(f"[INFO] Absolute Path: {output_path}")


if __name__ == "__main__":
    out_zip = PROJECT_ROOT / "Gemini.zip"
    create_gemini_zip(out_zip)
