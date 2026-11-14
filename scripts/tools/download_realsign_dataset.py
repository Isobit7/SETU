"""
SETU — RealSign ISL Alphabet Dataset Downloader & Integrity Verifier
Downloads Dataset.zip from RealSign62/RealSign-Indian-Sign-Language-Dataset via Git LFS media endpoint.
Verifies SHA-256 checksum and ZIP file integrity without modifying any existing project files.
"""

import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://media.githubusercontent.com/media/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Dataset.zip"
EXPECTED_SIZE = 656689688
EXPECTED_SHA256 = "008cae248e346b8c31fbbea057fcc3f69c6909d29a88e6bb1fb0369f528de2b5"
TARGET_DIR = Path("data/raw/realsign")
TARGET_ZIP = TARGET_DIR / "Dataset.zip"


def download_with_progress(url: str, dest_path: Path, expected_size: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already fully downloaded
    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"[INFO] Archive already exists ({dest_path.stat().st_size:,} bytes). Skipping download.")
        return

    temp_path = dest_path.with_suffix(".tmp")
    headers = {"User-Agent": "SETU-Dataset-Audit/1.0"}
    
    print(f"[INFO] Starting download from: {url}")
    print(f"[INFO] Destination: {dest_path}")
    print(f"[INFO] Expected size: {expected_size:,} bytes (~{expected_size / 1024 / 1024:.2f} MB)")
    
    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    downloaded = 0
    last_print = 0
    
    with urllib.request.urlopen(req, timeout=120) as resp, open(temp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)  # 1 MB chunk
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 2.0 or downloaded == expected_size:
                elapsed = max(0.1, now - start_time)
                speed_mb = (downloaded / 1024 / 1024) / elapsed
                percent = (downloaded / expected_size) * 100
                print(f"  [Progress] {downloaded:,} / {expected_size:,} bytes ({percent:.1f}%) | Speed: {speed_mb:.2f} MB/s")
                last_print = now

    temp_path.replace(dest_path)
    print(f"[INFO] Download completed in {time.time() - start_time:.1f}s.")


def verify_integrity(dest_path: Path, expected_sha256: str) -> str:
    print(f"[INFO] Computing SHA-256 for {dest_path}...")
    h = hashlib.sha256()
    with open(dest_path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    actual_sha256 = h.hexdigest()
    print(f"[INFO] Actual SHA-256:   {actual_sha256}")
    print(f"[INFO] Expected SHA-256: {expected_sha256}")
    
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"Checksum mismatch! Expected {expected_sha256}, got {actual_sha256}")
    print("[SUCCESS] SHA-256 checksum matches perfectly!")

    print("[INFO] Testing ZIP archive integrity with zipfile.testzip()...")
    with zipfile.ZipFile(dest_path, "r") as z:
        bad_file = z.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"Corrupted entry in zip: {bad_file}")
        print(f"[SUCCESS] ZIP archive integrity verified. Total entries: {len(z.infolist())}")

    return actual_sha256


def main():
    try:
        download_with_progress(DATASET_URL, TARGET_ZIP, EXPECTED_SIZE)
        actual_hash = verify_integrity(TARGET_ZIP, EXPECTED_SHA256)
        print(f"\n[DONE] RealSign Dataset.zip successfully acquired and verified.")
        print(f"Path: {TARGET_ZIP.resolve()}")
        print(f"Size: {TARGET_ZIP.stat().st_size:,} bytes")
        print(f"SHA-256: {actual_hash}")
    except Exception as e:
        print(f"[ERROR] Failed to acquire or verify dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
