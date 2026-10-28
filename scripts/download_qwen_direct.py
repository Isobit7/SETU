"""
SETU — Direct Streaming Downloader for Qwen2.5-0.5B-Instruct
Downloads weights directly via HTTP stream into models/weights/qwen_0.5b
with progress tracking, avoiding git-lfs / hf-xet stalls.
"""

import os
import sys
import time
from pathlib import Path
import requests

BASE_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"
TARGET_DIR = Path("models/weights/qwen_0.5b")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


def download_file(filename: str) -> None:
    dest = TARGET_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        # If it's model.safetensors, check size
        if filename == "model.safetensors" and dest.stat().st_size >= 900_000_000:
            print(f"[SKIP] {filename} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        elif filename != "model.safetensors":
            print(f"[SKIP] {filename} already exists")
            return

    url = f"{BASE_URL}/{filename}"
    print(f"\n[DOWNLOADING] {filename} from {url}...")
    t0 = time.time()

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int((downloaded / total_size) * 100)
                        if pct != last_pct and pct % 10 == 0:
                            mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            elapsed = time.time() - t0
                            speed = mb / max(elapsed, 0.1)
                            print(f" -> {pct}% ({mb:.1f}/{total_mb:.1f} MB) @ {speed:.2f} MB/s", flush=True)
                            last_pct = pct

    print(f"[DONE] {filename} saved in {time.time() - t0:.1f}s ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    print(f"Downloading Qwen2.5-0.5B-Instruct to {TARGET_DIR.resolve()}...")
    for fn in FILES:
        download_file(fn)
    print("\nAll model files downloaded successfully!")


if __name__ == "__main__":
    main()

