"""
SETU - Selective AI4Bharat INCLUDE Video Acquisition
Direct HTTP-Range extraction of genuine ISL video files from Zenodo Record 4010759.
Extracts individual compressed video streams in a single HTTP 206 request and decompresses
via zlib, preserving the original raw .MOV video files in data/raw/include/.
"""

import io
import struct
import urllib.request
import zipfile
import zlib
import hashlib
from pathlib import Path
from typing import Dict, Any, List

class RangeHTTPFile(io.RawIOBase):
    """
    Seekable wrapper around HTTP Range requests for reading the ZIP central directory.
    """
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        length = len(b)
        if self.pos >= self.size:
            return 0
        end = min(self.pos + length - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.pos}-{end}",
            "User-Agent": "SETU-ISL-DataAcquisition/1.0"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            b[:len(data)] = data
            self.pos += len(data)
            return len(data)


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "SETU-ISL-DataAcquisition/1.0"}, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length"))


def extract_member_direct(url: str, info: zipfile.ZipInfo, dest_path: Path) -> Path:
    """
    Extracts a zip member using two minimal HTTP range requests:
    1. Read 30-byte local header to determine exact data offset (including filename & extra length)
    2. Read the entire compressed payload in a single HTTP range request
    3. Decompress in memory using zlib and save to disk
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Read local file header
    req_hdr = urllib.request.Request(url, headers={
        "Range": f"bytes={info.header_offset}-{info.header_offset + 29}",
        "User-Agent": "SETU-ISL-DataAcquisition/1.0"
    })
    with urllib.request.urlopen(req_hdr, timeout=30) as resp:
        hdr_bytes = resp.read()
    
    sig, ver, flag, method, time_, date_, crc, csize, usize, fn_len, extra_len = struct.unpack('<IHHHHHIIIHH', hdr_bytes)
    data_offset = info.header_offset + 30 + fn_len + extra_len
    
    # 2. Download compressed stream with retry on transient socket drops
    csize_to_read = csize if csize > 0 else info.compress_size
    max_retries = 3
    compressed_bytes = None
    for attempt in range(max_retries):
        try:
            req_data = urllib.request.Request(url, headers={
                "Range": f"bytes={data_offset}-{data_offset + csize_to_read - 1}",
                "User-Agent": "SETU-ISL-DataAcquisition/1.0"
            })
            with urllib.request.urlopen(req_data, timeout=120) as resp:
                buf = bytearray()
                while len(buf) < csize_to_read:
                    chunk = resp.read(min(1024 * 1024, csize_to_read - len(buf)))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) == csize_to_read:
                    compressed_bytes = bytes(buf)
                    break
                print(f"      [Retry {attempt+1}/{max_retries}] Read {len(buf)}/{csize_to_read} bytes, retrying...")
        except Exception as ex:
            print(f"      [Retry {attempt+1}/{max_retries}] Network exception {ex}, retrying in 3s...")
            import time
            time.sleep(3)
            
    if compressed_bytes is None or len(compressed_bytes) != csize_to_read:
        raise IOError(f"Failed to fully read {csize_to_read} bytes after {max_retries} attempts.")
        
    # 3. Decompress
    if method == 8:
        raw_data = zlib.decompress(compressed_bytes, -15)
    elif method == 0:
        raw_data = compressed_bytes
    else:
        raise ValueError(f"Unsupported zip compression method {method} for {info.filename}")
        
    with open(dest_path, "wb") as f:
        f.write(raw_data)
        
    return dest_path


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# Verified target genuine ISL subset from AI4Bharat INCLUDE (Zenodo Record 4010759)
TARGETS = [
    {
        "class": "hello",
        "class_id": 0,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/48. Hello/MVI_0029.MOV",
        "signer_session": "signer_S01",
        "dest_filename": "hello_s01_mvi_0029.mov"
    },
    {
        "class": "hello",
        "class_id": 0,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/48. Hello/MVI_0089.MOV",
        "signer_session": "signer_S02",
        "dest_filename": "hello_s02_mvi_0089.mov"
    },
    {
        "class": "hello",
        "class_id": 0,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/48. Hello/MVI_9914.MOV",
        "signer_session": "signer_S03",
        "dest_filename": "hello_s03_mvi_9914.mov"
    },
    {
        "class": "thank_you",
        "class_id": 1,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content",
        "archive_name": "Greetings_2of2.zip",
        "internal_path": "Greetings/55. Thank you/MVI_0006.MOV",
        "signer_session": "signer_S04",
        "dest_filename": "thank_you_s04_mvi_0006.mov"
    },
    {
        "class": "thank_you",
        "class_id": 1,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content",
        "archive_name": "Greetings_2of2.zip",
        "internal_path": "Greetings/55. Thank you/MVI_0058.MOV",
        "signer_session": "signer_S05",
        "dest_filename": "thank_you_s05_mvi_0058.mov"
    },
    {
        "class": "thank_you",
        "class_id": 1,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content",
        "archive_name": "Greetings_2of2.zip",
        "internal_path": "Greetings/55. Thank you/MVI_9948.MOV",
        "signer_session": "signer_S06",
        "dest_filename": "thank_you_s06_mvi_9948.mov"
    },
    {
        "class": "good_morning",
        "class_id": 2,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/51. Good Morning/MVI_0047.MOV",
        "signer_session": "signer_S07",
        "dest_filename": "good_morning_s07_mvi_0047.mov"
    },
    {
        "class": "good_morning",
        "class_id": 2,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/51. Good Morning/MVI_0099.MOV",
        "signer_session": "signer_S08",
        "dest_filename": "good_morning_s08_mvi_0099.mov"
    },
    {
        "class": "good_morning",
        "class_id": 2,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content",
        "archive_name": "Greetings_1of2.zip",
        "internal_path": "Greetings/51. Good Morning/MVI_9971.MOV",
        "signer_session": "signer_S09",
        "dest_filename": "good_morning_s09_mvi_9971.mov"
    },
    {
        "class": "i",
        "class_id": 3,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_1of2.zip/content",
        "archive_name": "Pronouns_1of2.zip",
        "internal_path": "Pronouns/40. I/MVI_0013.MOV",
        "signer_session": "signer_S10",
        "dest_filename": "i_s10_mvi_0013.mov"
    },
    {
        "class": "i",
        "class_id": 3,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_1of2.zip/content",
        "archive_name": "Pronouns_1of2.zip",
        "internal_path": "Pronouns/40. I/MVI_9922.MOV",
        "signer_session": "signer_S11",
        "dest_filename": "i_s11_mvi_9922.mov"
    },
    {
        "class": "i",
        "class_id": 3,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_1of2.zip/content",
        "archive_name": "Pronouns_1of2.zip",
        "internal_path": "Pronouns/40. I/MVI_0014.MOV",
        "signer_session": "signer_S12",
        "dest_filename": "i_s12_mvi_0014.mov"
    },
    {
        "class": "you",
        "class_id": 4,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_2of2.zip/content",
        "archive_name": "Pronouns_2of2.zip",
        "internal_path": "Pronouns/46. you (plural)/MVI_0022.MOV",
        "signer_session": "signer_S13",
        "dest_filename": "you_s13_mvi_0022.mov"
    },
    {
        "class": "you",
        "class_id": 4,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_2of2.zip/content",
        "archive_name": "Pronouns_2of2.zip",
        "internal_path": "Pronouns/46. you (plural)/MVI_0032.MOV",
        "signer_session": "signer_S14",
        "dest_filename": "you_s14_mvi_0032.mov"
    },
    {
        "class": "you",
        "class_id": 4,
        "archive_url": "https://zenodo.org/api/records/4010759/files/Pronouns_2of2.zip/content",
        "archive_name": "Pronouns_2of2.zip",
        "internal_path": "Pronouns/46. you (plural)/MVI_9907.MOV",
        "signer_session": "signer_S15",
        "dest_filename": "you_s15_mvi_9907.mov"
    }
]


def download_subset(raw_dir: Path) -> List[Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading genuine INCLUDE video clips to {raw_dir}...")

    # Group targets by archive URL
    archives = {}
    for t in TARGETS:
        archives.setdefault(t["archive_url"], []).append(t)

    results = []
    for archive_url, targets in archives.items():
        print(f"\nAccessing remote archive: {targets[0]['archive_name']}...")
        size = get_remote_file_size(archive_url)
        print(f"  Remote size: {size / (1024*1024):.2f} MB. Reading central directory...")
        stream = RangeHTTPFile(archive_url, size)
        zf = zipfile.ZipFile(stream)

        for t in targets:
            dest = raw_dir / t["dest_filename"]
            # Check if file exists and has valid size
            info = zf.getinfo(t["internal_path"])
            if dest.exists() and dest.stat().st_size == info.file_size:
                print(f"  [Verified Exists] {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
            else:
                print(f"  [Downloading Direct Range] {t['internal_path']} ({info.file_size / (1024*1024):.2f} MB) -> {dest.name}...")
                extract_member_direct(archive_url, info, dest)
                print(f"    Saved: {dest.stat().st_size / (1024*1024):.2f} MB")

            sha256 = compute_sha256(dest)
            results.append({
                "class": t["class"],
                "class_id": t["class_id"],
                "signer_session": t["signer_session"],
                "internal_path": t["internal_path"],
                "local_path": str(dest),
                "size_bytes": dest.stat().st_size,
                "sha256": sha256
            })
            print(f"    SHA-256: {sha256}")

    print(f"\nAll {len(results)} target videos verified on disk.")
    return results


if __name__ == "__main__":
    from configs.settings import PROJECT_ROOT
    dest_dir = PROJECT_ROOT / "data" / "raw" / "include"
    download_subset(dest_dir)
