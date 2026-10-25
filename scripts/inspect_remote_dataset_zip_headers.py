import io
import urllib.request
import zipfile

class RangeHTTPFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        length = len(b)
        if self.pos >= self.size:
            return 0
        end = min(self.pos + length - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.pos}-{end}",
            "User-Agent": "SETU-ISL/1.0"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            b[:len(data)] = data
            self.pos += len(data)
            return len(data)

url = "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content"

# 1. Get exact Content-Length
req = urllib.request.Request(url, headers={"User-Agent": "SETU-ISL/1.0"}, method="HEAD")
with urllib.request.urlopen(req, timeout=15) as resp:
    size = int(resp.headers.get("Content-Length"))
    accept_ranges = resp.headers.get("Accept-Ranges")
    print(f"Archive Size: {size} bytes ({size / (1024*1024):.2f} MB), Accept-Ranges: {accept_ranges}")

# 2. Open ZIP using remote seekable stream
stream = RangeHTTPFile(url, size)
zf = zipfile.ZipFile(stream)
print("Listing files in Greetings_1of2.zip...")
namelist = zf.namelist()
print(f"Total entries in archive: {len(namelist)}")

# Find hello and thank you
hello_files = [n for n in namelist if "Hello" in n and n.endswith((".MOV", ".mov", ".mp4"))]
print(f"\nFound {len(hello_files)} Hello video files:")
for hf in hello_files[:10]:
    info = zf.getinfo(hf)
    print(f"  {hf} ({info.file_size / (1024*1024):.2f} MB)")

thank_files = [n for n in namelist if "Thank" in n and n.endswith((".MOV", ".mov", ".mp4"))]
print(f"\nFound {len(thank_files)} Thank you video files:")
for tf in thank_files[:10]:
    info = zf.getinfo(tf)
    print(f"  {tf} ({info.file_size / (1024*1024):.2f} MB)")
