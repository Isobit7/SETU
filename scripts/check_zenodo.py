import urllib.request
import json

url = "https://zenodo.org/api/records/4010759"
req = urllib.request.Request(url, headers={"User-Agent": "SETU-ISL/1.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print("Title:", data.get("metadata", {}).get("title"))
print("License:", data.get("metadata", {}).get("license"))
print("Creators:", [c.get("name") for c in data.get("metadata", {}).get("creators", [])])
print("Publication Date:", data.get("metadata", {}).get("publication_date"))

files = data.get("files", [])
print(f"Total files: {len(files)}")
for f in files:
    name = f.get("key")
    size_mb = f.get("size", 0) / (1024 * 1024)
    download_url = f.get("links", {}).get("self")
    print(f"  {name:30s} | {size_mb:8.2f} MB | {download_url}")
