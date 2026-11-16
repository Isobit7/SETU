import urllib.request
import json

base_raw = "https://raw.githubusercontent.com/ai4bharat/include/master/"

# 1. License
req = urllib.request.Request(base_raw + "LICENSE", headers={"User-Agent": "SETU-ISL/1.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    license_text = resp.read().decode("utf-8")
print("=== AI4Bharat INCLUDE GitHub License ===")
print(license_text[:400])

# 2. Check label map
req = urllib.request.Request(base_raw + "label_maps/label_map_include50.json", headers={"User-Agent": "SETU-ISL/1.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    label_map_50 = json.loads(resp.read().decode("utf-8"))
print("\n=== INCLUDE-50 Label Map (Sample) ===")
print("Total classes in INCLUDE-50:", len(label_map_50))
for k, v in list(label_map_50.items())[:10]:
    print(f"  {k} -> {v}")

# 3. Check train / val / test paths for Hello and Thank you
for split in ["include50_train.txt", "include50_val.txt", "include50_test.txt"]:
    req = urllib.request.Request(base_raw + "train_test_paths/" + split, headers={"User-Agent": "SETU-ISL/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        lines = [l.strip() for l in resp.read().decode("utf-8").splitlines() if l.strip()]
        hello_lines = [l for l in lines if "Hello" in l or "48." in l]
        thank_lines = [l for l in lines if "Thank" in l or "55." in l]
        print(f"\n=== {split} (Total: {len(lines)}) ===")
        print(f"  Hello entries ({len(hello_lines)}):", hello_lines[:3])
        print(f"  Thank you entries ({len(thank_lines)}):", thank_lines[:3])
