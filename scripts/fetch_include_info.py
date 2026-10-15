import urllib.request
import json

req = urllib.request.Request("https://api.github.com/repos/ai4bharat/include/contents", headers={"User-Agent": "SETU-ISL/1.0"})
with urllib.request.urlopen(req, timeout=10) as resp:
    items = json.loads(resp.read().decode("utf-8"))

for item in items:
    print(f"  {item['name']} ({item['type']})")

# Check if there is a data or splits directory
for item in items:
    if item['type'] == 'dir':
        sub_req = urllib.request.Request(item['url'], headers={"User-Agent": "SETU-ISL/1.0"})
        with urllib.request.urlopen(sub_req, timeout=10) as sub_resp:
            sub_items = json.loads(sub_resp.read().decode("utf-8"))
            print(f"\nInside {item['name']}:")
            for si in sub_items:
                print(f"    {si['name']}")
