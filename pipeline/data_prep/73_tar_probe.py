import sys, io, tarfile, requests, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter, defaultdict
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"

print("=== MahmoodLab/UNI2-h-features : CPTAC / TCGA 文件清单 ===", flush=True)
for sub in ["CPTAC", "TCGA"]:
    r = requests.get(B + "/api/datasets/MahmoodLab/UNI2-h-features/tree/main/" + sub, headers=H, timeout=60)
    if r.status_code != 200:
        print("  HTTP", r.status_code, flush=True); continue
    for it in r.json():
        if any(k in it["path"].lower() for k in ["nsclc", "luad", "lusc", "lung"]):
            print("  %-34s %8.1f MB" % (it["path"], (it.get("size") or 0)/1e6), flush=True)

print("\n=== LeonYe cptac_virchow_v2.tar 流式读前若干成员 ===", flush=True)
url = B + "/datasets/LeonYe/WSI-CPTAC-LUNG-VIRCHOW-V2-Features/resolve/main/cptac_virchow_v2.tar"
s = requests.Session(); s.trust_env = False
r = s.get(url, headers=dict(H, Range="bytes=0-33554431"), stream=True, timeout=(30, 120))
print("  HTTP", r.status_code, " 首个 32MB 已取", flush=True)
buf = io.BytesIO()
for ch in r.iter_content(1 << 20):
    buf.write(ch)
    if buf.tell() >= (32 << 20): break
r.close()
data = buf.getvalue()
names, sizes = [], []
off = 0
while off + 512 <= len(data):
    hdr = data[off:off + 512]
    if hdr[:1] == b"\0": break
    name = hdr[:100].split(b"\0")[0].decode("utf-8", "replace")
    try: size = int(hdr[124:136].split(b"\0")[0].strip() or b"0", 8)
    except Exception: size = 0
    typ = hdr[156:157].decode()
    if typ in "0125" and name:
        names.append(name); sizes.append(size)
    off += 512 + ((size + 511) // 512) * 512
print("  前 32MB 内解析到 %d 个成员" % len(names), flush=True)
for n in names[:15]:
    print("     %s" % n, flush=True)
pref = Counter()
for n in names:
    parts = n.split("/")
    pref["/".join(parts[:2])] += 1
print("\n  前缀分布(前32MB):", dict(list(pref.items())[:10]), flush=True)
allp = "/".join(names[0].split("/")[:1]) if names else ""
print("\n  尝试列 tar 全部成员名(只读 header, 最多 400 个) ===", flush=True)