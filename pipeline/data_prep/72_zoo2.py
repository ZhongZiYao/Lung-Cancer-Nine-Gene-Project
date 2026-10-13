import sys, os, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"

def ls(repo, path):
    u = B + "/api/datasets/%s/tree/main/%s" % (repo, path) if path else B + "/api/datasets/%s/tree/main" % repo
    for _ in range(3):
        try:
            r = requests.get(u, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception: pass
    return []

def walk(repo, path="", depth=0, maxdepth=3):
    out = []
    for it in ls(repo, path):
        p = it["path"]
        if it["type"] == "directory":
            if depth < maxdepth:
                out += walk(repo, p, depth + 1, maxdepth)
        else:
            out.append((p, it.get("size") or 0))
    return out

REPO = "Dearcat/CPathPatchFeature"
print("=== Dearcat/CPathPatchFeature 完整树 ===", flush=True)
agg = defaultdict(lambda: [0, 0])
items = walk(REPO, maxdepth=3)
for p, sz in items:
    parts = p.split("/")
    key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts[:2])
    agg[key][0] += 1; agg[key][1] += sz
for k in sorted(agg):
    n, s = agg[k]
    print("  %-30s %5d files  %8.1f GB" % (k, n, s/1e9), flush=True)
print("  TOTAL %d files, %.1f GB" % (len(items), sum(s for _, s in items)/1e9), flush=True)

print("\n=== cptac_nsclc 明细(文件名前缀) ===", flush=True)
cp = [(p, s) for p, s in items if p.startswith("cptac_nsclc")]
pre = defaultdict(lambda: [0, 0])
for p, s in cp:
    key = "/".join(p.split("/")[:3])
    pre[key][0] += 1; pre[key][1] += s
for k in sorted(pre): print("  %-34s %5d files %8.1f GB" % (k, pre[k][0], pre[k][1]/1e9), flush=True)
print("  样例文件名:", [p.split("/")[-1] for p, _ in cp[:6]], flush=True)

print("\n=== HF 搜索: 病理预提取特征 ===", flush=True)
seen = set()
for q in ["wsi features", "patch feature", "pathology feature extraction", "slide embedding",
          "TCGA WSI", "histopathology embedding", "CPathPatchFeature", "LUAD", "LUSC", "UNI2-h"]:
    try:
        r = requests.get(B + "/api/datasets", params={"search": q, "limit": 15}, headers=H, timeout=40)
        if r.status_code != 200: continue
        for d in r.json():
            i = d.get("id")
            if i in seen: continue
            seen.add(i)
            print("  %-50s gated=%-6s dl=%s" % (i, d.get("gated"), d.get("downloads")), flush=True)
    except Exception as e:
        print("  [%s] ERR %s" % (q, str(e)[:50]), flush=True)