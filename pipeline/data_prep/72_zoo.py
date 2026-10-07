import sys, os, json, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"

print("=== 本地已有的预提取特征缓存 ===", flush=True)
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
import pandas as pd
for f in sorted(os.listdir(C)):
    if not f.endswith("_index.csv"): continue
    d = pd.read_csv(os.path.join(C, f))
    npy = os.path.join(C, f.replace("_index.csv", ".npy"))
    sz = os.path.getsize(npy)/1e6 if os.path.exists(npy) else 0
    print("  %-34s %5d case  %7.0f MB  cols=%s" % (f, len(d), sz, [x for x in d.columns][:6]), flush=True)

print("\n=== HF: Dearcat/CPathPatchFeature 全树 ===", flush=True)
def tree(repo, path=""):
    u = B + "/api/datasets/%s/tree/main/%s?recursive=true&expand=true" % (repo, path)
    r = requests.get(u, headers=H, timeout=60)
    if r.status_code != 200:
        print("  HTTP", r.status_code, r.text[:80], flush=True); return []
    return r.json()

items = tree("Dearcat/CPathPatchFeature")
from collections import defaultdict
byd = defaultdict(lambda: [0, 0])
for it in items:
    p = it.get("path", "")
    if it.get("type") != "file": continue
    parts = p.split("/")
    if len(parts) < 2: continue
    key = "/".join(parts[:2])
    byd[key][0] += 1
    byd[key][1] += it.get("size") or 0
for k in sorted(byd):
    n, s = byd[k]
    print("  %-28s %5d files  %8.1f GB" % (k, n, s/1e9), flush=True)

print("\n=== HF: 搜病理 patch 特征数据集 ===", flush=True)
for q in ["wsi patch features", "pathology features", "TCGA features histopathology",
          "LUAD features", "slide embeddings", "patch embeddings pathology"]:
    try:
        r = requests.get(B + "/api/datasets", params={"search": q, "limit": 12, "full": "true"}, headers=H, timeout=40)
        if r.status_code != 200:
            print("  [%s] HTTP %d" % (q, r.status_code), flush=True); continue
        for d in r.json():
            print("  %-46s gated=%-6s dl=%s" % (d.get("id"), d.get("gated"), d.get("downloads")), flush=True)
    except Exception as e:
        print("  [%s] ERR %s" % (q, str(e)[:60]), flush=True)
    print("", flush=True)