import sys, os, re, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"

def tree(repo, path=""):
    u = ("%s/api/datasets/%s/tree/main/%s" % (B, repo, path)) if path else ("%s/api/datasets/%s/tree/main" % (B, repo))
    for _ in range(3):
        try:
            r = requests.get(u, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception: pass
    return []

print("=== cptac_nsclc/uni/pt_files 完整清单 ===", flush=True)
allf, off = [], 0
while True:
    t = tree("Dearcat/CPathPatchFeature", "cptac_nsclc/uni/pt_files")
    allf = [x for x in t if x.get("type") == "file"]
    break
names = [x["path"].split("/")[-1] for x in allf]
print("  文件数 %d, 合计 %.2f GB" % (len(allf), sum(x.get("size") or 0 for x in allf)/1e9), flush=True)
cases = Counter(re.sub(r"-[A-Za-z0-9]+\.(pt|h5)$", "", n) for n in names)
print("  唯一 case(去后缀) = %d" % len(cases), flush=True)
print("  ID 前缀: %s" % dict(Counter(c[:4] for c in cases)), flush=True)
print("  每例 block 数分布: %s" % dict(Counter(cases.values())), flush=True)
print("  样例: %s" % names[:4], flush=True)

import pandas as pd
loc = pd.read_csv(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\CPTAC_uni_sub512_index.csv")
have = set(loc.case_id)
new = sorted(set(cases) - have)
print("\n  本地已有 CPTAC UNI: %d 例 ; 现成特征多出 %d 例" % (len(have), len(new)), flush=True)
if new: print("  新增样例: %s" % new[:10], flush=True)
cl = pd.read_csv(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\cptac_labels_rebuilt.csv")
print("  本地 CPTAC 标签: %d 例 ; 覆盖情况列=%s" % (len(cl), list(cl.columns)[:8]), flush=True)

print("\n=== 搜 CPTAC / 非TCGA 的 Prov-GigaPath / GigaPath 特征 ===", flush=True)
for q in ["cptac gigapath", "cptac features", "gigapath features", "prov-gigapath",
          "cptac lung features", "TCGA CPTAC patch features"]:
    try:
        r = requests.get(B + "/api/datasets", params={"search": q, "limit": 10}, headers=H, timeout=40)
        if r.status_code != 200: continue
        for d in r.json():
            print("  %-48s gated=%-6s dl=%s" % (d.get("id"), d.get("gated"), d.get("downloads")), flush=True)
    except Exception as e:
        print("  [%s] ERR %s" % (q, str(e)[:50]), flush=True)
    print("", flush=True)