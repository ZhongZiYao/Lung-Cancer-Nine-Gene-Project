import sys, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"
REPO = "prov-gigapath/prov-gigapath-tile-embeddings"

def tree(path=""):
    u = ("%s/api/datasets/%s/tree/main/%s" % (B, REPO, path)) if path else ("%s/api/datasets/%s/tree/main" % (B, REPO))
    for _ in range(3):
        try:
            r = requests.get(u, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception: pass
    return []

meta = requests.get(B + "/api/datasets/" + REPO, headers=H, timeout=40).json()
print("gated=%s  downloads=%s" % (meta.get("gated"), meta.get("downloads")), flush=True)
print("description:", (meta.get("description") or "")[:400].replace("\n", " "), flush=True)
print("card:", requests.get(B + "/datasets/%s/resolve/main/README.md" % REPO, headers=H, timeout=40).text[:1200], flush=True)
print("\n=== 顶层 ===", flush=True)
agg = defaultdict(lambda: [0, 0])
for it in tree():
    if it.get("type") == "file":
        print("  FILE %-40s %.2f GB" % (it["path"], (it.get("size") or 0)/1e9), flush=True)
    else:
        sub = tree(it["path"])
        n = len([x for x in sub if x.get("type") == "file"])
        s = sum(x.get("size") or 0 for x in sub)/1e9
        print("  DIR  %-40s %5d files %7.2f GB" % (it["path"], n, s), flush=True)