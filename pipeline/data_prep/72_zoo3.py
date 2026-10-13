import sys, os, re, requests
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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

print("=== Dearcat/CPathPatchFeature : cptac_nsclc + nsclc ===", flush=True)
for sub in ["cptac_nsclc", "nsclc"]:
    for enc in tree("Dearcat/CPathPatchFeature", sub):
        if enc.get("type") != "directory": continue
        pf = [f for f in tree("Dearcat/CPathPatchFeature", enc["path"]) if f.get("type") == "file"]
        sz = sum(f.get("size") or 0 for f in pf) / 1e9
        names = [f["path"].split("/")[-1] for f in pf]
        cases = set()
        for n in names:
            m = re.match(r"^(.+?)(?:-[A-Z0-9]+)?\.(pt|h5|npy)$", n)
            if m: cases.add(m.group(1))
        print("  %-26s %5d files %7.1f GB  unique-case~%d" % (enc["path"], len(pf), sz, len(cases)), flush=True)
        print("        样例: %s" % names[:3], flush=True)

print("\n=== 其他候选仓库 ===", flush=True)
for repo in ["LeonYe/WSI-CPTAC-LUNG-UNI-V2-Features",
             "LeonYe/WSI-CPTAC-LUNG-VIRCHOW-V2-Features",
             "Eswar1885/tcga-luad-provgigapath-embeddings",
             "Eswar1885/tcga-luad-pathfoundation-embeddings",
             "Gloria0845/cptac_luad", "Gloria0845/luad",
             "huoweicheng/Multi-Embed_LUAD",
             "MahmoodLab/UNI2-h-features",
             "seandavis/tcga_provgigapath_embeddings"]:
    t = tree(repo)
    files = [x for x in t if x.get("type") == "file"]
    tot = sum(x.get("size") or 0 for x in files) / 1e9
    dirs = sorted({x["path"].split("/")[0] for x in t if x.get("type") == "directory"})
    print("  %s" % repo, flush=True)
    print("      files=%d  %.2f GB  subdirs=%s" % (len(files), tot, dirs[:8]), flush=True)
    if files:
        print("      样例: %s" % [x["path"] for x in files[:3]], flush=True)
    for d in dirs[:3]:
        f0 = tree(repo, d)
        print("      %s/ -> %d 项, 样例 %s" % (d, len(f0), [x["path"].split("/")[-1] for x in f0[:3]]), flush=True)

print("\n=== cBioPortal 肺癌/CPTAC 研究 ===", flush=True)
CB = "https://www.cbioportal.org/api"
S = requests.Session(); S.trust_env = False
try:
    st = S.get(CB + "/studies?projection=SUMMARY", timeout=60).json()
    for s in st:
        sid = s.get("studyId", "") or ""
        nm = (s.get("name") or "")
        if any(k in (sid + " " + nm).lower() for k in ["luad", "lusc", "cptac", "nsclc", "lung"]):
            print("  %-42s %s" % (sid, nm[:60]), flush=True)
except Exception as e:
    print("  ERR", str(e)[:100], flush=True)