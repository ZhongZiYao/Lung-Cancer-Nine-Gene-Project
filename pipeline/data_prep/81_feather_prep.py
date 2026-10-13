import requests, json
S = requests.Session(); S.trust_env = False
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"
print("=== FEATHER collection 里的仓库 ===", flush=True)
try:
    r = S.get(B + "/api/collections/MahmoodLab/feather-6875570e0c755f6c9128a85d", headers=H, timeout=40)
    print("HTTP", r.status_code, flush=True)
    if r.status_code == 200:
        for it in r.json().get("items", []):
            print("   %-46s type=%s" % (it.get("id"), it.get("type")), flush=True)
except Exception as e:
    print("ERR", str(e)[:80], flush=True)
print()
print("=== 我们本地 MIL/CMIL 的已有实现在哪 ===", flush=True)
import os
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
for root, dirs, files in os.walk(os.path.join(R, "pipeline")):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    for f in files:
        if f.endswith(".py"):
            fp = os.path.join(root, f)
            try:
                t = open(fp, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            if "class CLAM" in t or "class Attn_Net" in t or "GatedAttention" in t or "ABMIL" in t or "abmil" in t.lower():
                print("   %s  (%d lines)" % (os.path.relpath(fp, R), t.count(chr(10))), flush=True)