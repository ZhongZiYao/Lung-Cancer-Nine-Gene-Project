import requests
S = requests.Session(); S.trust_env = False
TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com"
for repo in ["MahmoodLab/abmil.base.uni.pc108-24k",
             "MahmoodLab/abmil.base.uni_v2.pc108-24k",
             "MahmoodLab/abmil.base.conch_v15.pc108-24k"]:
    r = S.get(B + "/api/models/" + repo + "?blobs=true", headers=H, timeout=40)
    print("=== %s -> %s" % (repo, r.status_code), flush=True)
    if r.status_code != 200:
        print("   ", r.text[:120], flush=True); continue
    j = r.json()
    print("   gated=%s" % j.get("gated"), flush=True)
    for s in j.get("siblings", []):
        print("   %-44s %8.1f MB" % (s["rfilename"], (s.get("size") or 0)/1e6), flush=True)