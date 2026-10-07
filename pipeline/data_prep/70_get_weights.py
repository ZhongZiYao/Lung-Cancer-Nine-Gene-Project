# -*- coding: utf-8 -*-
"""70_get_weights.py - resumable direct download of Midnight weights (hf_hub stalls on this box)"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests

TOK = open(r"C:\Users\ziyao\.cache\huggingface\token").read().strip()
H = {"Authorization": "Bearer " + TOK}
B = "https://hf-mirror.com/david4real"
DL = {
    "midnight/model.safetensors": ("https://hf-mirror.com/kaiko-ai/midnight/resolve/main/model.safetensors", 4546000000),
    "midnight/config.json": ("https://hf-mirror.com/kaiko-ai/midnight/resolve/main/config.json", -1),
    "phikon/model.safetensors": ("https://hf-mirror.com/owkin/phikon/resolve/main/model.safetensors", 345600000),
    "phikon/preprocessor_config.json": ("https://hf-mirror.com/owkin/phikon/resolve/main/preprocessor_config.json", -1),
    "phikon/config.json": ("https://hf-mirror.com/owkin/phikon/resolve/main/config.json", -1),
}
ROOT = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\models"
S = requests.Session(); S.trust_env = False
for rel, (url, total) in DL.items():
    dst = os.path.join(ROOT, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    if total > 0 and have >= total * 0.999:
        print("[skip] %s already %.0f MB" % (rel, have / 1e6), flush=True); continue
    for attempt in range(8):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        hdr = dict(H)
        if have: hdr["Range"] = "bytes=%d-" % have
        try:
            r = S.get(url, headers=hdr, stream=True, timeout=(30, 120))
            if r.status_code == 416:
                print("[done] %s %.0f MB" % (rel, have / 1e6), flush=True); break
            r.raise_for_status()
            mode = "ab" if (have and r.status_code == 206) else "wb"
            if mode == "wb": have = 0
            t0 = time.time(); got = have
            with open(dst, mode) as f:
                for chunk in r.iter_content(1 << 22):
                    if not chunk: continue
                    f.write(chunk); got += len(chunk)
                    if time.time() - t0 > 20:
                        print("   %-28s %7.1f MB  %6.1f MB/s" % (rel, got / 1e6, (got - have) / 1e6 / (time.time() - t0)), flush=True)
                        t0 = time.time(); base = got
            have = got
            if total <= 0 or have >= total * 0.999:
                print("[ok] %s %.1f MB" % (rel, have / 1e6), flush=True); break
        except Exception as e:
            print("   retry %d on %s: %s" % (attempt + 1, rel, str(e)[:70]), flush=True)
            time.sleep(3)
    else:
        print("[FAIL] %s" % rel, flush=True)
print("DONE", flush=True)