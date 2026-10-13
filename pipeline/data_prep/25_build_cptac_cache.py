# -*- coding: utf-8 -*-
"""25_build_cptac_cache.py — CPTAC-NSCLC patch 特征定长缓存(用于合并训练)"""
import os, glob, argparse, numpy as np, pandas as pd, torch
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
OUT = os.path.join(R,"data","patch_cache"); os.makedirs(OUT, exist_ok=True)
CASE = lambda p: "-".join(os.path.splitext(os.path.basename(p))[0].split("-")[:2])
ap = argparse.ArgumentParser()
ap.add_argument("--n-patch", type=int, default=512)
A = ap.parse_args()
SRC = os.path.join(R,"data","public_features","CPathPatchFeature","cptac_nsclc","uni","pt_files")
by = {}
for f in glob.glob(os.path.join(SRC,"*.pt")):
    by.setdefault(CASE(f), []).append((os.path.getsize(f), f))
pick = {c: max(v)[1] for c, v in by.items()}
order = sorted(pick)
print(f"[1] slide={sum(len(v) for v in by.values())} case={len(pick)}")
mm = np.lib.format.open_memmap(os.path.join(OUT,f"CPTAC_uni_sub{A.n_patch}.npy"), mode="w+",
                               dtype=np.float16, shape=(len(order), A.n_patch, 1024))
rows=[]; drop=0
for i,c in enumerate(order):
    try:
        t = torch.load(pick[c], map_location="cpu", weights_only=False)
        a = np.asarray(t if not hasattr(t,"numpy") else t.numpy(), dtype=np.float32)
        if a.ndim!=2 or a.shape[1]!=1024 or a.shape[0]<256: drop+=1; continue
    except Exception: drop+=1; continue
    idx = np.random.RandomState(abs(hash(c))%(2**31)).choice(a.shape[0], A.n_patch, replace=a.shape[0]<A.n_patch)
    mm[i] = a[idx].astype(np.float16)
    rows.append(dict(case_id=c, n_patch_orig=int(a.shape[0]), mem_row=i))
mm.flush()
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT,f"CPTAC_uni_sub{A.n_patch}_index.csv"), index=False)
print(f"[done] kept={len(rows)} dropped={drop}  -> CPTAC_uni_sub{A.n_patch}.npy "
      f"({os.path.getsize(os.path.join(OUT,f'CPTAC_uni_sub{A.n_patch}.npy'))/1e9:.2f} GB)")
print(f"       patch 分布: min={df.n_patch_orig.min()} med={int(df.n_patch_orig.median())} max={df.n_patch_orig.max()}")
