# -*- coding: utf-8 -*-
"""05b_build_cache_generic.py — 为指定 encoder 构建 2048 定长 patch 缓存
用法: python 05b_build_cache_generic.py --encoder gigap --dim 1536
"""
import os, glob, argparse, numpy as np, pandas as pd, torch
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CP = os.path.join(R,"data","public_features","CPathPatchFeature")
OUT = os.path.join(R,"data","patch_cache"); os.makedirs(OUT, exist_ok=True)
CASE = lambda p: "-".join(os.path.splitext(os.path.basename(p))[0].split("-")[:3])
ap = argparse.ArgumentParser()
ap.add_argument("--encoder", required=True)
ap.add_argument("--dim", type=int, required=True)
ap.add_argument("--n-patch", type=int, default=2048)
ap.add_argument("--min-patch", type=int, default=256)
A = ap.parse_args()

SRC = os.path.join(CP,"nsclc",A.encoder,"pt_files")
files = glob.glob(os.path.join(SRC,"*.pt"))
by_case = {}
for f in files: by_case.setdefault(CASE(f), []).append((os.path.getsize(f), f))
pick = {c: max(v)[1] for c, v in by_case.items()}
order = sorted(pick)
print(f"[1] encoder={A.encoder} dim={A.dim}  slide={len(files)} case={len(pick)}")

mm_path = os.path.join(OUT, f"NSCLC_{A.encoder}_sub{A.n_patch}.npy")
mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float16, shape=(len(order), A.n_patch, A.dim))
rows, kept, drop = [], 0, 0
for i, c in enumerate(order):
    try:
        t = torch.load(pick[c], map_location="cpu", weights_only=False)
        a = np.asarray(t if not hasattr(t,"numpy") else t.numpy(), dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != A.dim: drop += 1; continue
    except Exception: drop += 1; continue
    m = a.shape[0]
    if m < A.min_patch: drop += 1; continue
    idx = np.random.RandomState(abs(hash(c)) % (2**31)).choice(m, A.n_patch, replace=m < A.n_patch)
    mm[i] = a[idx].astype(np.float16)
    rows.append(dict(case_id=c, slide=os.path.basename(pick[c]), n_patch_orig=int(m), mem_row=i, n_patch_used=A.n_patch))
    kept += 1
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(order)} kept={kept}", flush=True)
mm.flush()
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, f"NSCLC_{A.encoder}_sub{A.n_patch}_index.csv"), index=False)
print(f"\n[done] kept={kept} dropped={drop}  -> {mm_path} ({os.path.getsize(mm_path)/1e9:.2f} GB)")
print(f"[idx ] NSCLC_{A.encoder}_sub{A.n_patch}_index.csv (含 mem_row, 已修正行号)")
print(f"       patch 数分布: min={df.n_patch_orig.min()} med={int(df.n_patch_orig.median())} max={df.n_patch_orig.max()}")
