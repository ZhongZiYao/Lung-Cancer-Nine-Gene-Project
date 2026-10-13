# -*- coding: utf-8 -*-
"""
05_build_patch_cache.py
把 CPathPatchFeature 的 NSCLC patch 特征 (.pt, 中位 1 万 patch/slide, 52GB)
压成定长子采样缓存,供 nnMIL 风格训练直接吃。

  - 每 case 取 patch 数最多的那一张 slide
  - 随机定长子采样到 MAX_PATCH (默认 2048), 固定随机种子
  - 存 float16 单个 memmap + index csv
输出: data/patch_cache/{NSCLC_sub2048.npy, NSCLC_sub2048_index.csv}
"""
import os, re, glob, argparse, numpy as np, pandas as pd, torch

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
SRC = os.path.join(R,"data","public_features","CPathPatchFeature","nsclc","uni","pt_files")
OUT = os.path.join(R,"data","patch_cache")
os.makedirs(OUT, exist_ok=True)
CASE = lambda p: "-".join(os.path.splitext(os.path.basename(p))[0].split("-")[:3])

ap = argparse.ArgumentParser()
ap.add_argument("--max-patch", type=int, default=2048)
ap.add_argument("--min-patch", type=int, default=256, help="低于此 patch 数的 slide 丢弃")
A = ap.parse_args()
N = A.max_patch

files = glob.glob(os.path.join(SRC,"*.pt"))
by_case = {}
for f in files:
    by_case.setdefault(CASE(f), []).append((os.path.getsize(f), f))
pick = {c: max(v)[1] for c, v in by_case.items()}     # 取最大的那张 slide
print(f"[1] slide={len(files)}  case={len(pick)}  max_patch={N}")

order = sorted(pick)
mm_path = os.path.join(OUT, f"NSCLC_sub{N}.npy")
mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float16, shape=(len(order), N, 1024))

rows, kept, dropped = [], 0, 0
for i, c in enumerate(order):
    p = pick[c]
    try:
        t = torch.load(p, map_location="cpu", weights_only=False)
        a = np.asarray(t if not hasattr(t, "numpy") else t.numpy(), dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 1024: dropped += 1; continue
    except Exception as e:
        print(f"    ! {c}: {type(e).__name__}"); dropped += 1; continue
    m = a.shape[0]
    if m < A.min_patch: dropped += 1; continue
    idx = np.random.RandomState(abs(hash(c)) % (2**31)).choice(m, N, replace=m < N)
    mm[i] = a[idx].astype(np.float16)
    rows.append(dict(case_id=c, slide=os.path.basename(p), n_patch_orig=int(m), n_patch_used=N))
    kept += 1
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(order)}  kept={kept}", flush=True)

mm.flush()
idx_df = pd.DataFrame(rows)
idx_df.to_csv(os.path.join(OUT, f"NSCLC_sub{N}_index.csv"), index=False)
print(f"\n[done] kept={kept}  dropped={dropped}")
print(f"[wrote] {mm_path}  ({os.path.getsize(mm_path)/1e9:.2f} GB)")
print(f"[wrote] {OUT}\\NSCLC_sub{N}_index.csv")
