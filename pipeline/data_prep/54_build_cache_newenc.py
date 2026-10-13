# -*- coding: utf-8 -*-
"""54_build_cache_newenc.py — 为 CHIEF / UNI2-h 构建 512 定长 patch 缓存

比 51 版更简单也更稳:
  - .pt 实测是【裸 tensor】(不是 dict), 直接当 (N, D) 用
  - .h5 里有 features 键
  - 不做 compaction(上一版死在压缩步骤): 循环内直接写 mem_row, 跳过的行留空

输出(写入 data/patch_cache/):
  NSCLC_chief_sub512.npy + _index.csv      (N, 512, 768)  float16
  NSCLC_uni2h_sub512.npy + _index.csv     (N, 512, 1536) float16
"""
import os, sys, glob, re, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="both", choices=["chief", "uni2h", "both"])
ap.add_argument("--n-patch", type=int, default=512)
ap.add_argument("--min-patch", type=int, default=256)
A = ap.parse_args()

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
OUT = os.path.join(R, "data", "patch_cache")
os.makedirs(OUT, exist_ok=True)
FEAT = r"D:\features"
CID = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def load_pt(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("features", "embeddings", "x"):
            if k in d:
                d = d[k]
                break
    f = d.numpy() if isinstance(d, torch.Tensor) else np.asarray(d)
    f = np.asarray(f)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    return f.reshape(-1, f.shape[-1]).astype(np.float32)


def load_h5(p):
    import h5py
    with h5py.File(p, "r") as h:
        f = np.asarray(h["features"][:], dtype=np.float32)
    f = np.asarray(f)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    return f.reshape(-1, f.shape[-1]).astype(np.float32)


def build(enc, src, dim, loader):
    files = [p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith((".pt", ".h5"))]
    pick = {}
    for p in sorted(files):
        m = CID.search(os.path.basename(p))
        if m and m.group(1) not in pick:
            pick[m.group(1)] = p
    order = sorted(pick)
    print("[%s] %d 文件 -> %d case, 开始建 %s" % (enc, len(files), len(order), A.n_patch), flush=True)
    if not order:
        print("[%s] 无匹配 case" % enc, flush=True)
        return
    npy = os.path.join(OUT, "NSCLC_%s_sub%d.npy" % (enc, A.n_patch))
    csv = os.path.join(OUT, "NSCLC_%s_sub%d_index.csv" % (enc, A.n_patch))
    mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(len(order), A.n_patch, dim))
    rows, drop, nskip = [], 0, 0
    for i, c in enumerate(order):
        try:
            f = loader(pick[c])
            m = f.shape[0]
            if m < A.min_patch:
                drop += 1; nskip += 1; continue
        except Exception as e:
            print("  [skip] %s %s" % (c, str(e)[:60]), flush=True)
            drop += 1; nskip += 1; continue
        idx = np.random.RandomState(abs(hash(c)) % (2 ** 31)).choice(m, A.n_patch, replace=m < A.n_patch)
        mm[i] = f[idx].astype(np.float16)
        rows.append(dict(case_id=c, slide=os.path.basename(pick[c]), n_patch_orig=int(m),
                         mem_row=i, n_patch_used=A.n_patch))
        if (i + 1) % 150 == 0:
            print("  %d/%d 已写入 %d" % (i + 1, len(order), len(rows)), flush=True)
    mm.flush()
    del mm
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    print("[%s] 缓存完成 %s  写入 %d/%d  丢弃 %d" % (enc, os.path.basename(npy), len(df), len(order), drop), flush=True)
    if len(df):
        print("       patch 分布 min=%d med=%d max=%d" % (df.n_patch_orig.min(), int(df.n_patch_orig.median()), df.n_patch_orig.max()), flush=True)
        print("       前 3 行 case=%s" % list(df.case_id[:3]), flush=True)


if A.which in ("chief", "both"):
    build("chief", os.path.join(FEAT, "chief_nsclc", "nsclc", "chief", "pt_files"), 768, load_pt)
if A.which in ("uni2h", "both"):
    for proj in ["TCGA-LUAD", "TCGA-LUSC"]:
        sub = os.path.join(FEAT, "uni2h_nsclc", proj, "features")
        if not os.path.isdir(sub):
            print("[uni2h_%s] 目录不存在, 跳过" % proj, flush=True)
            continue
        build("uni2h_" + proj.replace("TCGA-", "").lower(), sub, 1536, load_h5)
print("DONE", flush=True)