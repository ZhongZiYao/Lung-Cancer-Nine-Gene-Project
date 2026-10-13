# -*- coding: utf-8 -*-
"""51_build_cache_newenc.py — 为 CHIEF / UNI2-h 构建 512 定长 patch 缓存

背景: 05b_build_cache_generic.py 只会读 CPathPatchFeature 仓库下已就位的
     uni / gigap 两个子目录。新编码器的特征下在 D:\features 下, 格式也不同
     (CHIEF 是 .pt, UNI2-h 是 .h5), 故单独写一个。

输出:
  NSCLC_chief_sub512.npy  + _index.csv      (N, 512, 768)  float16
  NSCLC_uni2h_sub512.npy  + _index.csv      (N, 512, 1536) float16

case_id 归一化: 文件名形如
  CHIEF : TCGA-05-4380-01Z-00-DX1.<uuid>.pt   -> TCGA-05-4380
  UNI2-h: TCGA-05-4380-01Z-00-DX1.<uuid>.h5   -> TCGA-05-4380
          CPTAC 用 C3L-00001-21 这类, 本脚本只处理 TCGA
"""
import os, sys, glob, argparse, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--which", default="chief", choices=["chief", "uni2h", "both"])
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
    f = d["features"]
    if isinstance(f, torch.Tensor):
        f = f.numpy()
    f = np.asarray(f)
    return f.reshape(-1, f.shape[-1]).astype(np.float32)


def load_h5(p):
    import h5py
    with h5py.File(p, "r") as h:
        f = h["features"][:]
    f = np.asarray(f)
    return f.reshape(-1, f.shape[-1]).astype(np.float32)


def build(enc, src, dim, loader):
    files = glob.glob(os.path.join(src, "*"))
    files = [p for p in files if p.lower().endswith((".pt", ".h5"))]
    pick = {}
    for p in sorted(files):
        m = CID.search(os.path.basename(p))
        if not m:
            continue
        c = m.group(1)
        if c not in pick:                      # 每 case 只取第一张切片(与 05b 口径一致)
            pick[c] = p
    print("[%s] 源目录 %d 文件 -> %d case" % (enc, len(files), len(pick)), flush=True)
    if not pick:
        print("[%s] 没有匹配到 TCGA case, 跳过" % enc, flush=True)
        return
    order = sorted(pick)
    mm = np.lib.format.open_memmap(os.path.join(OUT, "NSCLC_%s_sub%d.npy" % (enc, A.n_patch)),
                                    mode="w+", dtype=np.float16,
                                    shape=(len(order), A.n_patch, dim))
    rows, drop, ok = [], 0, 0
    for i, c in enumerate(order):
        try:
            f = loader(pick[c])
        except Exception as e:
            print("  [skip] %s %s" % (c, str(e)[:80]), flush=True); drop += 1; continue
        m = f.shape[0]
        if m < A.min_patch:
            drop += 1; continue
        idx = np.random.RandomState(abs(hash(c)) % (2 ** 31)).choice(m, A.n_patch, replace=m < A.n_patch)
        mm[i] = f[idx].astype(np.float16)
        rows.append(dict(case_id=c, slide=os.path.basename(pick[c]),
                         n_patch_orig=int(m), mem_row=len(rows) - 1, n_patch_used=A.n_patch))
        ok += 1
        if (i + 1) % 100 == 0:
            print("  %d/%d  已写入 %d" % (i + 1, len(order), ok), flush=True)
    mm.flush()
    del mm
    # 丢弃未写入的行, 紧凑化
    df = pd.DataFrame(rows)
    full = np.load(os.path.join(OUT, "NSCLC_%s_sub%d.npy" % (enc, A.n_patch)), mmap_mode="r")
    keep = df.mem_row.values
    arr = np.ascontiguousarray(full[keep])
    np.save(os.path.join(OUT, "NSCLC_%s_sub%d.npy" % (enc, A.n_patch)), arr)
    df = df.reset_index(drop=True)
    df["mem_row"] = np.arange(len(df))
    df.to_csv(os.path.join(OUT, "NSCLC_%s_sub%d_index.csv" % (enc, A.n_patch)), index=False)
    print("[%s] 缓存 %s  shape=%s  drop=%d" % (enc, "NSCLC_%s_sub%d.npy" % (enc, A.n_patch),
                                             arr.shape, drop), flush=True)
    print("       patch 数分布 min=%d med=%d max=%d" % (df.n_patch_orig.min(),
                                                      int(df.n_patch_orig.median()),
                                                      df.n_patch_orig.max()), flush=True)


if A.which in ("chief", "both"):
    build("chief", os.path.join(FEAT, "chief_nsclc", "nsclc", "chief", "pt_files"), 768, load_pt)
if A.which in ("uni2h", "both"):
    for proj in ["TCGA-LUAD", "TCGA-LUSC"]:
        build("uni2h_" + proj.replace("TCGA-", "").lower(),
              os.path.join(FEAT, "uni2h_nsclc", proj, "features"), 1536, load_h5)
print("DONE", flush=True)