"""
step2_to_deepgem_pkl.py
──────────────────────
把 step1 输出的 .npz 转为 DeepGEM test.py 期望的 .pkl 格式。

DeepGEM 的 .pkl 是 list of dict,每个 patch 是:
    {"val": np.array([D], float32), "feat_name": "patch_X_Y"}

注:如果设为 "tr" 则训练时随机选;这里只做 test,只用 "val"。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def convert_one(npz_path: Path, out_pkl_path: Path, pid: str, target_dim: int = 768):
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"]
    paths = d["patch_paths"]

    # ⚠️ DeepGEM 默认 patch_dim=768(CTransPath 输出)
    # UNI 输出 1024,我们截断到 768(linear probe 实验里 1024→256 投影首层会学
    # 适配,但这是 sanity check,最简单就是截断)
    if feats.shape[1] > target_dim:
        feats = feats[:, :target_dim]

    # ⚠️ DeepGEM 期望默认 feature_len=500,太多会被截断,太少会被 0-pad
    # 我们全部 patch 都塞进去,但准备 feature_len 个列表槽
    # DeepGEM 会做: if len > feat_len: feat[:feat_len]
    # 所以数据多也没事,会截断

    patch_list = []
    for i, (vec, p) in enumerate(zip(feats, paths)):
        patch_list.append({
            "val": vec.astype(np.float32),
            "feat_name": f"{pid}_patch_{i}_{Path(p).stem}",
        })
    out_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(patch_list, f)
    return len(patch_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="pipeline/features")
    ap.add_argument("--out-dir", default="pipeline/deepgem_feat")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    feats_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)

    if args.cases:
        cases = args.cases
    else:
        cases = sorted(p.stem for p in feats_dir.glob("*.npz"))

    for c in cases:
        npz = feats_dir / f"{c}.npz"
        out_pkl = out_dir / f"{c}.pkl"
        if not npz.exists():
            print(f"[skip] {npz} not exist")
            continue
        n = convert_one(npz, out_pkl, c)
        print(f"  {c}: {n} patches -> {out_pkl}")


if __name__ == "__main__":
    main()
