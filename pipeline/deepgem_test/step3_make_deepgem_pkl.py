"""
deepgem_test/step3_make_deepgem_pkl.py  (per-patch .pkl 模式,跟 DeepGEM sample 一致)

每个 patch 一个 .pkl 文件,文件名包含 slice_uid + (x,y):
  deepgem_feat/<case>/<slice_uid>__x_y.pkl
  内容: {"val": np.array([768]), "feat_name": "..."}

可指定 --per-patch 来切换。原 --mode combined 是把 N 个 patch 装进 1 个 .pkl(用于 easy testing)。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent.resolve()


def convert_per_patch(npz_path: Path, out_root: Path, pid: str):
    """每个 patch 一个 .pkl"""
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"]                       # [N, 768]
    slice_ids = d["slice_ids"]                  # [N] str
    slice_uids_unique = d["slice_uids"]         # [n_slices]

    n = 0
    for s_idx, slice_uid in enumerate(slice_uids_unique):
        mask = slice_ids == slice_uid
        slice_feats = feats[mask]
        # 用 slice_idx 简化目录: feat_root/<pid>/s<i>/
        slice_dir = out_root / pid / f"s{s_idx}_{str(slice_uid)[:8]}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        for i, vec in enumerate(slice_feats):
            out_pkl = slice_dir / f"p{i}_f768.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump({
                    "val": vec.astype(np.float32),
                    "feat_name": f"{pid}_s{s_idx}_p{i}",
                }, f)
            n += 1

    return n


def convert_combined(npz_path: Path, out_pkl_path: Path, pid: str):
    """1 .pkl 装所有 patch list (旧 step3 模式)"""
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"]
    patch_list = []
    for i, vec in enumerate(feats):
        patch_list.append({
            "val": vec.astype(np.float32),
            "feat_name": f"{pid}_p{i}",
        })
    out_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(patch_list, f)
    return len(patch_list)


def convert_slice_combined(npz_path: Path, out_dir: Path, pid: str):
    """每个 slice 一个 .pkl,1 slice = 1 .pkl (list of dict)"""
    d = np.load(npz_path, allow_pickle=True)
    feats = d["features"]
    slice_ids = d["slice_ids"]
    slice_uids = d["slice_uids"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for s_idx, slice_uid in enumerate(slice_uids):
        mask = slice_ids == slice_uid
        slice_feats = feats[mask]
        slice_pkl = out_dir / f"{pid}_s{s_idx}.pkl"
        patch_list = [
            {"val": vec.astype(np.float32),
             "feat_name": f"{pid}_s{s_idx}_p{i}"}
            for i, vec in enumerate(slice_feats)
        ]
        with open(slice_pkl, "wb") as f:
            pickle.dump(patch_list, f)
        results.append((str(slice_uid), len(slice_feats), str(slice_pkl)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=str(HERE / "features"))
    ap.add_argument("--out-dir", default=str(HERE / "deepgem_feat"))
    ap.add_argument("--cases", nargs="*",
                    default=["TCGA-05-4382", "TCGA-05-4249", "TCGA-05-4395"])
    ap.add_argument("--mode", default="per_patch",
                    choices=["per_patch", "combined", "slice_combined"],
                    help="per_patch = DeepGEM 原版 (1 patch 1 .pkl)")
    args = ap.parse_args()

    feats_dir = Path(args.features_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for c in args.cases:
        npz = feats_dir / f"{c}.npz"
        if not npz.exists():
            print(f"[skip] {npz} missing"); continue
        if args.mode == "per_patch":
            n = convert_per_patch(npz, out_root, c)
            print(f"  {c}: {n} per-patch .pkl files under {out_root / c}/")
        elif args.mode == "combined":
            n = convert_combined(npz, out_dir / f"{c}.pkl", c)
            print(f"  {c}: combined {n} patches -> {out_dir}/{c}.pkl")
        else:  # slice_combined
            results = convert_slice_combined(npz, out_dir, c)
            for uid, n, path in results:
                print(f"  {c} / {uid[:30]}...: {n} patches -> {path}")


if __name__ == "__main__":
    main()
