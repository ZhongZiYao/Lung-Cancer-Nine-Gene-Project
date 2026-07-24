"""
step1_v3_biggest_sm_kmeans.py
──────────────────────────────
改进版 step1,融合 3 个改进:
  ① 每个 case 只选最大 SM(减少噪声,统一数据源)
  ② 在 SM 内 K-means(K=2)聚类,过滤非组织 patch
  ③ 上限 800 patch/case(原 500 → 800,补足 K-means 过滤后的损失)

为什么最大 SM?
  - 同一 case 多个 SM 是相邻物理切片,共享同一肿瘤的基因
  - 大 SM = 完整切片,组织覆盖全,tumor 比例通常最高
  - 跳过小 SM = 跳过边缘/碎片,减少噪声

为什么 K-means?
  - 1024 维 UNI 特征已经编码了"组织 vs 背景"
  - K=2 聚类:大 cluster = 组织,tiny cluster = 背景/空白
  - 过滤比 ~30-50%,保留大概率是肿瘤的 patch

输出:
  <out>/<case>.npz
    features:    [N_filtered, 1024]
    slice_ids:   [N_filtered] int  (但只有 0, 因为是 1 个 SM)
    slice_uids:  [1] str
    patch_paths: [N_filtered]

用法:
  python step1_v3_biggest_sm_kmeans.py --cases-file case_manifest.csv --n-max 800 --k 2

注意:
  - 如果某个 SM 的 patch 数 < 50,K-means 不可靠,跳过过滤
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ─────────────── 复用 step1 的关键部分 ───────────────
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_sm_dirs(case_dir: Path) -> list[Path]:
    """找 SM_<sopInstanceUID> 子目录"""
    return sorted([p for p in case_dir.iterdir()
                   if p.is_dir() and p.name.startswith("SM_")])


def select_biggest_sm(sm_dirs: list[Path]) -> Path:
    """从多个 SM 里选 patch 数最多的那个"""
    if len(sm_dirs) == 1:
        return sm_dirs[0]
    counts = [(sm, len([p for p in sm.iterdir() if p.suffix.lower() in IMG_EXTS]))
              for sm in sm_dirs]
    counts.sort(key=lambda x: -x[1])
    return counts[0][0]


def sample_patches_random(sm_dir: Path, n_total: int,
                          seed: int = 42) -> list[Path]:
    """从单个 SM 随机抽 n_total 个 patch"""
    rng = np.random.default_rng(seed)
    ps = sorted([p for p in sm_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if len(ps) <= n_total:
        return ps
    idx = rng.choice(len(ps), size=n_total, replace=False)
    idx.sort()
    return [ps[i] for i in idx]


def kmeans_filter(features: np.ndarray, k: int = 2,
                  min_keep_ratio: float = 0.3) -> np.ndarray:
    """
    K-means 二类聚类,只保留最大 cluster。
    返回保留的 mask (bool array)。
    """
    n = features.shape[0]
    if n < 50:
        # 太少了,K-means 不可靠,全保留
        return np.ones(n, dtype=bool)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    counts = np.bincount(labels)
    big_cluster = int(np.argmax(counts))
    keep = (labels == big_cluster)
    # 安全网:如果过滤掉太多(>70%),放弃过滤
    if keep.sum() / n < min_keep_ratio:
        return np.ones(n, dtype=bool)
    return keep


class PatchDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.tx = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tx(img), str(self.paths[i])


def get_transform():
    """UNI 的标准 transform"""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_uni(ckpt_path: str, device: str = "cpu"):
    """加载 UNI(ViT-L/16, 1024 dim)"""
    import timm
    model = timm.create_model("vit_large_patch16_224",
                              pretrained=False, num_classes=0)
    state = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(state, strict=False)
    print(f"[UNI] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_features(model, paths: list[Path], device: str,
                     batch_size: int = 16) -> np.ndarray:
    """抽特征,返回 [N, 1024]"""
    tx = get_transform()
    ds = PatchDataset(paths, tx)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2,
                    pin_memory=(device != "cpu"))
    feats = []
    for imgs, _ in dl:
        imgs = imgs.to(device, non_blocking=True)
        f = model(imgs)
        feats.append(f.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def extract_one_case_v3(model, case_dir: Path,
                         n_max: int, k_kmeans: int,
                         batch_size: int, device: str) -> dict:
    """每个 case: 选最大 SM → 抽 800 → K-means 过滤 → 返回"""
    sm_dirs = list_sm_dirs(case_dir)
    if not sm_dirs:
        return None
    # 1. 选最大 SM
    big_sm = select_biggest_sm(sm_dirs)
    # 2. 随机抽 n_max 个 patch
    chosen = sample_patches_random(big_sm, n_max, seed=42)
    if not chosen:
        return None
    n_before = len(chosen)
    # 3. 抽特征
    feats = extract_features(model, chosen, device, batch_size)
    # 4. K-means 过滤
    keep_mask = kmeans_filter(feats, k=k_kmeans)
    feats_f = feats[keep_mask]
    paths_f = [chosen[i] for i in range(len(chosen)) if keep_mask[i]]
    n_after = feats_f.shape[0]
    n_drop = n_before - n_after
    # 5. 单 SM → slice_ids 全是 0, slice_uids 1 个
    slice_ids = np.zeros(n_after, dtype=np.int32)
    slice_uids = np.array([big_sm.name], dtype=object)
    return {
        "features": feats_f.astype(np.float32),
        "slice_ids": slice_ids,
        "slice_uids": slice_uids,
        "patch_paths": np.array(paths_f, dtype=object),
        "_n_before": n_before,
        "_n_after": n_after,
        "_n_drop": n_drop,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-root", default="data/TCGA-LUAD-WSI/patches_direct")
    ap.add_argument("--out", default="pipeline/week_1_baseline/outputs/features_v3")
    ap.add_argument("--ckpt", default="UNI/assets/ckpts/uni/pytorch_model.bin")
    ap.add_argument("--cases-file",
                    default="pipeline/week_1_baseline/case_manifest.csv")
    ap.add_argument("--n-max", type=int, default=800,
                    help="每个 case 最多抽多少 patch(从最大 SM)")
    ap.add_argument("--k", type=int, default=2, help="K-means 的 k")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step1-v3] device={device}  n_max={args.n_max}  k={args.k}")
    model = load_uni(args.ckpt, device=device)

    cases = []
    with open(args.cases_file) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cases.append(r["case_id"])
    print(f"[Step1-v3] 共 {len(cases)} case")

    n_total_before = 0
    n_total_after = 0
    n_filtered_cases = 0
    t0 = time.time()
    for ci, case in enumerate(cases, 1):
        cd = Path(args.patch_root) / case
        if not cd.is_dir():
            print(f"  [{ci}/{len(cases)}] {case}  ❌ no dir")
            continue
        d = extract_one_case_v3(model, cd, args.n_max, args.k,
                                args.batch_size, device)
        if d is None:
            print(f"  [{ci}/{len(cases)}] {case}  ❌ no patches")
            continue
        np.savez(out / f"{case}.npz",
                 features=d["features"],
                 slice_ids=d["slice_ids"],
                 slice_uids=d["slice_uids"],
                 patch_paths=d["patch_paths"])
        n_total_before += d["_n_before"]
        n_total_after += d["_n_after"]
        if d["_n_drop"] > 0:
            n_filtered_cases += 1
        elapsed = time.time() - t0
        print(f"  [{ci}/{len(cases)}] {case}  SM=1  "
              f"before={d['_n_before']}  after={d['_n_after']}  "
              f"drop={d['_n_drop']}  ({elapsed:.0f}s elapsed)")

    print(f"\n[Step1-v3] 完成")
    print(f"  总 case: {len(cases)}")
    print(f"  抽 patch (过滤前): {n_total_before:,}")
    print(f"  抽 patch (过滤后): {n_total_after:,}")
    print(f"  被 K-means 过滤掉: {n_total_before - n_total_after:,} "
          f"({(1 - n_total_after/max(n_total_before,1))*100:.1f}%)")
    print(f"  启用 K-means 过滤的 case: {n_filtered_cases}/{len(cases)}")
    print(f"  输出: {out}/")


if __name__ == "__main__":
    main()