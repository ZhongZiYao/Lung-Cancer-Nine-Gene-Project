"""
step1_extract_features.py
─────────────────────────
对每个 case 抽取 patch 特征(冻结 UNI,ViT-L/16,1024 维)。

关键设计(融合 GAMIL + DeepGEM):
  - 每个 case 最多保留 N=500 patch(DeepGEM 一致)
  - 按 SM 均衡采样(避免大 SM 主导,GAMIL 多 SM 场景需要)
  - 保留 SM 归属信息(slice_ids),供 step2 做 per-SM 聚合
  - 不切 patch、只读 patch(零修改你的数据)

输出:
  <out>/<case>.npz
    features:    [N, 1024] float32  (N ≤ 500)
    slice_ids:   [N]        int      (每个 patch 来自哪个 SM, 0..K-1)
    slice_uids:  [K]        str      (SM_<sopInstanceUID>)
    patch_paths: [N]        str      (相对 case_dir 的路径, debug 用)

用法:
  python step1_extract_features.py --cases TCGA-05-4244
  python step1_extract_features.py --cases-file case_manifest.csv
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ─────────────── 1. 数据结构 ───────────────
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_sm_dirs(case_dir: Path) -> list[Path]:
    """找 case_dir 下所有 SM_<sopInstanceUID> 子目录,按名字排序保证可复现"""
    return sorted([p for p in case_dir.iterdir()
                   if p.is_dir() and p.name.startswith("SM_")])


def sample_patches_balanced(sm_dirs: list[Path], n_total: int,
                            seed: int = 42) -> tuple[list[Path], list[int], list[str]]:
    """
    在 SM 之间均衡采样,总 patch 数 ≤ n_total。
    返回 (patch_paths, slice_ids, slice_uids)
    """
    rng = np.random.default_rng(seed)
    # 先收集每个 SM 的所有 patch
    sm_to_patches: dict[Path, list[Path]] = {}
    for sm in sm_dirs:
        ps = sorted([p for p in sm.iterdir()
                     if p.is_file() and p.suffix.lower() in IMG_EXTS])
        sm_to_patches[sm] = ps

    # 算每个 SM 分多少 patch:平均分,余数给大 SM
    n_sm = len(sm_dirs)
    if n_sm == 0:
        return [], [], []

    per_sm = [n_total // n_sm] * n_sm
    leftover = n_total - per_sm[0] * n_sm
    # 把余数给 patch 数最多的 SM(优先保留信号)
    sizes = [(len(sm_to_patches[sm]), i) for i, sm in enumerate(sm_dirs)]
    sizes.sort(reverse=True)
    for _, i in sizes[:leftover]:
        per_sm[i] += 1

    patch_paths: list[Path] = []
    slice_ids: list[int] = []
    slice_uids: list[str] = []
    for sm_idx, sm in enumerate(sm_dirs):
        ps = sm_to_patches[sm]
        k = min(per_sm[sm_idx], len(ps))
        if k == 0:
            continue
        if k < len(ps):
            idx = rng.choice(len(ps), size=k, replace=False)
            idx.sort()
            chosen = [ps[i] for i in idx]
        else:
            chosen = ps
        patch_paths.extend(chosen)
        slice_ids.extend([sm_idx] * len(chosen))
        slice_uids.append(sm.name)

    return patch_paths, slice_ids, slice_uids


class PatchDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.tx = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:
            print(f"  [WARN] {self.paths[i]}: {e}, 用黑图")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.tx(img), str(self.paths[i])


# ─────────────── 2. UNI backbone ───────────────
def load_uni(ckpt_path: str | None, device: str):
    import timm
    print(f"  [UNI] 创建 vit_large_patch16_224 ...")
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=False, num_classes=0
    )
    if ckpt_path and Path(ckpt_path).exists():
        print(f"  [UNI] 加载权重 {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
    else:
        print(f"  [UNI] ⚠️ 随机初始化")
    return model.to(device).eval(), 1024


ENCODERS = {"uni": load_uni}


# ─────────────── 3. 主流程 ───────────────
def get_cases(args) -> list[str]:
    if args.cases:
        return args.cases
    if not args.cases_file:
        sys.exit("必须提供 --cases 或 --cases-file")
    out = []
    with open(args.cases_file) as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append(r["case_id"])
    return out


def extract_one_case(case_id: str, patch_root: Path, out_path: Path,
                     model, feat_dim: int, transform, device: str,
                     batch_size: int, num_workers: int,
                     n_patches: int, force: bool, seed: int):
    if out_path.exists() and not force:
        try:
            d = np.load(out_path, allow_pickle=True)
            n_existing = d["features"].shape[0]
            # 只要有任一不一致就重抽:
            #   - 维度不对(模型变了)
            #   - 数量 < n_patches 且总 patch 够(残缺版本)
            total_patches = sum(
                len(list((patch_root / case_id / sm).glob("*.jpg")))
                + len(list((patch_root / case_id / sm).glob("*.jpeg")))
                + len(list((patch_root / case_id / sm).glob("*.png")))
                for sm in [d.name for d in (patch_root / case_id).iterdir()
                           if d.is_dir() and d.name.startswith("SM_")]
            ) if (patch_root / case_id).exists() else 0

            if (d["features"].shape[1] == feat_dim
                    and (n_existing >= n_patches or total_patches < n_patches)):
                print(f"  [SKIP] {case_id}: 已有 {d['features'].shape}")
                return
            else:
                print(f"  [REDO] {case_id}: 已有 {n_existing} patches, "
                      f"重抽到 {n_patches}")
        except Exception as e:
            print(f"  [REDO] {case_id}: 旧文件损坏 ({e}),重抽")

    case_dir = patch_root / case_id
    if not case_dir.exists():
        print(f"  [SKIP] {case_id}: 不存在")
        return

    sm_dirs = list_sm_dirs(case_dir)
    if not sm_dirs:
        print(f"  [SKIP] {case_id}: 没有 SM_* 目录")
        return

    patch_paths, slice_ids, slice_uids = sample_patches_balanced(
        sm_dirs, n_total=n_patches, seed=seed
    )
    if not patch_paths:
        print(f"  [SKIP] {case_id}: 没有 patch")
        return

    print(f"  [RUN]  {case_id}: K={len(sm_dirs)} SM, "
          f"N={len(patch_paths)}/{n_patches} patches")
    t0 = time.time()
    ds = PatchDataset(patch_paths, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    feats = np.zeros((len(patch_paths), feat_dim), dtype=np.float32)
    rels = []
    cursor = 0
    with torch.no_grad():
        for batch_imgs, batch_paths in dl:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                v = model(batch_imgs)
            v = v.float().cpu().numpy()
            n = v.shape[0]
            feats[cursor:cursor + n] = v
            rels.extend(batch_paths)
            cursor += n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=feats[:cursor],
        slice_ids=np.array(slice_ids[:cursor], dtype=np.int32),
        slice_uids=np.array(slice_uids, dtype=object),
        patch_paths=np.array(rels[:cursor], dtype=object),
    )
    print(f"        耗时 {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-root", default="data/TCGA-LUAD-WSI/patches_direct")
    ap.add_argument("--out", default="pipeline/week_1_baseline/outputs/features_per_case")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--cases-file", default=None)
    ap.add_argument("--encoder", default="uni", choices=list(ENCODERS.keys()))
    ap.add_argument("--ckpt", default="UNI/assets/ckpts/uni/pytorch_model.bin")
    ap.add_argument("--n-patches", type=int, default=500,
                    help="每 case 截断到多少 patch(DeepGEM 一致)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step1] device={device}, encoder={args.encoder}, n_patches={args.n_patches}")
    model, feat_dim = ENCODERS[args.encoder](args.ckpt, device)

    tx = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cases = get_cases(args)
    print(f"[Step1] 共 {len(cases)} case")
    patch_root = Path(args.patch_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cid in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {cid}")
        extract_one_case(
            cid, patch_root, out_dir / f"{cid}.npz",
            model, feat_dim, tx, device,
            args.batch_size, args.num_workers,
            args.n_patches, args.force, args.seed,
        )

    print("\n[Step1] 完成 ✓")


if __name__ == "__main__":
    main()