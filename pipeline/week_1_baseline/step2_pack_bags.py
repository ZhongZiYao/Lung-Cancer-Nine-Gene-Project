"""
step2_pack_bags.py
──────────────────
把 step1 的 [N, 1024] patch 特征,按 SM 做 per-SM Gated Attention → SM vector(500 维),
再对 case 内所有 SM 做一次 case-level attention → 1 个 case 向量。

输入:
  <features_dir>/<case>.npz  (step1 输出)

输出:
  <out>/<case>.npz
    sm_features:  [K, 500]   float32  (每个 SM 一个向量)
    case_feature: [500]      float32  (case-level 聚合向量,供 step3 用)
    sm_uids:      [K]        str
    slice_ids:    [N]        int      (保留原 slice_ids)
    patch_features: [N, 1024] float32(保留原 features 供 step3 prototype 复用)

注:本步不训练任何参数,只是结构化打包。如果不想分两步,可以直接用 step1 的 .npz 训 step3。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch


def _gated_attn_core(x: torch.Tensor, hidden: int) -> torch.Tensor:
    """对 [n, d] tensor 做 Gated Attention,返回 [d] tensor"""
    n, d = x.shape
    proj = torch.randn(d, hidden, dtype=x.dtype) * (1.0 / np.sqrt(d))
    v_branch = torch.tanh(x @ proj)
    u_branch = torch.sigmoid(x @ proj)
    logits = (v_branch * u_branch).sum(dim=1, keepdim=True)
    weights = torch.softmax(logits, dim=0)
    return (weights * x).sum(dim=0)


def _random_project(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    d = x.shape[0]
    proj = torch.randn(d, out_dim, dtype=x.dtype) * (1.0 / np.sqrt(d))
    return x @ proj


def sm_gated_attention(patch_feats: np.ndarray, out_dim: int = 500,
                       hidden: int = 128) -> np.ndarray:
    """
    对单个 SM 内 patch 做 Gated Attention(无监督,固定随机权重)。
    输入: [n_patches, D] numpy, 输出: [out_dim] numpy
    """
    torch.manual_seed(42)
    x = torch.from_numpy(patch_feats).float()
    sm_feat = _gated_attn_core(x, hidden)
    out = _random_project(sm_feat, out_dim)
    return out.numpy().astype(np.float32)


def case_gated_attention(sm_feats: np.ndarray, out_dim: int = 500,
                         hidden: int = 128) -> np.ndarray:
    """对 case 内所有 SM vector 做 Gated Attention"""
    torch.manual_seed(123)
    x = torch.from_numpy(sm_feats).float()
    case_feat = _gated_attn_core(x, hidden)
    out = _random_project(case_feat, out_dim)
    return out.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir",
                    default="pipeline/week_1_baseline/outputs/features_per_case")
    ap.add_argument("--out",
                    default="pipeline/week_1_baseline/outputs/bags_per_case")
    ap.add_argument("--sm-dim", type=int, default=500)
    ap.add_argument("--case-dim", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npzs = sorted(feat_dir.glob("*.npz"))
    print(f"[Step2] 输入 {len(npzs)} 个 case")
    for npz in npzs:
        cid = npz.stem
        out_path = out_dir / f"{cid}.npz"
        if out_path.exists() and not args.force:
            print(f"  [SKIP] {cid}")
            continue
        d = np.load(npz, allow_pickle=True)
        feats = d["features"].astype(np.float32)         # [N, 1024]
        slice_ids = d["slice_ids"]                       # [N]
        slice_uids = d["slice_uids"]                     # [K]
        n_sm = len(slice_uids)
        print(f"  [RUN]  {cid}: N={len(feats)} patches, K={n_sm} SM")

        sm_feats = []
        for k in range(n_sm):
            mask = (slice_ids == k)
            if mask.sum() == 0:
                continue
            ps = feats[mask]
            sm_vec = sm_gated_attention(ps, out_dim=args.sm_dim)
            sm_feats.append(sm_vec)
        sm_feats = np.stack(sm_feats, axis=0)             # [K, 500]
        case_vec = case_gated_attention(sm_feats, out_dim=args.case_dim)

        np.savez_compressed(
            out_path,
            sm_features=sm_feats,
            case_feature=case_vec,
            sm_uids=slice_uids,
            slice_ids=slice_ids,
            patch_features=feats,
        )
    print("\n[Step2] 完成 ✓")


if __name__ == "__main__":
    main()