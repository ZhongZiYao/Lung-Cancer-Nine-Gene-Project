"""
step4_run_deepgem_sanity.py
───────────────────────────
不调 DeepGEM 原 test.py(它有 DDP/config/distributed 噪音,本地 debug 麻烦),
自己写等价 forward:
  1) 加载 internal.pickle 拿到 3 个 case 的 feat_fp + label
  2) 加载每个 .pkl 的 patch 列表
  3) 截断/pad 到 feature_len (从 dummy ckpt['parameter'] 拿)
  4) 用 dummy 初始化的 DeepGEM 跑 forward
  5) softmax 后看 positive 概率 vs 真 label 是否"看起来有点道理"

⚠️ DeepGEM 是随机初始化权重,**输出没意义**,这只验证:
  - 你的特征 .pkl 格式对吗
  - 你的 internal.pickle 格式对吗
  - DeepGEM 接受 feature_dim=你给的
  - 不会崩
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def load_internal_pickle(path: Path, wsi_type: str, gene: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[wsi_type][gene]["test"]


def load_one_case(pkl_path: Path, feature_len: int):
    with open(pkl_path, "rb") as f:
        patch_list = pickle.load(f)

    feats = np.stack([p["val"] for p in patch_list]).astype(np.float32)  # [N, D]
    if feats.shape[0] >= feature_len:
        feats = feats[:feature_len]
        mask = np.ones(feature_len, dtype=bool)
    else:
        pad = feature_len - feats.shape[0]
        feats = np.concatenate([feats, np.zeros((pad, feats.shape[1]), dtype=np.float32)])
        mask = np.concatenate([np.ones(feats.shape[0] - pad, dtype=bool),
                               np.zeros(pad, dtype=bool)])
    return torch.from_numpy(feats), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", default="DeepGEM/main/data/internal/internal_3case.pickle")
    ap.add_argument("--ckpt", default="DeepGEM/main/checkpoints/dummy_3case.pickle")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    ap.add_argument("--gene", default="EGFR")
    args = ap.parse_args()

    sys.path.insert(0, str(Path("DeepGEM/main").resolve()))
    from util.misc import NestedTensor

    # 1) 读 internal.pickle 的 test split
    info = load_internal_pickle(Path(args.internal), args.wsi_type, args.gene)
    pids = info["pid"]
    labels = info["label"]
    feat_fps = info["feat_fp"]
    print(f"[load] {len(pids)} cases, gene={args.gene}")
    for p, l in zip(pids, labels):
        print(f"  {p}: label={l}")

    # 2) 读 dummy checkpoint 拿 feature_len / 架构
    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    feature_len = ck["parameter"]["feature_len"]
    hidden_dim = ck["parameter"]["hidden_dim"]
    depth = ck["parameter"]["depth"]
    patch_dim = ck["checkpoint"]["patch_encoder.patch_to_embedding.weight"].shape[1]
    print(f"\n[ckpt] feature_len={feature_len}, hidden_dim={hidden_dim}, "
          f"depth={depth}, patch_dim={patch_dim}")

    # 3) 加载模型
    from models.model_deepgem import DeepGEM
    model = DeepGEM(
        num_classes=2, patch_dim=patch_dim, dim=hidden_dim, depth=depth, num_queries=5,
    )
    # strict=False:允许 missing keys(随机初始化的)
    missing, unexpected = model.load_state_dict(ck["checkpoint"], strict=False)
    print(f"[model] loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  first 3 missing: {missing[:3]}")
    if unexpected:
        print(f"  first 3 unexpected: {unexpected[:3]}")
    model.eval()

    # 4) 对每个 case 跑 forward
    print(f"\n[forward] {args.gene} predictions")
    print(f"  {'pid':<14} {'label':<6} {'neg_prob':<10} {'pos_prob':<10} {'pred':<6}")
    print(f"  {'-'*48}")
    preds = []
    for pid, label, fp in zip(pids, labels, feat_fps):
        feats, mask = load_one_case(Path(fp), feature_len)
        # NestedTensor 期望 (tensor, mask)
        x = feats.unsqueeze(0)        # [1, feature_len, D]
        m = mask.unsqueeze(0)         # [1, feature_len]
        sample = NestedTensor(x, m)
        with torch.no_grad():
            logits = model(sample)    # [1, 2]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # [2]
        pred = int(np.argmax(probs))
        print(f"  {pid:<14} {label:<6} {probs[0]:<10.4f} {probs[1]:<10.4f} {pred:<6}")
        preds.append({
            "pid": pid,
            "label": int(label),
            "neg_prob": float(probs[0]),
            "pos_prob": float(probs[1]),
            "pred": pred,
        })

    # 5) sanity check 报告
    print(f"\n[sanity check]")
    correct = sum(int(p["pred"] == p["label"]) for p in preds)
    print(f"  random-init accuracy: {correct}/{len(preds)} "
          f"(expect ~50% if weights are truly random)")
    pos_probs = [p["pos_prob"] for p in preds]
    print(f"  pos_prob distribution: min={min(pos_probs):.3f}, "
          f"max={max(pos_probs):.3f}, mean={np.mean(pos_probs):.3f}")
    print(f"  ✅ sanity check passed = forward+shape+label mapping all work")


if __name__ == "__main__":
    main()
