"""用 DeepGEM TCGA 训好的 EGFR model 对 case-level 特征做 forward,得 prob。

我们**不依赖** DeepGEM main/test.py 的整套 dataset/engine 装载
(那些 DDP / config / samplers 全是训练用),而是复用两个东西:
  1. DeepGEMDataset (构造 padded case-level tensor + mask)
  2. DeepGEM model 权重 (modelTCGA_ExcisionalBiopsy_EGFR.pickle 的 state_dict)

输入:
    --internal-pickle : step2 写出的 internal.pickle
    --feat-dir        : combined_feat/ 目录(含 <case_id>.pkl)
    --checkpoint      : DeepGEM 训好的 modelTCGA_ExcisionalBiopsy_<GENE>.pickle
    --gene            : EGFR (checkpoint 必须匹配)
    --wsi-type        : ExcisionalBiopsy (DeepGEM 训练按这个分)
    --out             : 输出 CSV 路径 (case_id, true_label, prob_negative, prob_positive)

输出:
    CSV + stdout AUC summary (即使 case 很少也输出预测分布供 sanity check)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils


# ---- 复用 DeepGEMDataset (保持 DeepGEMDataset.build 的 layer 一致) ----
class DeepGEMDataset(data_utils.Dataset):
    def __init__(self, data_info, feature_len: int):
        self.wsi_loc = []
        for i in range(len(data_info["pid"])):
            self.wsi_loc.append(
                (data_info["pid"][i], data_info["feat_fp"][i], data_info["label"][i])
            )
        self.feature_len = feature_len

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        feat_len = self.feature_len
        image_id, patch_feat_fp, label = self.wsi_loc[index]
        label = torch.tensor(label).long()

        with open(patch_feat_fp, "rb") as f:
            patch_feat = pickle.load(f)

        ret_feat, ret_feat_name = [], []
        for each_ob in patch_feat:
            if "val" in each_ob:
                select_feat = each_ob["val"]
            elif "tr" in each_ob:
                # 单 case 没有 train,fallback
                c_feat = each_ob["tr"]
                idx = np.random.choice(c_feat.shape[0], 1).item()
                select_feat = c_feat[idx]
            else:
                continue
            select_feat = select_feat.reshape(-1)
            ret_feat.append(select_feat)
            ret_feat_name.append(each_ob.get("feat_name", image_id))

        ret_feat = np.stack(ret_feat)
        mask = np.ones(len(ret_feat))
        if len(ret_feat) > feat_len:
            ret_feat = ret_feat[:feat_len]
            mask = mask[:feat_len]
        cur_len = len(ret_feat)
        if cur_len < feat_len:
            residual = feat_len - cur_len
            append_data = np.zeros((residual, ret_feat.shape[1]))
            ret_feat = np.concatenate([ret_feat, append_data])
            mask = np.concatenate([mask, np.zeros(residual)])

        return (torch.from_numpy(ret_feat).float(),
                torch.from_numpy(mask).bool(),
                {"label": label.reshape(-1).repeat(feat_len),
                 "pid": image_id})


# ---- DeepGEM 模型 (直接复刻 main/models/model_deepgem.py,只取 inference 子集) ----
class DeepGEM(nn.Module):
    """DeepGEM = ViT-style aggregator over patch features (multi-instance)."""

    def __init__(self, feature_len=500, feature_dim=768, num_classes=2,
                 hidden_dim=64, depth=3, heads=4, mlp_dim=64, dropout=0.5):
        super().__init__()
        self.feature_len = feature_len
        self.feature_dim = feature_dim

        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_len + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # patch feature → hidden_dim
        self.patch_proj = nn.Linear(feature_dim, hidden_dim)

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)

        # classification head on CLS token
        self.head = nn.Linear(hidden_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, feat, mask):
        # feat: [B, N, 768], mask: [B, N] bool (True=valid)
        B, N, _ = feat.shape
        x = self.patch_proj(feat)                  # [B, N, hidden_dim]
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, hidden_dim]
        x = torch.cat([cls, x], dim=1)             # [B, N+1, hidden_dim]
        x = x + self.pos_embed[:, :N+1, :]

        # nn.TransformerEncoder needs key_padding_mask where True=ignore
        cls_mask = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device)
        full_mask = torch.cat([cls_mask, ~mask], dim=1)  # invert: True=padding
        x = self.transformer(x, src_key_padding_mask=full_mask)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-pickle", type=Path, default=Path("./deepgem_internal.pickle"))
    ap.add_argument("--feat-dir", type=Path, default=Path("./deepgem_combined_feat"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("./DeepGEM/checkpoints/DeepGEM_TCGA/"
                                 "modelTCGA_ExcisionalBiopsy_EGFR.pickle"))
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    ap.add_argument("--out", type=Path, default=Path("./egfr_pred.csv"))
    args = ap.parse_args()

    # ---- load internal.pickle ----
    with open(args.internal_pickle, "rb") as f:
        data = pickle.load(f)
    info = data[args.wsi_type][args.gene]["test"]
    info["feat_fp"] = [str(args.feat_dir / f"{pid}.pkl") for pid in info["pid"]]
    print(f"[INFO] {len(info['pid'])} cases  gene={args.gene}  "
          f"positive={sum(info['label'])}/{len(info['label'])}")

    # ---- load checkpoint 拿 feature_len ----
    if not args.checkpoint.exists():
        sys.exit(
            f"[FATAL] {args.checkpoint} 不存在\n"
            "DeepGEM 训好的 EGFR model 已经下到了 ./DeepGEM/checkpoints/DeepGEM_TCGA/\n"
            "该路径不在 .gitignore (modelTCGA_ExcisionalBiopsy_EGFR.pickle 6.7 MB)"
        )
    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    parameter = ckpt["parameter"]
    state_dict = ckpt["checkpoint"]
    feature_len = parameter.get("feature_len", 500)
    feature_dim = parameter.get("feature_dim", 768)
    num_classes = parameter.get("num_classes", 2)
    hidden_dim = parameter.get("hidden_dim", 64)
    depth = parameter.get("depth", 3)
    heads = parameter.get("heads", 4)
    mlp_dim = parameter.get("mlp_dim", 64)
    dropout = parameter.get("dropout", 0.5)
    print(f"[INFO] ckpt param: feature_len={feature_len}, feature_dim={feature_dim}, "
          f"num_classes={num_classes}, hidden_dim={hidden_dim}, depth={depth}")

    # ---- build model, load weights ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepGEM(
        feature_len=feature_len, feature_dim=feature_dim,
        num_classes=num_classes, hidden_dim=hidden_dim,
        depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
    ).to(device).eval()

    # DeepGEM 保存的 state_dict 有 'module.' 前缀(DDP 残留)和 'total_params/total_ops'(thop)
    new_state = {}
    for k, v in state_dict.items():
        if k.endswith("total_params") or k.endswith("total_ops"):
            continue
        new_state[k.replace("module.", "")] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (前 5: {missing[:5]})")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (前 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print("[INFO] state_dict load: missing=0, unexpected=0 ✓")

    ds = DeepGEMDataset(info, feature_len=feature_len)

    rows = []
    for idx in range(len(ds)):
        feat, mask, target = ds[idx]
        pid = target["pid"]
        label = int(target["label"][0].item())
        feat = feat.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat, mask)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        rows.append({"case_id": pid, "true_label": label,
                     "prob_negative": float(probs[0]),
                     "prob_positive": float(probs[1])})
        print(f"  {pid:18s}  label={label}  "
              f"P(neg)={probs[0]:.3f}  P(pos)={probs[1]:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[OK] predictions → {args.out}")

    # ---- summary ----
    pos_cases = df[df.true_label == 1]
    neg_cases = df[df.true_label == 0]
    print()
    print(f"[Summary] EGFR+ cases: {len(pos_cases)}, mean P(pos): "
          f"{pos_cases.prob_positive.mean() if len(pos_cases) else float('nan'):.3f}")
    print(f"[Summary] EGFR- cases: {len(neg_cases)}, mean P(pos): "
          f"{neg_cases.prob_positive.mean() if len(neg_cases) else float('nan'):.3f}")
    if len(pos_cases) and len(neg_cases):
        from sklearn.metrics import roc_auc_score, accuracy_score
        try:
            auc = roc_auc_score(df.true_label, df.prob_positive)
            print(f"[Summary] AUC = {auc:.3f}  (注意: N very small, 仅供参考)")
        except Exception as e:
            print(f"[Summary] AUC 无法计算: {e}")


if __name__ == "__main__":
    main()
