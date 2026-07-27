"""
egfr_only_model.py
─────────────────
专门针对 EGFR 单基因训练的简化模型。

设计目标:用极少参数适应 83 case + EGFR 13 阳性的极小样本场景。

架构(简化版):
  patch_feats [N, 1024]
       ↓
  PatchBackbone(1024 → 64) + Residual + Dropout(0.5)
       ↓
  Per-SM Gated Attention (单层) → SM vectors
       ↓
  Case-level Gated Attention → case vector [64]
       ↓
  Linear(64 → 1)  ← 关键:1 个 head,不是 9 个
       ↓
  sigmoid → EGFR 概率

参数量(hidden=64):约 0.15M
训练样本 10 个 → 参数量/样本 = 15,过拟合仍严重,需更强正则
所以额外加:
  - dropout 0.5
  - weight_decay 1e-3
  - 单层 attention(无 2 层)
  - 早期停更严(patience=8)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── 1. Backbone(更小) ───────────────
class EGFRBackbone(nn.Module):
    """1024 → hidden_dim,主路 + 残差(都从 hidden_dim=64 开始)"""
    def __init__(self, in_dim=1024, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x):
        return self.proj(x), self.residual_proj(x)


# ─────────────── 2. Gated Attention(单层,小 h) ───────────────
class GatedAttentionSmall(nn.Module):
    """简化版 Gated Attention,hidden 64 + attention 64"""
    def __init__(self, d=64, h=64):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(d, h), nn.Sigmoid())
        self.w = nn.Linear(h, 1)

    def forward(self, x, return_weights=False):
        a = self.w(self.v(x) * self.u(x))
        w = F.softmax(a, dim=0)
        out = (w * x).sum(dim=0)
        if return_weights:
            return out, w.squeeze(-1)
        return out


# ─────────────── 3. EGFR 单基因模型 ───────────────
class EGFROnlyModel(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.backbone = EGFRBackbone(in_dim, hidden_dim, dropout)
        self.sm_attn = GatedAttentionSmall(hidden_dim, h=64)
        self.case_attn = GatedAttentionSmall(hidden_dim, h=64)
        self.bag_head = nn.Linear(hidden_dim, 1)  # ← 关键:只 1 个输出
        self.hidden_dim = hidden_dim

    def encode_patches(self, patch_feats):
        return self.backbone(patch_feats)

    def aggregate_per_sm(self, hidden, sm_indices, n_sm):
        sm_vecs = []
        attn_ws = []
        for k in range(n_sm):
            mask = (sm_indices == k)
            if mask.sum() == 0:
                sm_vecs.append(torch.zeros(self.hidden_dim, device=hidden.device))
                attn_ws.append(None)
                continue
            ps = hidden[mask]
            v, w = self.sm_attn(ps, return_weights=True)
            sm_vecs.append(v)
            attn_ws.append((mask.nonzero(as_tuple=True)[0], w))
        return torch.stack(sm_vecs, dim=0), attn_ws

    def forward(self, patch_feats, sm_indices, n_sm, return_attn=False):
        hidden, residual = self.encode_patches(patch_feats)
        # Stage-1
        sm_vecs, sm_ws = self.aggregate_per_sm(hidden, sm_indices, n_sm)
        # Stage-2
        case_vec, case_w = self.case_attn(sm_vecs, return_weights=True)
        # Residual
        sm_residual, _ = self.aggregate_per_sm(residual, sm_indices, n_sm)
        case_residual = self.case_attn(sm_residual)
        case_vec = case_vec + case_residual
        # Bag head(只 1 个输出)
        bag_logit = self.bag_head(case_vec).squeeze(-1)  # [scalar]
        return bag_logit, case_w


if __name__ == "__main__":
    # 测一下参数量
    import sys
    sys.path.insert(0, '.')
    m = EGFROnlyModel()
    total = sum(p.numel() for p in m.parameters())
    print(f"EGFR-only 模型:hidden=64")
    print(f"  total params: {total:,} ({total/1e6:.3f} M)")
    print(f"  vs 10 train: {total/10:.2f}x")
    print()
    for name, mod in m.named_modules():
        import torch.nn as nn
        if isinstance(mod, nn.Linear):
            n = sum(p.numel() for p in mod.parameters())
            print(f"  {name:50s}  in={mod.in_features:5d}  out={mod.out_features:5d}  params={n:,}")