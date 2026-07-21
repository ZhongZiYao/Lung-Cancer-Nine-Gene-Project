"""
deepgem_style_model.py
──────────────────────
DeepGEM 风格模型定义(融合 GAMIL + DeepGEM + Residual):
  - Backbone:    Linear(1024 → 512) + LN + GELU + Dropout
                 ↑ 与 GAMIL 方向一致(GAMIL: 384→500, 我们: 1024→512)
                 ↑ 512 是 2 的幂, GPU 矩阵运算友好
  - Residual:    原 1024 维 patch 特征通过 Linear(1024 → 512) → 残差加到 case vec 上
                 ↑ DeepGEM 风格(model_deepgem.py line 160 的设计)
                 ↑ 防止 1024→512 投影丢信息
  - Stage-1:     per-SM Gated Attention (Ilse 2018) → SM vector [K, 512]
  - Stage-2:     case-level Gated Attention → case vector [512]
  - Prototype:   每个基因 2 个 prototype (pos/neg), cosine 打 patch 软标签
  - Bag head:    case vec → 9 个 sigmoid head (BCE)
  - Instance head: 每个 patch 的 prototype 距离 → 9 个聚合 (BCE)

调用:
    model = DeepGEMStyleModel(in_dim=1024, hidden=512, num_genes=9)
    bag_logits, inst_logits, aux = model(patch_feats, sm_indices, n_sm)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── 1. Backbone: patch feature 投影 + Residual ───────────────
class PatchBackbone(nn.Module):
    """
    两路:
      - 主路: 1024 → hidden_dim, 经 LN + GELU + Dropout
      - 残差: 1024 → hidden_dim (线性, 不激活), 用于后续跨层残差
    初始化: 残差权重=0(训练前期不影响主路,逐渐学出有用信息)
    """
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512,
                 dropout: float = 0.25):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, in_dim]
        return: (hidden [N, hidden_dim], residual [N, hidden_dim])
        """
        return self.proj(x), self.residual_proj(x)


# ─────────────── 2. Gated Attention (Ilse 2018, GAMIL 也用) ───────────────
class GatedAttention(nn.Module):
    """
    Gated Attention MIL aggregator (Ilse et al., ICML 2018)
    输入: [N, D] 的 patch 特征
    输出: [D] 的 bag 特征
    """
    def __init__(self, d: int, h: int = 128):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(d, h), nn.Sigmoid())
        self.w = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        x: [N, D]
        return: [D] (and optionally [N] weights)
        """
        a = self.w(self.v(x) * self.u(x))     # [N, 1]
        w = F.softmax(a, dim=0)               # [N, 1]
        out = (w * x).sum(dim=0)              # [D]
        if return_weights:
            return out, w.squeeze(-1)
        return out


# ─────────────── 3. Prototype-based 标签消歧 (DeepGEM 简化版) ───────────────
class PrototypeLayer(nn.Module):
    """
    每个基因 2 个 prototype (pos/neg), 可学习。
    cosine distance → patch 属于该基因阳性的概率。
    """
    def __init__(self, hidden_dim: int, num_genes: int):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.randn(num_genes, 2, hidden_dim) * 0.1
        )
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        patch_feats: [N, hidden_dim]
        return: [N, num_genes, 2] 的 softmax 分数 (pos/neg)
        """
        pf = F.normalize(patch_feats, dim=-1)
        pt = F.normalize(self.prototypes, dim=-1)             # [G, 2, D]
        sim = torch.einsum("nd,gcd->ngc", pf, pt)
        logits = self.scale.abs() * sim
        probs = F.softmax(logits, dim=-1)
        return probs


# ─────────────── 4. 主模型 ───────────────
class DeepGEMStyleModel(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512,
                 num_genes: int = 9, dropout: float = 0.25):
        super().__init__()
        self.backbone = PatchBackbone(in_dim, hidden_dim, dropout)
        self.sm_attn = GatedAttention(hidden_dim)
        self.case_attn = GatedAttention(hidden_dim)
        self.prototype = PrototypeLayer(hidden_dim, num_genes)

        self.bag_head = nn.Linear(hidden_dim, num_genes)
        self.instance_attn = GatedAttention(num_genes)
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim

    def encode_patches(self, patch_feats: torch.Tensor):
        """patch_feats: [N, in_dim] → (hidden [N, D], residual [N, D])"""
        return self.backbone(patch_feats)

    def aggregate_per_sm(self, feats: torch.Tensor,
                         sm_indices: torch.Tensor,
                         n_sm: int) -> tuple[torch.Tensor, list]:
        """
        per-SM Gated Attention。
        feats: [N, D]; sm_indices: [N]; n_sm: K
        return: (sm_vectors [K, D], sm_attn_weights list)
        """
        sm_vecs = []
        attn_ws = []
        for k in range(n_sm):
            mask = (sm_indices == k)
            if mask.sum() == 0:
                sm_vecs.append(torch.zeros(self.hidden_dim,
                                           device=feats.device))
                attn_ws.append(None)
                continue
            ps = feats[mask]
            v, w = self.sm_attn(ps, return_weights=True)
            sm_vecs.append(v)
            attn_ws.append((mask.nonzero(as_tuple=True)[0], w))
        return torch.stack(sm_vecs, dim=0), attn_ws

    def aggregate_case(self, sm_vecs: torch.Tensor) -> torch.Tensor:
        """sm_vecs: [K, D] → [D]"""
        return self.case_attn(sm_vecs)

    def forward(self, patch_feats: torch.Tensor,
                sm_indices: torch.Tensor,
                n_sm: int,
                return_attn: bool = False):
        """
        patch_feats: [N, in_dim]
        sm_indices:  [N] int
        n_sm: int
        return:
          bag_logits:      [num_genes]
          instance_logits: [num_genes]
          aux:             dict
        """
        # 1. backbone: 投影 + 残差 (DeepGEM 风格)
        hidden, residual = self.encode_patches(patch_feats)    # [N, 512] x 2
        # 2. Stage-1: per-SM attention (主路)
        sm_vecs, sm_ws = self.aggregate_per_sm(hidden, sm_indices, n_sm)  # [K, 512]
        # 3. Stage-2: case-level attention (主路)
        case_vec, case_w = self.case_attn(sm_vecs, return_weights=True)   # [512]
        # 4. ★ Residual: 原 1024 维特征走另一路 → 加到 case_vec
        sm_residual, _ = self.aggregate_per_sm(residual, sm_indices, n_sm)  # [K, 512]
        case_residual = self.case_attn(sm_residual)                          # [512]
        case_vec = case_vec + case_residual                                 # [512]
        # 5. Bag head
        bag_logits = self.bag_head(case_vec)                                # [9]

        # 6. Prototype-based instance head
        proto_probs = self.prototype(hidden)                                # [N, 9, 2]
        instance_feats = proto_probs[..., 1]                                # [N, 9]
        instance_logits = self.instance_attn(instance_feats)                # [9]

        aux = {
            "sm_attn": sm_ws,
            "case_attn": case_w,
            "proto_probs": proto_probs,
            "hidden": hidden,
        }
        return bag_logits, instance_logits, aux