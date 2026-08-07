"""
abmil_model.py
──────────────
ABMIL (Gated Attention) baseline for WSI-level binary mutation prediction.

Reference: CLAM (Lu et al., Nature BME 2021) — 256-dim hidden, Gated Attention,
single-branch.

Why this exists:
  用户原 EGFROnlyModel 是 DeepGEM 风格（dim=32, depth=4 transformer + prototype +
  PartialLoss EMA），在 UNI 1024-dim 输入 + 416 case + per-case SGD 配置下 AUC ≈ 0.5。
  改用 ABMIL 标准 baseline：1024 → 256 保留 25% 信息（vs DeepGEM 32 = 3%），
  单层 attention pooling，无 transformer，mini-batch 多 case 训练。
  30 分钟内可验证"AUC 0.5 是模型问题不是数据问题"。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """CLAM-style gated attention scoring."""
    def __init__(self, dim: int = 256, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(dim, attn_dim)
        self.attn_U = nn.Linear(dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, D]  →  attention scores [N] (per-instance, NOT softmaxed)"""
        h = self.proj(x)
        a_v = torch.tanh(self.attn_V(h))
        a_u = torch.sigmoid(self.attn_U(h))
        a = self.attn_w(a_v * a_u).squeeze(-1)  # [N]
        return a


class ABMIL(nn.Module):
    """Attention-based MIL — single branch (CLAM-SB 同款, 但去 instance head)."""
    def __init__(self, in_dim: int = 1024, hidden: int = 256,
                 attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.patch_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden, attn_dim, dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, patch_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patch_feats: [N, in_dim]
        returns:
            bag_logit: scalar
            attn:      [N] attention weights (post-softmax, sum=1)
        """
        h = self.patch_proj(patch_feats)  # [N, hidden]
        a = self.attention(h)             # [N]
        w = torch.softmax(a, dim=0)        # [N]
        z = (w.unsqueeze(-1) * h).sum(0)   # [hidden]
        logit = self.classifier(z).squeeze(-1)
        return logit, w


def abmil_bce_loss(logit: torch.Tensor, label: torch.Tensor,
                   pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Standard WSI-level BCE with optional pos_weight for class imbalance."""
    return F.binary_cross_entropy_with_logits(
        logit.reshape(1), label.reshape(1).float(),
        pos_weight=pos_weight,
    )


if __name__ == "__main__":
    # quick smoke
    m = ABMIL(in_dim=1024)
    pf = torch.randn(800, 1024)
    logit, w = m(pf)
    print(f"ABMIL params: {sum(p.numel() for p in m.parameters()):,}")
    print(f"  logit: {logit.item():.3f}")
    print(f"  attn sum: {w.sum().item():.3f} (should be 1.0)")
    loss = abmil_bce_loss(logit, torch.tensor([1.0]))
    print(f"  loss: {loss.item():.3f}")
