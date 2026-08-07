"""
egfr_only_model.py
─────────────────
EGFR-only 弱监督模型 (严格按 DeepGEM 设计 + 小数据适配)。

参考:
  DeepGEM (TencentAILabHealthcare 2022) 代码 + 论文
  https://github.com/TencentAILabHealthcare/DeepGEM

DeepGEM 原始架构 (按论文 Section 3):
  1. TransformerBackbone (depth=6, dim=32, nhead=2)
     per-layer: depth (MLP) + TransformerEncoder + residual
  2. WSI Aggregator (TransformerDecoder, 2 layers)
     - 5 个 learnable queries
     - Self-Attn → Cross-Attn → FFN
  3. PartialLoss (核心创新): EMA confidence × log_softmax
     - 每个 patch 维护一个 confidence vector
     - 用 patch-level 预测做伪标签
     - EMA 更新 confidence
     - 损失 = confidence × log_softmax (加权 CE)

小数据适配 (你当前 13 阳性 / 115 case):
  - dim: 32 (同 DeepGEM, 比 64 更稳)
  - depth: 3 (从 6 降为 3, 参数量减半)
  - num_queries: 3 (从 5 降为 3)
  - num_prototypes: 2 (从 10 降为 2, 二分类只需 pos/neg)
  - LR: 5e-4 (从 1e-3 降, 小数据更稳)
  - 单 SM 场景:跳过 case-level mean 聚合
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────── 1. Positional Encoding ───────────────
def sinusoid_pos_embed(n_patches: int, dim: int) -> torch.Tensor:
    """DeepGEM 风格的位置编码 (sinusoidal)"""
    pe = torch.zeros(n_patches, dim)
    position = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() *
                         (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ─────────────── 2. Transformer Backbone (DeepGEM 风格) ───────────────
class TransformerBlock(nn.Module):
    """DeepGEM 单层 block (depth_i):
       Linear→Tanh→LN→Dropout 5 次 → TransformerEncoder → 残差
    """
    def __init__(self, patch_dim: int, dim: int, dropout: float = 0.1):
        super().__init__()
        # MLP 主路 (5 层 Linear+Tanh+LN+Dropout)
        self.depth = nn.Sequential(
            nn.Linear(patch_dim, dim * 2),
            nn.Tanh(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 4),
            nn.Tanh(),
            nn.LayerNorm(dim * 4),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim * 2),
            nn.Tanh(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim * 2),
            nn.Tanh(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        # Transformer Encoder
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=2, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True,
        )
        # 残差 (从 patch_dim 到 dim)
        self.residual = nn.Linear(patch_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, patch_dim]"""
        residual = self.residual(x)
        x = self.depth(x)                              # [N, dim]
        x = self.encoder(x.unsqueeze(0)).squeeze(0)    # Transformer
        return x + residual                             # 残差


class TransformerBackbone(nn.Module):
    """DeepGEM 的 TransformerBackbone (depth=3 for small data)"""
    def __init__(self, patch_dim: int, dim: int = 32, depth: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(patch_dim if i == 0 else dim, dim, dropout)
            for i in range(depth)
        ])
        # Per-patch classifier (DeepGEM mlp_head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1)  # 二分类: 1 个 logit
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [N, patch_dim]
        return: (instance_logits [N], x_instance [N, dim])
        """
        for block in self.blocks:
            x = block(x)
        # Per-patch 预测 (instance head) - DeepGEM 关键!
        instance_logits = self.mlp_head(x).squeeze(-1)  # [N]
        return instance_logits, x


# ─────────────── 3. WSI Aggregator (DeepGEM Cross-Attention Decoder) ───────────────
class WSIAggregator(nn.Module):
    """DeepGEM 的 WSIAggregator: TransformerDecoder with learnable queries"""
    def __init__(self, d_model: int = 32, nhead: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 64,
                 dropout: float = 0.1, num_queries: int = 3):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)

    def forward(self, x_instance: torch.Tensor) -> torch.Tensor:
        """
        x_instance: [N, dim]  patch 特征
        return: [num_queries * dim]  flattened queries output
        """
        N = x_instance.shape[0]
        # Learnable queries
        q = self.query_embed.weight.unsqueeze(0)  # [1, Q, dim]
        # Cross-Attention: queries attend to patches
        hs = self.decoder(q, x_instance.unsqueeze(0))  # [1, Q, dim]
        return hs.flatten()  # [Q*dim]


# ─────────────── 4. PartialLoss + Confidence EMA (DeepGEM 核心) ───────────────
class PartialLoss(nn.Module):
    """DeepGEM 的 PartialLoss:
       维护每个 patch 的 confidence vector,
       用 prototype 预测做伪标签, EMA 更新,
       损失 = confidence × log_softmax
    """
    def __init__(self, num_patches: int, num_classes: int,
                 conf_ema_m: float = 0.99):
        super().__init__()
        # confidence: [N, num_classes], 初始化均匀
        self.register_buffer(
            "confidence",
            torch.full((num_patches, num_classes), 1.0 / num_classes),
        )
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch: int, max_epoch: int = 200,
                       start: float = 0.95, end: float = 0.8):
        """EMA 衰减 schedule (DeepGEM 风格)"""
        ratio = min(epoch / max_epoch, 1.0)
        self.conf_ema_m = start + (end - start) * ratio

    @torch.no_grad()
    def update(self, instance_logits: torch.Tensor, batch_indices: torch.Tensor):
        """用 patch 预测更新 confidence (EMA)"""
        pseudo_label = F.softmax(instance_logits.detach(), dim=-1)
        self.confidence[batch_indices] = (
            self.conf_ema_m * self.confidence[batch_indices]
            + (1 - self.conf_ema_m) * pseudo_label
        )

    def forward(self, instance_logits: torch.Tensor,
                batch_indices: torch.Tensor) -> torch.Tensor:
        """
        instance_logits: [B, num_classes]  patch logits
        return: scalar loss
        """
        logsm = F.log_softmax(instance_logits, dim=-1)         # [B, C]
        # 加权 CE: confidence * log_softmax
        weighted_logsm = logsm * self.confidence[batch_indices]
        return -weighted_logsm.sum(dim=-1).mean()


# ─────────────── 5. 主模型 (DeepGEM 风格) ───────────────
class EGFROnlyModel(nn.Module):
    """EGFR-only 弱监督模型 (DeepGEM 严格实现, 小数据适配)

    架构:
      patch_feats [N, 1024]
        ↓ TransformerBackbone (3 blocks, dim=32)
        ↓ → instance_logits [N], x_instance [N, 32]
        ↓
        ↓─── PartialLoss 用 instance_logits + EMA confidence
        ↓
        ↓ WSIAggregator (3 queries, cross-attn)
        ↓ → wsi_feat [3*32]
        ↓ wsi_classifier: Linear(96 → 1)
        ↓ → bag_logit
    """
    def __init__(self, in_dim: int = 1024, dim: int = 32, depth: int = 3,
                 dropout: float = 0.2, num_queries: int = 3):
        super().__init__()
        self.backbone = TransformerBackbone(in_dim, dim, depth, dropout)
        self.aggregator = WSIAggregator(dim, nhead=2,
                                          num_decoder_layers=2,
                                          dim_feedforward=dim * 2,
                                          dropout=dropout,
                                          num_queries=num_queries)
        self.wsi_classifier = nn.Linear(dim * num_queries, 1)
        self.dim = dim
        self.depth = depth

    def forward(self, patch_feats: torch.Tensor, sm_indices: torch.Tensor,
                n_sm: int, return_attn: bool = False):
        """返回: (bag_logit, instance_logits, instance_probs)"""
        # 1. Transformer Backbone
        instance_logits, x_instance = self.backbone(patch_feats)
        instance_probs = F.softmax(instance_logits.unsqueeze(-1), dim=-1)
        # [N, 1] -> [N, 2] (二分类: pos vs neg)
        instance_probs_2c = torch.cat([1 - instance_probs, instance_probs], dim=-1)
        # 2. WSI Aggregator (cross-attn with learnable queries)
        wsi_feat = self.aggregator(x_instance)
        # 3. WSI classifier
        bag_logit = self.wsi_classifier(wsi_feat).squeeze(-1)
        return bag_logit, instance_logits, instance_probs_2c


def compute_egfr_loss(bag_logit, instance_logits, label,
                      partial_loss: PartialLoss, batch_indices,
                      pos_weight_bag: float = 1.0,
                      pos_weight_patch: float = 1.0,
                      lambda_partial: float = 0.3,
                      conf_ema_m: float = 0.95):
    """DeepGEM 风格的联合损失:
       bag_loss (WSI-level CE) + λ * partial_loss (patch-level EMA confidence)

    Args:
        bag_logit: scalar WSI-level 预测
        instance_logits: [N] per-patch 预测
        label: scalar 0/1
        partial_loss: PartialLoss 实例
        batch_indices: [N] 全 0..N-1 (单 case)
        lambda_partial: 0.3 (DeepGEM 默认)
    """
    # 1. WSI-level 主损失
    bag_loss = F.binary_cross_entropy_with_logits(
        bag_logit.reshape(1), label.reshape(1).float(),
        pos_weight=torch.tensor([pos_weight_bag], device=bag_logit.device),
    )
    # 2. Patch-level PartialLoss (DeepGEM 核心)
    # instance_logits [N] → reshape [N, 2] (二分类)
    inst_logits_2c = torch.stack([instance_logits, -instance_logits], dim=-1)
    # EMA 更新 confidence
    partial_loss.set_conf_ema_m(conf_ema_m)
    partial_loss.update(inst_logits_2c.detach(), batch_indices)
    # Partial Loss
    part_loss = partial_loss(inst_logits_2c, batch_indices)
    # 3. 联合
    total = bag_loss + lambda_partial * part_loss
    return total, bag_loss, part_loss


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    m = EGFROnlyModel(in_dim=1024, dim=32, depth=3)
    total = sum(p.numel() for p in m.parameters())
    print(f"EGFR-only (DeepGEM strict): dim=32, depth=3")
    print(f"  total params: {total:,} ({total/1e6:.3f} M)")
    # 测试 forward
    N = 1000
    pf = torch.randn(N, 1024)
    sm = torch.zeros(N, dtype=torch.long)
    bag, inst, probs = m(pf, sm, 1)
    print(f"  bag_logit shape: {bag.shape}")
    print(f"  instance_logits shape: {inst.shape}")
    print(f"  instance_probs shape: {probs.shape}")
    # 测试 loss
    pl = PartialLoss(num_patches=N, num_classes=2)
    label = torch.tensor([1.0])
    idx = torch.arange(N)
    loss, bl, pl_ = compute_egfr_loss(bag, inst, label, pl, idx)
    print(f"  bag_loss={bl.item():.3f}, part_loss={pl_.item():.3f}, total={loss.item():.3f}")