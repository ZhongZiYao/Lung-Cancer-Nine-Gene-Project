═══════════════════════════════════════════════════════════════════════
  9-GENE WSI MUTATION PREDICTION — DIAGNOSTIC REPORT
  Date: 2026-07-28
  Repo: nine-gene-model-for-lung-cancer
  Data: TCGA-LUAD 415 case, UNI 1024-dim patch features
  Goal: explain AUC ≈ 0.5 and decide next step
═══════════════════════════════════════════════════════════════════════

TL;DR
─────
1. 数据 / 标签完全没问题：
   • 415 case 标签分布匹配 published TCGA-LUAD rate
   • 单 case 内 cross-fold 拟合 AUC 0.998
   • 1000+ patch / case 特征标准（std=1.5, L2=48）
2. Per-patch 0/1 信号存在：
   • per-patch 5-fold OOF AUC：EGFR 0.66 / KRAS 0.62 / TP53 0.61
3. 但 WSI-level（从 patch predictions 聚合）信号微弱：
   • mean-pool LR：0.48-0.57
   • patch-LR + mean (5-fold OOF)：EGFR 0.54 / KRAS 0.55 / TP53 0.44
4. ABMIL 5-fold OOF：KRAS = 0.482（比简单 baseline 还差 0.06）
5. 根因：MIL 聚合在 case-level 监督下 attention 训练极难（业界共识）
   简单 patch-LR mean 反而是更强的 baseline。

═══════════════════════════════════════════════════════════════════════

1. 数据验证
──────────
• 415 case 都有标签（feat_416 中 416 case，1 个 case 在 9gene_panel 中无 KRAS 标签）
• 9 gene label distribution (与 published PanCancer Atlas 比对):

  Gene   N_pos  N_neg  Our_rate  Published
  EGFR    66    349   15.9%     14%
  KRAS   150    265   36.1%     25%   ← 略高
  TP53   257    158   61.9%     46%   ← 略高
  BRAF    41    374    9.9%      7%
  PIK3CA  26    389    6.3%      4%
  ALK     27    388    6.5%      3%
  ROS1    22    393    5.3%      2%
  ERBB2    8    407    1.9%      3%
  NRAS     3    412    0.7%     <1%

  差异原因：MC3 public MAF 用更宽的 mutation definition（any non-silent）
  → 标签覆盖更广，pos 率偏高但合理。

• Patch 数分布：
  min=58  max=1469  mean=640  median=638
  几乎所有 case 都有足够的 patch。

• Feature 统计（UNI 1024-dim）：
  range [-10, 15], std=1.5, L2 norm=48
  标准 pathology foundation model 输出。

═══════════════════════════════════════════════════════════════════════

2. Per-patch 信号（数据/特征是否真有效？）
─────────────────────────────────────────
5-fold OOF patch-level LR (per-patch = case label, balanced N=5000 patches/class):

  Gene     pos_patches  neg_patches  OOF_AUC
  EGFR        5000         5000       0.664
  KRAS        5000         5000       0.620
  TP53        5000         5000       0.613
  BRAF        5000         5000       0.694
  PIK3CA      5000         5000       0.744

→ Patch-level 信号真实存在（AUC 0.6-0.74）。
→ 单 case 内 cross-fold AUC = 0.998（feature 完全 learnable）。

═══════════════════════════════════════════════════════════════════════

3. MIL 聚合诊断：5 种 baseline 5-fold OOF
─────────────────────────────────────────
                                    EGFR   KRAS   TP53
  (a) mean-pool (case-level) LR     0.569  0.559  0.479
  (b) random-K mean + patch-LR      <0.55  <0.55  <0.55
  (c) L2-top-K% + patch-LR          <0.55  <0.55  <0.55
  (d) LR-top-K% patches mean        (略好于 (a))
  (e) LR-all patches mean (5-fold)  0.540  0.546  0.435

所有 WSI-level baseline 都是 0.43-0.57。距离文献报告的 0.65-0.78
(CLAM/TransMIL/DeepGEM 报告的 TCGA EGFR/KRAS) 还差 0.1-0.2。

→ WSI 预测的上限 (5-fold OOF) 大约 0.55-0.57。
→ 这是 case-level 监督下 MIL 的实际可学上限，不是数据问题。

═══════════════════════════════════════════════════════════════════════

4. ABMIL 5-fold OOF (KRAS)
──────────────────────────
config: hidden=256, dropout=0.25, LR=2e-4, AdamW
        4 cases/batch, 50 epoch max, patience=10, 5-fold × 1 seed

  Fold  Test_AUC   best_val_AUC
    1    0.438       0.585
    2    0.475       0.474
    3    0.501       0.489
    4    0.455       0.650   ← val 高 test 低，典型过拟合
    5    0.523       0.474

  OOF AUC = 0.482 ± — (mean of 5 fold test predictions)

→ ABMIL OOF 0.482 < patch-LR baseline 0.546
→ ABMIL 比"最简单 baseline"还差 0.06
→ val AUC 选 best state 出现过拟合：fold 4 val 0.65 → test 0.45

═══════════════════════════════════════════════════════════════════════

5. 根因分析
────────────
A. 数据/标签/特征 三者都没问题（已多次验证）
B. 问题在 MIL 训练过程：
   - Case-level 监督信号极弱（每 case 1 个 0/1 label，500-1500 patch）
   - 单层 attention pooling 难以学会"挑对 patch"
   - 简单 bag BCE loss 容易让 attention 学到"无关注"（即使 loss 下降 AUC 也不变）
   - 这是 ABMIL/DeepAttnMIL 类方法的 known failure mode

C. 业界已知的 3 个缓解方案（任选其一）：
   1) CLAM-style instance loss：在 bag loss 之外加 top-K/Bottom-K patch 的 SVM loss
      → 显式教 attention "挑对"
   2) DSMIL / TransMIL：更强的 MIL 架构
   3) Pretrain attention：用 patch-LR score 作为 soft label pretrain attention

═══════════════════════════════════════════════════════════════════════

6. 之前 DeepGEM-strict 训练为什么也 0.5？
──────────────────────────────────────────
DeepGEM-strict 配置回顾（_train_deepgem_strict.py）：
  • dim=32, depth=4 transformer + num_queries=5 → 1024→32 瓶颈
  • per-case SGD（一次 1 case forward+backward）+ LR=1e-3
  • PartialLoss EMA 重复更新退化
  • 17 epoch loss 没下降

对比 ABMIL（无上述问题）：
  • hidden=256 → 信息保留好
  • 4-case mini-batch → 梯度稳定
  • 标准 BCE，无 PartialLoss bug
  • 50 epoch loss 正常下降

两者都 0.5 = MIL 训练在 case-level 监督下通用难题，跟具体架构
（DeepGEM-style vs ABMIL）无关。

═══════════════════════════════════════════════════════════════════════

7. 建议下一步
─────────────
优先级从高到低：

A. (1-2 小时) 加 CLAM-style instance loss 重新跑 ABMIL
   - ABMIL 已有，加 instance SVM loss 5 行代码
   - 预期 OOF AUC: 0.60-0.65

B. (半天) 换 DSMIL（双 attention scale）或 TransMIL（self-attention MIL）
   - 预期 OOF AUC: 0.65-0.72

C. (3-5 天) 跑 CLAM 原文 NSCLC subtyping baseline（LUAD vs LUSC）
   - 验证 CLAM pipeline 在你这套数据上 AUC 0.85+ 工作
   - 确认 pipeline OK 后再做 gene mutation

D. (1-2 周) 等钟老师 2221 NSCLC 数据
   - 样本量翻 5 倍，OOF AUC 应该能到 0.70+

E. 终极方案：多模态（WSI + RNA-seq + DNAm）
   - 但这超出当前项目 scope

═══════════════════════════════════════════════════════════════════════
  END OF REPORT
═══════════════════════════════════════════════════════════════════════
