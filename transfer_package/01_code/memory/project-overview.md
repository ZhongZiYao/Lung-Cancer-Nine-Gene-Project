---
name: project-overview
description: 肺癌 9 基因预测模型科研项目 — 背景、合作方、合同硬指标、9 基因定义
metadata: 
  node_type: memory
  type: project
  originSessionId: c5983032-f1b6-4a5e-bbd7-29bfd26b2c3e
---

肺癌 9 基因预测模型科研项目,工作区 `D:\pan-caner\`。

**合作方 / 合同**:
- 北科大(乙方,算法研发) × 聚时医疗科技(甲方,数据提供)
- 合同附件三明确提到 DINOv3 自监督预训练、三维病理、LIS-PACS 接口
- 交付节点 2026-09 / 2027-03 / 2027-10

**硬指标**:
- 准确率 ≥ 95%(难点 — 钟论文 ALK AUC 0.987 但 EGFR 仅 0.825)
- 灵敏度 ≥ 90%
- 3 个专利归甲方,3 篇 SCI 归乙方(北科大)

**9 基因定义**:
- DeepGEM 已有 6 个: EGFR, KRAS, ALK, ROS1, TP53, LRP1B
- 合同 9 基因(必含): DeepGEM 6 + HER2/ERBB2 + BRAF + (RET 或 PIK3CA 或 NRAS)
- 实际训练时 `extract_9gene_panel.py` 用 11 基因留冗余: EGFR, KRAS, ALK, ROS1, TP53, BRAF, PIK3CA, ERBB2, NRAS, RET, MET

**数据集策略**(规划文档 `D:\pan-caner\肺癌九基因项目数据集与方向规划v2.md`):
- 内部数据 = 聚时医疗提供(合同第三条)
- 外部数据 = TCGA-LUAD 541 例 + 钟团队 CJFH 数据(必须异源,审稿人才认)
- 速查卡 `D:\pan-caner\TCGA_LUAD_下载速查卡.md`

**相关 memory**: [[mc3-download-status]] [[task-state]] [[environment-and-workflow]]