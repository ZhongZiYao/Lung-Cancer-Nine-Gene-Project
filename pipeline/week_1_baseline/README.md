# Week-1 Baseline — 9 基因小模型(组会汇报用)

> ⚠️ **当前状态(2026-07-21 20:36)**: 
> - **Step 1 + Step 2**: ✅ 完成(83/83 case)
> - **Step 3**: 已杀掉旧进程(旧代码不带进度,可视化补丁已加),用 `bash pipeline/week_1_baseline/run_step3_only.sh` 重启
> - **Step 4**: 等 Step 3 自动接
> - **可视化**: 新加 `flush_print` + case-by-case 进度条 + 实时监控仪表盘 `monitor_train.sh`
>
> ⚠️ **数据状态**: `outputs/features_per_case/` 83 个 .npz + `bags_per_case/` 83 个 .npz。**不要 git push** 这些 .npz(单 case ~1 MB × 83 ≈ 80 MB);训练结果 csv/PNG 必 push,代码必 push

---

## 📌 项目说明

9 基因(EGFR/KRAS/ALK/ROS1/TP53/BRAF/PIK3CA/ERBB2/NRAS)肺腺癌突变预测,从 TCGA-LUAD WSI patch 出发,弱监督 MIL。

**架构**(融合 GAMIL + DeepGEM + Residual):
- **Backbone**: UNI(冻结,1024 维)+ Linear(1024→512) + LN + GELU + Dropout
- **Residual**: 原 1024 维 patch 走另一 Linear(1024→512)→ 加到 case vector
- **Stage-1**: per-SM Gated Attention(Ilse 2018)→ SM vector
- **Stage-2**: case-level Gated Attention → case vector
- **Prototype**: 9 基因 × 2 prototype, cosine 打 patch 软标签(辅助)
- **Bag head**: 9 个 sigmoid head(共享 backbone,9 个独立输出)
- **Instance head**: prototype 距离 → 9 个聚合(BCE 辅助)

---

## 📁 文件清单

```
pipeline/week_1_baseline/
├── README.md                  ← 本文件
├── case_manifest.csv          ← 83 case 的"真值表"
├── label_matrix.csv           ← 训练用窄表(83 × 9 + case_id)
├── deepgem_style_model.py     ← 模型定义(backbone + 9 head + prototype + residual)
├── step1_extract_features.py  ← UNI 抽 patch 特征 → <case>.npz
├── step2_pack_bags.py         ← per-SM Gated Attention 打包(冗余,留作接口)
├── step3_train_deepgem_style.py  ← 5-fold CV 训练
├── step4_eval.py              ← 评估 + 出图
├── run_full_pipeline.sh       ← 后台跑全量(start pipeline)
├── stop_pipeline.sh           ← 停后台 pipeline
├── monitor.sh                 ← 监控脚本(进度 + GPU + 进程)
├── outputs/
│   ├── features_per_case/<case>.npz  ← 每 case 500 patch × 1024 维(实际平均 288)
│   ├── bags_per_case/<case>.npz      ← step2 打的 bag(含 patch_features)
│   ├── predictions.csv               ← step3 出的 OOF 概率
│   ├── metrics_per_gene.csv          ← step4 出的 9 基因 AUC/AP/F1
│   ├── fold_summary.csv              ← 5-fold 的 best AUC
│   ├── summary.txt                   ← 文字摘要(组会用)
│   └── figures/
│       ├── roc_curves.png            ← 9 子图 ROC
│       └── group_comparison.png      ← 高频组 vs 低频组对比
└── logs/
    ├── master.log                   ← 脚本总控
    └── step1_YYYYMMDD_HHMMSS.log    ← 各 step 详细
```

---

## 🚀 跑全量(后台,不阻塞)

```bash
cd /path/to/Lung-Cancer-Nine-Gene-Project
bash pipeline/week_1_baseline/run_full_pipeline.sh

# 等 ~17 小时,自动跑完 step1 → 2 → 3 → 4
# 结果在 outputs/predictions.csv + summary.txt
```

**停止**:
```bash
bash pipeline/week_1_baseline/stop_pipeline.sh
```

### 只跑 Step 3 (Step 1/2 已跑完时)

```bash
bash pipeline/week_1_baseline/run_step3_only.sh
# 自动接 step4 出图
```

---

## 🔧 监控进度

### 实时训练仪表盘(每 5 秒刷新,推荐 — 看 loss 趋势 + 分基因 AUC)

```bash
bash pipeline/week_1_baseline/monitor_train.sh
```
显示:
```
╔════════════════════════════════════════════════════════════════╗
║  Week-1 Baseline 训练实时监控                                  ║
║  时间: 2026-07-21 20:45:00                                     ║
╚════════════════════════════════════════════════════════════════╝
▶ 全局进度
  fold:       Fold 1/5
  当前 case:  45/66
  全局:       全局 1980/13888 (14.3%)
  ETA:        ~6 小时 12 分钟   (avg epoch 142s × 157 剩余 epoch)
▶ 训练 loss 趋势 (最近 8 个值, max=0.95)
    ep1: ███████ 0.952
    ep2: ██████ 0.832
    ep3: █████ 0.711
    ...
▶ 验证 mean AUC 趋势 (最近 8 个值, max=0.65)
    ep1: ████████ 0.523
    ep2: ██████████ 0.612
    ...
▶ 资源
  step3: PID=1055282  CPU=1457%  CPU_TIME=2430:12  RSS=0.9 GB
  GPU0: util=100% mem=22971/24576 MiB
  内存: used=8.2GB total=125GB
▶ 最新日志 (tail -8)
  >>> ep3: loss=0.711 val_mean_auc=0.612 (142s)
      per-gene AUC: EGFR=0.54 KRAS=0.65 ALK=0.52 ROS1=N/A TP53=0.71 BRAF=0.55 PIK3CA=N/A ERBB2=N/A NRAS=N/A
      [fold1/5 ep4/50] case 5/66  loss_so_far=0.692  全局 2095/13888 (15.1%)
═══════════════════════════════════════════════════════════════════
```

### 基础监控(每 5 秒刷 case 数 + GPU)

```bash
bash pipeline/week_1_baseline/monitor.sh
```

### 单点查询(快速)

```bash
# case 数
ls pipeline/week_1_baseline/outputs/features_per_case/ | wc -l

# 主进程
ps -p 937159

# GPU 占用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# tail 日志
tail -f pipeline/week_1_baseline/logs/step1_*.log
```

---

## ⚙️ 超参(全量跑用这套)

| 参数 | 值 | 来源 |
|---|---|---|
| patch per case (上限) | 500 | DeepGEM 一致 |
| SM 采样策略 | **均衡**(每 SM 抽 ~N/K) | 我加的 |
| 投影维度 | 1024 → **512** | GAMIL 方向(384→500),我=512(2 的幂) |
| Residual | Linear(1024→512) 零初始化 | DeepGEM 风格 |
| Optimizer | AdamW(lr=2e-4, wd=1e-4) | DeepGEM 一致 |
| Grad clip | 1.0 | 我加的 |
| Epochs | 50 | DeepGEM 一致 |
| Patience | 15 | 我加的(早停) |
| CV | 5-fold StratifiedKFold(by any_positive) | GAMIL/DeepGEM 类似 |
| Loss | w_bag(0.7)×BCE + w_inst(0.3→0.1)×BCE | 融合设计 |
| pos_weight | clip 到 10.0 | 我加的 |
| warmup | 5 epoch(只用 bag loss) | 我加的 |

---

## 📊 数据规模

| 项 | 值 |
|---|---|
| 总 case | **83**(有标签且 patch 已切) |
| 总 SM(切片) | ~371(平均 ~4 SM/case) |
| 总 patch | 407,666(全 patch) |
| **抽特征后每 case** | 平均 288 patch(因 SM 极度不均,部分 case 不满 500) |
| 9 基因阳性率 | TP53 67.5% / KRAS 26.5% / EGFR 15.7% / ... / NRAS 2.4% |
| 高频组(每折≥1 阳性) | EGFR/KRAS/ALK/TP53/BRAF(5 基因) |
| 低频组(可能 0 阳性折) | ROS1/PIK3CA/ERBB2/NRAS(4 基因) |

---

## 📈 预期结果

| 基因 | 期望 AUC 范围 | 备注 |
|---|---|---|
| TP53 | 0.65-0.75 | 最高频,模型最稳 |
| KRAS | 0.60-0.70 | 中高频 |
| EGFR | 0.55-0.70 | 中频 |
| ALK | 0.55-0.65 | 中频 |
| BRAF | 0.55-0.65 | 中频 |
| PIK3CA | 0.50-0.60 | 低频 |
| ERBB2 | 0.50-0.60 | 低频 |
| ROS1 | 0.50-0.60 | 低频 |
| NRAS | N/A(2 例) | 5-fold 至少 1 折 0 阳性 |

> ⚠️ 论文 GAMIL 在 ~2000 case 上 AUC 0.83-0.99,我们 83 case 数字会**显著低于论文**。组会汇报时**强调小样本 + TCGA 多中心噪声**,不要对比论文数字。

---

## 📅 时间线(从 2026-07-21 14:40 起)

```
14:40  34/83 case 已抽完(后台跑着)
17:10  step1 抽完全部 83 case(预计)
17:11  step2 打包(~1 分钟)
17:15  step3 开始 5-fold CV 训练
~09:30 step3 完(5-fold × 50 epoch ≈ 15-20 小时)
~09:31 step4 出 reports
~10:00 全部完成 ✓
```

---

## 🆘 出问题怎么排查

| 现象 | 排查命令 |
|---|---|
| step1 抽不完 | `tail logs/step1_*.log` 看最后一行 |
| step3 训练慢 | `bash monitor.sh` 看 GPU/CPU 占用 |
| AUC 全 0.5 | 看 `outputs/predictions.csv` 第 1 行,9 个概率都接近 0.5 |
| OOM | 改 `--batch-size 8`,或换 GPU 跑 |
| step1 log 空 | Python buffered,看 .npz 是否在增加即可 |

---

## 📋 引用

- UNI: Chen et al., *Nat Med* 2024
- GAMIL: Zhao et al., *Diagnostic Pathology* 2025(本项目 GAMIL 方法对应文件)
- DeepGEM: He et al., *Lancet Oncology* 2025(本项目 DeepGEM 方法对应文件)
- Gated Attention: Ilse et al., ICML 2018