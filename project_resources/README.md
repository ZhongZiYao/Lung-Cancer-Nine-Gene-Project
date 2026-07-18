# `project_resources/` — 项目资料袋

> 这是从原 `D:/pan-caner/` 工作区迁过来的"上下游资料"，已重新组织为
> **SDPC 主线** 的布局。

**目标工作区**: `D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/`
**当前位置**: `D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/project_resources/`
**最后整理**: 2026-07-18

---

## 📁 目录结构（最终方案）

```
project_resources/
├── README.md                          ← 本文件
│
├── data_public/                       ← ⭐ 进 repo 的公开标签数据（训练契约）
│   ├── 9gene_panel_LUAD.csv            ← 421 cases × 11 genes 0/1 labels
│   ├── tcga_luad_cases.json           ← cBioPortal 病例白名单 (584 cases)
│   ├── manifest_50.tsv                ← 50 cases IDC 元数据
│   ├── picked_50_case_ids.json        ← 50 case_id 列表
│   └── README.md                      ← 数据契约 / DICOM 拼接规范
│
├── sdpc_pipeline/                     ← 🔥 SDPC 主线（最终生产 pipeline）
│   ├── tools/                         ← SDPC 数据准备脚本（10 个 .py）
│   │   ├── extract_patches.py         ← .sdpc → 512×512 patches
│   │   ├── extract_9gene_panel.py     ← MC3 MAF → 9gene CSV
│   │   ├── download_mc3_maf.py       ← Synapse PAT 拉 MC3
│   │   ├── sdpc_inspect.py            ← .sdpc 头部解析
│   │   ├── reconcile_sdpc_labels.py   ← .sdpc ↔ xlsx 标签对齐
│   │   ├── profile_dataset.py         ← 数据集统计
│   │   ├── _opensdpc_runtime.py      ← opensdpc .so 自动注入
│   │   ├── dump_xlsx.py / xlsx_view.py / test_sdpc.py
│   │
│   ├── notes/                         ← SDPC 相关调研笔记（不进 repo）
│   │   ├── PROJECT_HANDOVER.md       ← 项目接续指南
│   │   └── sdpc_调研报告.md          ← 13 KB，.sdpc 格式调研，国内罕见
│   │
│   └── sample_data/                   ← SDPC 抽样（不进 repo）
│       ├── sdpc_metadata_all.csv      ← 500 个 .sdpc 全量元数据 81 KB
│       ├── sdpc_metadata_samples.csv
│       └── sdpc_sample_full.json
│
└── references/                        ← 论文 reference（进 repo）
    ├── zhong_GAMIL/                   ← 钟论文 README 索引（代码见根 submodule）
    └── zhong_paper_cn/                ← 钟论文中文版 HTML + 5 figures
```

---

## 🎯 主线 vs 副线

| | 数据 | Pipeline | 模型训练 | 状态 |
|---|---|---|---|---|
| **主线（生产）** | **.sdpc** 中日冰冻切片 | `sdpc_pipeline/tools/extract_patches.py` | 最终训模型用 .sdpc | 工具就绪，等医院合作 |
| 副线（开发/sanity） | DICOM TCGA-LUAD | 根 `pipeline/extract_patches_direct.py` + `pipeline/deepgem_test/` | 当前仅做 9-gene baseline（AUC sanity） | running |

**关键概念**：
- 当前代码（根 `pipeline/` 下）跑的是 **TCGA-LUAD DICOM**——这是开发期用来
  sanity check 9-gene pipeline 框架的；
- **.sdpc 是最终生产数据**——脚本在 `sdpc_pipeline/tools/`
  里，把它们 promote 到主仓 `pipeline/` 是下一阶段的事。

---

## 🚀 复现/接力指南（在新机器上恢复）

### 1. 装 WSL Ubuntu 环境
```bash
# Linux/Mac 或 WSL
conda create -n lung9gene python=3.11
conda activate lung9gene
pip install synapseclient pydicom idc-index \
            numpy pillow matplotlib pandas scikit-learn opensdpc
```

### 2. 拉公开标签（已现成，进 repo）
- `data_public/9gene_panel_LUAD.csv`（421 cases × 11 genes，17 KB）
- `data_public/picked_50_case_ids.json`（50 case_id）

无需下载。如果你需要自己重抽：
```bash
python sdpc_pipeline/tools/extract_9gene_panel.py \
    --maf /path/to/mc3.v0.2.8.PUBLIC.maf.gz \
    --luad-whitelist data_public/tcga_luad_cases.json \
    --out  data_public/9gene_panel_LUAD.csv
```

### 3. 拉原始数据（DICOM / SDPC，进 .gitignore）
```bash
# TCGA-LUAD DICOM (50 cases, ~6 GB 当前已下完,见 data/TCGA-LUAD-WSI/)
python -c "
from idc_index import IDCClient
c = IDCClient()
import json
ids = json.load(open('data_public/picked_50_case_ids.json'))
c.download_from_selection(
    downloadDir='./data/TCGA-LUAD-WSI/dicom/',
    collection_id='tcga_luad',
    patientId=ids,
)
"

# MC3 MAF (753 MB, 走 Synapse)
echo "your_pat" > .synapse_token
python sdpc_pipeline/tools/download_mc3_maf.py
```

### 4. SDPC 路径（最终生产数据）
物理拷贝 `.sdpc` 文件到 `data/中日冰冻切片/`（不进 repo），然后：
```bash
python sdpc_pipeline/tools/extract_patches.py \
    --sdpc-dir ./data/中日冰冻切片 \
    --out-dir ./data/patches \
    --workers 8
```

### 5. 训练入口（下一步要写）
参考 `references/zhong_GAMIL/README.md` 和根 `DeepGEM/submodule` 的 amil_train.py，
写出 9 基因 multi-label MTL 训练脚本。

---

## 📚 参考论文

| Source | 位置 | 用途 |
|---|---|---|
| 钟团队 GAMIL | `references/zhong_GAMIL/`（README） + 根 submodule `Prediction-of-Mutated-Genes-.../` | 9 基因 baseline 原型 |
| 钟论文中文译版 | `references/zhong_paper_cn/index.html` | 写作参考 |
| Tencent DeepGEM | 根 submodule `DeepGEM/` | 多基因 SOTA 框架参考 |
| UNI / CONCH / TransPath | 根 submodule | patch encoder 备选 |

---

## ⚠️ 不该进 repo 的内容

| 文件/目录 | 大小 | 为何 |
|---|---|---|
| `data/`（根） | ~86 GB | TCGA-LUAD DICOM + 抽出的 patches |
| `sdpc_pipeline/sample_data/` | ~90 KB | SDPC 元数据 CSV，被 .gitignore 排除 |
| `.claude/_old_memory/` | ~20 KB | 旧 D:/pan-caner/ 会话记忆，**路径已过期**，新 Claude 会误导 |
| 所有 model weights | GB 级 | 走 huggingface / submodule |

---

## 迁移变更记录

- **2026-07-18** 重新整理：
  - 去掉了旧的 `01_code/02_data_public/04_papers/05_docs/06_assets/` 数字编号前缀
  - SDPC 工具从 `_archive/` 升级为 `sdpc_pipeline/tools/`（= **最终生产主线**）
  - 删掉了组会 PPT/Word 一次性脚本 + scan.ps1 / organize.ps1 / verify.ps1
  - 删掉了"假目录"：`04_papers/tencent_DeepGEM_code`（实际是 requirements.txt 文件）
  - 旧 Claude session memory（路径 = `D:/pan-caner/...`）挪到 `.claude/_old_memory/`
  - 钟论文代码副本删除，统一指根 submodule
- **2026-07-18** 进一步改名为 `project_resources/`（从 `transfer_package/`）
