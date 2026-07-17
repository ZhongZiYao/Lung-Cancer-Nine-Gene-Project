# 转移资料包 — 9 基因肺癌预测项目

**目的**:把所有可迁移的代码 / 文档 / 公开标签 / 论文 reference 整到一起,
方便拷到另一台有 GPU 算力的机器继续训练。

**位置**: `D:\pan-caner\transfer_package\`
**生成时间**: 2026-07-12
**作者**: zzy-ustb (中日友好医院 / 北京协和医学院 合作项目,北科大算法端)

---

## 目录结构(按重要度分类)

```
transfer_package/
├── README.md                              ← 本文件
├── organize.ps1                           ← PowerShell 整理脚本(可重跑)
│
├── 01_code/                               ← 自己写的 Python 代码
│   ├── tools/                             ← 数据处理 10 个脚本(必带)
│   ├── group_meeting_report/              ← PPT/Word 生成 + 演讲稿脚本
│   ├── scripts/                           ← PowerShell 辅助脚本
│   └── memory/                            ← Claude auto-memory(上下文)
│
├── 02_data_public/                        ← 公开数据(可分享,小)
│   ├── 9gene_panel_LUAD.csv               ← 421 cases × 11 genes 0/1 labels ⭐
│   ├── tcga_luad_cases.json               ← cBioPortal 病例白名单
│   ├── manifest_50.tsv                    ← 50 cases IDC 元数据
│   ├── picked_50_case_ids.json            ← 50 case_id 列表
│   └── README.md                          ← 数据契约说明
│
├── 03_data_sensitive/                     ← (空,留给手动添加敏感数据)
│
├── 04_papers/                             ← 论文 + 参考代码
│   ├── zhong_GAMIL_code/                  ← 钟 paper 中文版 + GAMIL 模型代码
│   └── tencent_DeepGEM_code/              ← Tencent DeepGEM 模型代码(Lancet Oncology 2024)
│
├── 05_docs/                               ← 项目文档
│   ├── 规划v2.md                          ← 项目方向规划
│   ├── 速查卡.md                          ← TCGA-LUAD 下载速查
│   ├── sdpc_metadata_all.csv              ← 中日冰冻切片元数据
│   └── zhong_paper_cn/                    ← 钟 paper 中文译版 HTML
│
└── 06_assets/                             ← 演示素材
    └── group_meeting_report/
        ├── fig_gene_prevalence.png        ← 11 基因 prevalence 图
        ├── fig_pipeline.png               ← 4 步 pipeline 流程图
        ├── fig_dicom_stitch.png           ← DICOM 拼接示意图
        ├── fig_stat_tiles.png             ← 数据集 stat tiles
        ├── patch_sample.png               ← 真实切片 1024x1024 patch
        └── wsi_thumbnail.png              ← 完整切片缩略图
```

---

## ⚠️ 需要手动添加的敏感数据(不在本包)

| 文件/目录 | 大小 | 重新获取方式 |
|---|---|---|
| `data/MC3/mc3.v0.2.8.PUBLIC.maf.gz` | 753 MB | Synapse + PAT 重新跑 `download_mc3_maf.py` |
| `data/TCGA-LUAD-WSI/dicom/tcga_luad/` | ~42 GB | `idc-index` Python 拉 50 cases |
| `data/中日冰冻切片/*.sdpc` | **~87 GB PRIVATE** | **必须物理搬运,不可云传输** |
| `data/patches/` | 558 MB | 从 .sdpc 用 `extract_patches.py` 重新提取 |
| `DeepGEM/checkpoints/` | 重 | Zenodo 下载 https://doi.org/10.5281/zenodo.13917850 |

---

## 在新机器上恢复步骤

### 1. 安装环境

```bash
# Linux/Mac 或 WSL
conda create -n pancaner python=3.11
conda activate pancaner
pip install synapseclient idc-index pydicom wsidicom highdicom \
            numpy pillow matplotlib python-pptx python-docx \
            pandas scikit-learn
# DeepGEM 还需要 PyTorch + openslide-python (见 requirements.txt)
```

### 2. 拉数据

```python
# Synapse PAT 写到 .synapse_token
echo "your_pat" > .synapse_token
chmod 600 .synapse_token
python tools/download_mc3_maf.py

# IDC 拉 50 cases WSI
python -c "
from idc_index import IDCClient
c = IDCClient()
import json
case_ids = json.load(open('transfer_package/02_data_public/picked_50_case_ids.json'))
c.download_from_selection(
    downloadDir='./data/TCGA-LUAD-WSI/dicom/',
    collection_id='tcga_luad',
    patientId=case_ids,
)
"
```

### 3. 重新生成 11 基因标签(可选,如果仅用现有 CSV)

`02_data_public/9gene_panel_LUAD.csv` 已经包含 421 cases × 11 genes 0/1 标签,
无需重新生成。直接用于训练即可。

### 4. (可选) 验证数据集

```bash
# 跑 9 gene panel extraction
python tools/extract_9gene_panel.py \
    --maf ./data/MC3/mc3.v0.2.8.PUBLIC.maf.gz \
    --out ./data/MC3/9gene_panel_LUAD.csv

# 跑 patch extraction(从 DICOM)
python tools/extract_patches.py  # 注意:目前适配 SDPC,需改 DICOM
```

### 5. 训练起点

参考 `04_papers/zhong_GAMIL_code/GAMIL/amil_train.py` 和
`04_papers/tencent_DeepGEM_code/main/engine.py` 写出 9 基因 MTL 训练脚本。

---

## 关键发现总结(给新设备上的 Claude 看)

1. **数据**: 421 TCGA-LUAD cases × 11 genes(已有 CSV),50 cases WSI 已下(DICOM)
2. **不要用 ImageNet 预训练的 ResNet50** — 用 UNI / CHIEF / CTransPath 病理 foundation model
3. **多基因联合训练(MTL)** 是关键 novelty — 不要训 9 个独立模型
4. **钟团队 GAMIL** 是 9 基因 baseline 参考,但他们用的数据是 Zenodo 上公开的(不是 .sdpc)
5. **合同硬指标 95% 准确率基本不可能单 H&E 达到** — 实际目标"比 SOTA 改进"
6. **memory/ 目录** 是 Claude 自动记忆 — 让新会话能延续上下文

---

## 迁移清单(打包时确认)

```bash
# 把整个 transfer_package 打包
7z a transfer_package.7z D:\pan-caner\transfer_package\

# 单独手拷
# - D:\pan-caner\data\中日冰冻切片\  ← 必须物理搬运
# - Synapse PAT 写到 .synapse_token ← 在新机器上重新申请或携带
```