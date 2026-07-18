# 9-gene 项目 — 接续指南（给新会话 Claude 看）

> 工作区已从 `D:/pan-caner/` 迁到 `D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/`。
> 旧 memory 在根 `.claude/_old_memory/`（路径过期，**不要**自动加载）。

---

## 项目一句话

用 H&E 全片病理图像，预测 TCGA-LUAD（开发期）+ 中日友好医院冰冻切片（最终生产）的
**9 个 driver gene mutation 状态**：

`EGFR  KRAS  ALK  ROS1  TP53  BRAF  PIK3CA  ERBB2  NRAS  RET`

训练时留 11 基因冗余（含 LRP1B / MET）。详见 `project_resources/data_public/9gene_panel_LUAD.csv`。

---

## 双数据流并行的关键事实

| | 数据 | Pipeline 位置 | 状态 |
|---|---|---|---|
| **开发/sanity** | TCGA-LUAD DICOM（50 cases, 已下） | 根 `pipeline/extract_patches_direct.py` + `pipeline/deepgem_test/` | running，多基因 baseline |
| **生产** | 中日友好医院冰冻切片 `.sdpc`（500 例, 未下） | `project_resources/sdpc_pipeline/tools/extract_patches.py` | 工具就绪，等合作方数据 |

**不要混淆**：训练入口最终要切到 SDPC。当前根 `pipeline/` 是开发期验证，9 基因 baseline
做出来后，要被 SDPC pipeline 替代。

---

## 关键文件速查

| 用途 | 路径 |
|---|---|
| 训练标签 | `project_resources/data_public/9gene_panel_LUAD.csv` ⭐ |
| 50 cases WSI 列表 | `project_resources/data_public/picked_50_case_ids.json` |
| DICOM 数据契约 | `project_resources/data_public/README.md` |
| 当前 DICOM pipeline | 根 `pipeline/extract_patches_direct.py` |
| 最终 SDPC patch 工具 | `project_resources/sdpc_pipeline/tools/extract_patches.py` |
| MC3 拉取工具 | `project_resources/sdpc_pipeline/tools/download_mc3_maf.py` |
| 9 基因抽标签工具 | `project_resources/sdpc_pipeline/tools/extract_9gene_panel.py` |
| .sdpc 格式调研 | `project_resources/sdpc_pipeline/notes/sdpc_调研报告.md` |

---

## 进展 (2026-07-18)

- ✅ 重新整理 project_resources 布局，SDPC 升为正式主线目录
- ✅ 训练标签 CSV 已在 repo（data_public/）
- ✅ 目录从 `transfer_package/` 改名为 `project_resources/`
- 🔄 当前 50 cases 全量跑 + 5-fold CV（根 `pipeline/deepgem_test/`）doing

下一步接力 Claude 该做的事：

1. 读 `project_resources/data_public/README.md` 弄清训练契约
2. 检查 `pipeline/deepgem_test/` 进展，看 9-gene baseline AUC
3. 评估 SDPC pipeline 是否需要补 `extract_features.py`（对应根 `pipeline/step2_extract_ctranspath.py`）
4. 比较 baseline（DeepGEM 训好的 EGFR 模型 forward sanity）vs MTL 9-gene 训模型

---

## 工程坑（接力必读）

- TCGA-LUAD 数据走 IDC，不是 GDC（后者 PEAR-1166 legacy archive 已关闭）
- IDC collection 名是 `tcga_luad` 下划线，不是连字符
- SDPC 解码需 opensdpc，Linux 走 pip，Windows 需自带 `DecodeSdpcDll.dll`
- DICOM tile 用 `(0048,0070)/(0048,0071)` tag 拼，不要依赖文件名
- 病理 patch encoder 不要用 ImageNet ResNet50，用 CTransPath / UNI / CONCH

---

## 相关 memory

- 旧 `D:/pan-caner/` 会话记忆：根 `.claude/_old_memory/`（**过期，慎用**）
