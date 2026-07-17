# DeepGEM Sanity Check 管线

用 **DeepGEM 已训练好的 EGFR 模型** (`modelTCGA_ExcisionalBiopsy_EGFR.pickle`) 对**你本地重新切的 patch** 做 3-case sanity check。

## 约束

- DeepGEM 模型 patch_dim = **768**（CTransPath 输出）
- DeepGEM 模型 hidden_dim = **128**, depth = **2**, feature_len = **500**
- 你 pipeline 默认切 **256×256 patch** ← CTransPath 原生输入是 256×256，无 Resize 损失
- 3 个 case: TCGA-05-4382 (EGFR+), TCGA-05-4249 (-), TCGA-05-4395 (-)

## 5 步流水线

| 步骤 | 脚本 | 输出 |
|---|---|---|
| Step 1 | `step1_repatch_3cases.py` | `patches/TCGA-XX-XXXX/<slice>/<x>_<y>.jpg` |
| Step 2 | `step2_extract_ctranspath.py` | `features/<case>.npz` (768-dim / patch) |
| Step 3 | `step3_make_deepgem_pkl.py` | `deepgem_feat/<case>.pkl` (list of dict) |
| Step 4 | `step4_build_internal_pkl.py` | `data/internal_3case.pickle` (multilayer dict) |
| Step 5 | `step5_predict_egfr.py` | print 3 case forward + softmax |

## 跑法

环境: `conda activate deepgem`（torch + timm 已装）

```bash
cd pipeline/deepgem_test

# 1. 切 patch (10~30 min)
python step1_repatch_3cases.py

# 2. CTransPath 抽特征 (需 GPU, 5 min)
python step2_extract_ctranspath.py

# 3. 转 DeepGEM .pkl
python step3_make_deepgem_pkl.py

# 4. 构造 internal.pickle (从 9gene_panel_LUAD.csv 读 label)
python step4_build_internal_pkl.py --gene EGFR

# 5. 跑 forward (eager mode)
python step5_predict_egfr.py --gene EGFR
```

## ⚠️ 关于 768 vs 1024

如果你用 `step1_extract_features_uni.py` (UNI 抽 1024-dim),step3 会截断到 768 维,这样**特征语义空间错位 → 模型 forward 输出接近随机**。

正确做法是**用 step2 (CTransPath) 抽 768-dim**。

## Sanity vs Real Benchmark

| 项 | sanity check 输出 | 真实 benchmark |
|---|---|---|
| accuracy | 50~67% | 80~85% (论文) |
| pos_prob dist | 0.4~0.6 | EGFR+ > 0.5, EGFR- < 0.5 明显分离 |
| label mapping | 必须对 | 必须对 |
| forward shape | [1, 2] | [1, 2] |

只要管线能跑通 (forward 输出 shape 对、pos_prob 不全是 0.5、label 加载对),就验证了你处理后的数据**结构上兼容** DeepGEM。**真实精度** 取决于:
1. Patch 来自 CTransPath 训练域 (TCGA LUAD H&E)
2. Patch 切法跟 DeepGEM 一致 (256×256 + 简单 bg filter)
3. WSI 来自 TCGA → 训练域适配
