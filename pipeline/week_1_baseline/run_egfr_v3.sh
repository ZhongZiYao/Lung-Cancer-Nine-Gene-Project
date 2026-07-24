#!/bin/bash
# run_egfr_v3.sh —— v3 管线:每个 case 最大 SM + K-means 过滤 → EGFR 单基因训练
# 对比 run_egfr_only.sh:
#   - 数据:每个 case 只用最大 SM(K-means 过滤非组织 patch)
#   - 模型超参:hidden=128 dropout=0.25 wd=1e-4 lr=1e-4(放宽,避免学不到)

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

LOG_DIR="pipeline/week_1_baseline/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "==========================================================="
echo "EGFR v3 管线启动(单 SM + K-means 过滤)"
echo "时间: $STAMP"
echo "==========================================================="

# ── Step1 v3:抽特征(每个 case 最大 SM, n_max=800, K-means K=2)──
STEP1_LOG="$LOG_DIR/step1_v3_${STAMP}.log"
echo
echo "[$(date +%H:%M:%S)] Step1 v3:抽特征(单 SM + K-means 过滤)..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u \
    pipeline/week_1_baseline/step1_v3_biggest_sm_kmeans.py \
    --n-max 800 \
    --k 2 \
    --batch-size 16 \
    2>&1 | tee "$STEP1_LOG"
echo "[$(date +%H:%M:%S)] Step1 v3 完成"

# ── Step2 v3:打 bag(用新 features_v3 输出到 bags_v3)──
STEP2_LOG="$LOG_DIR/step2_v3_${STAMP}.log"
echo
echo "[$(date +%H:%M:%S)] Step2 v3:打 bag..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u \
    pipeline/week_1_baseline/step2_pack_bags.py \
    --features-dir pipeline/week_1_baseline/outputs/features_v3 \
    --out pipeline/week_1_baseline/outputs/bags_v3 \
    2>&1 | tee "$STEP2_LOG"
echo "[$(date +%H:%M:%S)] Step2 v3 完成"

# ── Step3:训练 EGFR 单基因(放宽超参)──
STEP3_LOG="$LOG_DIR/step3_egfr_v3_${STAMP}.log"
echo
echo "[$(date +%H:%M:%S)] Step3:训练 EGFR 单基因(hidden=128, dropout=0.25, wd=1e-4)..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u \
    pipeline/week_1_baseline/step3_egfr_only.py \
    --bags-dir pipeline/week_1_baseline/outputs/bags_v3 \
    --out-dir pipeline/week_1_baseline/outputs_egfr_v3 \
    --hidden-dim 128 \
    --epochs 80 \
    --patience 15 \
    --folds 5 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --dropout 0.25 \
    2>&1 | tee "$STEP3_LOG"
echo "[$(date +%H:%M:%S)] Step3 完成"

echo "==========================================================="
echo "EGFR v3 全部完成 ✓"
echo "结果对比:"
echo "  v1 单 EGFR(全 SM, hidden=64 dropout=0.5 wd=1e-3):AUC=0.516"
echo "  v2 共享模型 EGFR AUC:                                  0.643"
echo "  v3 EGFR(单 SM + K-means, hidden=128 dropout=0.25 wd=1e-4):见 outputs_egfr_v3/summary.txt"
echo "==========================================================="