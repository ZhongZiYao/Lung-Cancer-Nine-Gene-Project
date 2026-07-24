#!/bin/bash
# run_step3_v2.sh —— 后台跑 v2 模型(hidden=256, dropout=0.5, wd=1e-3)
# 跳过 step1(step1 输出 .npz 不变,hidden_dim 是模型内部维度,不用重抽)
# 跳过 step2(features + bags 已 copy 到 outputs/)
# 直接跑 step3 + step4

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

LOG_DIR="pipeline/week_1_baseline/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
STEP3_LOG="$LOG_DIR/step3_v2_${STAMP}.log"
STEP4_LOG="$LOG_DIR/step4_v2_${STAMP}.log"

echo "==================================================="
echo "Week-1 Baseline v2:hidden=256, dropout=0.5, wd=1e-3"
echo "时间: $STAMP"
echo "Step3 log: $STEP3_LOG"
echo "Step4 log: $STEP4_LOG"
echo "原 v1 结果: pipeline/week_1_baseline/outputs_v1_baseline/"
echo "新 v2 结果: pipeline/week_1_baseline/outputs/"
echo "==================================================="

# ───── 1. 跳过 step1(已抽 83 case,无需重抽) ─────
echo "[$(date +%H:%M:%S)] 跳过 Step 1(features_per_case/ 已有 83 个 .npz)"

# ───── 2. 跳过 step2(bags_per_case/ 已 copy) ─────
echo "[$(date +%H:%M:%S)] 跳过 Step 2(bags_per_case/ 已有 83 个 .npz)"

# ───── 3. 5-fold 训练(v2) ─────
echo "[$(date +%H:%M:%S)] Step 3 v2: 5-fold 训练 (hidden=256, dropout=0.5, wd=1e-3)..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step3_train_deepgem_style.py \
    --folds 5 \
    --epochs 50 \
    --patience 15 \
    --hidden-dim 128 \
    --lr 2e-4 \
    --weight-decay 1e-3 \
    2>&1 | tee "$STEP3_LOG"
echo "[$(date +%H:%M:%S)] Step 3 完成"

# ───── 4. 评估(v2) ─────
echo "[$(date +%H:%M:%S)] Step 4 v2: 评估..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step4_eval.py \
    2>&1 | tee "$STEP4_LOG"
echo "[$(date +%H:%M:%S)] Step 4 完成"

echo "==================================================="
echo "v2 训练完成 ✓"
echo "对比 v1 结果:"
echo "  v1: outputs_v1_baseline/summary.txt"
echo "  v2: outputs/summary.txt"
echo "==================================================="