#!/bin/bash
# run_full_pipeline.sh —— 后台跑全量 pipeline
# 用法: bash run_full_pipeline.sh
# 停止: bash stop_pipeline.sh  或  kill <PID>

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

LOG_DIR="pipeline/week_1_baseline/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
STEP1_LOG="$LOG_DIR/step1_${STAMP}.log"
STEP2_LOG="$LOG_DIR/step2_${STAMP}.log"
STEP3_LOG="$LOG_DIR/step3_${STAMP}.log"
STEP4_LOG="$LOG_DIR/step4_${STAMP}.log"

echo "==================================================="
echo "Week-1 Baseline 全量 pipeline 启动"
echo "时间: $STAMP"
echo "Step1 log: $STEP1_LOG"
echo "Step2 log: $STEP2_LOG"
echo "Step3 log: $STEP3_LOG"
echo "Step4 log: $STEP4_LOG"
echo "==================================================="

# ───── 1. 抽 UNI 特征(78 case,跳过已抽的 5 个) ─────
echo "[$(date +%H:%M:%S)] Step 1: UNI 抽特征..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step1_extract_features.py \
    --cases-file pipeline/week_1_baseline/case_manifest.csv \
    --n-patches 500 \
    --batch-size 16 \
    --num-workers 0 \
    2>&1 | tee "$STEP1_LOG"
echo "[$(date +%H:%M:%S)] Step 1 完成"

# ───── 2. 打 bag(可选,目前是冗余包装) ─────
echo "[$(date +%H:%M:%S)] Step 2: 打 bag..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step2_pack_bags.py \
    2>&1 | tee "$STEP2_LOG"
echo "[$(date +%H:%M:%S)] Step 2 完成"

# ───── 3. 5-fold 训练 ─────
echo "[$(date +%H:%M:%S)] Step 3: 5-fold 训练..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step3_train_deepgem_style.py \
    --folds 5 \
    --epochs 50 \
    --patience 15 \
    --hidden-dim 512 \
    2>&1 | tee "$STEP3_LOG"
echo "[$(date +%H:%M:%S)] Step 3 完成"

# ───── 4. 评估 ─────
echo "[$(date +%H:%M:%S)] Step 4: 评估..."
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step4_eval.py \
    2>&1 | tee "$STEP4_LOG"
echo "[$(date +%H:%M:%S)] Step 4 完成"

echo "==================================================="
echo "全部完成 ✓"
echo "  Step1 log: $STEP1_LOG"
echo "  Step3 log: $STEP3_LOG"
echo "  预测: pipeline/week_1_baseline/outputs/predictions.csv"
echo "  指标: pipeline/week_1_baseline/outputs/metrics_per_gene.csv"
echo "  ROC:  pipeline/week_1_baseline/outputs/figures/roc_curves.png"
echo "  摘要: pipeline/week_1_baseline/outputs/summary.txt"
echo "==================================================="