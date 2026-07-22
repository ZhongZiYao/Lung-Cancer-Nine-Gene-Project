#!/bin/bash
# run_step3_only.sh —— 只跑 step3 (5-fold 训练) + step4 (评估出图)
# 适用场景:step1/step2 已完成,只想从训练开始重跑。
#
# 用法: bash pipeline/week_1_baseline/run_step3_only.sh
# 停止: kill <PID> (PID 在 logs/master_step3.log 第一行打出)

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

LOG_DIR="pipeline/week_1_baseline/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
STEP3_LOG="$LOG_DIR/step3_${STAMP}.log"
STEP4_LOG="$LOG_DIR/step4_${STAMP}.log"
MASTER_LOG="$LOG_DIR/master_step3.log"

echo "===================================================" | tee "$MASTER_LOG"
echo "Step3+Step4 only 重跑启动" | tee -a "$MASTER_LOG"
echo "时间: $STAMP" | tee -a "$MASTER_LOG"
echo "Step3 log: $STEP3_LOG" | tee -a "$MASTER_LOG"
echo "Step4 log: $STEP4_LOG" | tee -a "$MASTER_LOG"
echo "===================================================" | tee -a "$MASTER_LOG"

# 健康检查
NBAGS=$(ls pipeline/week_1_baseline/outputs/bags_per_case/ 2>/dev/null | wc -l)
NFEAT=$(ls pipeline/week_1_baseline/outputs/features_per_case/ 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] 健康检查: features=$NFEAT/83 bags=$NBAGS/83" | tee -a "$MASTER_LOG"
if [ "$NBAGS" -lt 83 ]; then
    echo "❌ bags 不足 83,需要先跑 step2" | tee -a "$MASTER_LOG"
    exit 1
fi

# ───── 3. 5-fold 训练 ─────
echo "[$(date +%H:%M:%S)] Step 3: 5-fold 训练 (CPU 模式,预计 6-8 小时)..." | tee -a "$MASTER_LOG"
echo "[$(date +%H:%M:%S)] 主进程 PID=$$" | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step3_train_deepgem_style.py \
    --folds 5 \
    --epochs 50 \
    --patience 15 \
    --hidden-dim 512 \
    2>&1 | tee "$STEP3_LOG"
echo "[$(date +%H:%M:%S)] Step 3 完成" | tee -a "$MASTER_LOG"

# ───── 4. 评估 ─────
echo "[$(date +%H:%M:%S)] Step 4: 评估..." | tee -a "$MASTER_LOG"
CUDA_VISIBLE_DEVICES="" conda run -n UNI python -u pipeline/week_1_baseline/step4_eval.py \
    2>&1 | tee "$STEP4_LOG"
echo "[$(date +%H:%M:%S)] Step 4 完成" | tee -a "$MASTER_LOG"

echo "===================================================" | tee -a "$MASTER_LOG"
echo "全部完成 ✓" | tee -a "$MASTER_LOG"
echo "  预测:   pipeline/week_1_baseline/outputs/predictions.csv" | tee -a "$MASTER_LOG"
echo "  指标:   pipeline/week_1_baseline/outputs/metrics_per_gene.csv" | tee -a "$MASTER_LOG"
echo "  ROC 图: pipeline/week_1_baseline/outputs/figures/roc_curves.png" | tee -a "$MASTER_LOG"
echo "  摘要:   pipeline/week_1_baseline/outputs/summary.txt" | tee -a "$MASTER_LOG"
echo "===================================================" | tee -a "$MASTER_LOG"
