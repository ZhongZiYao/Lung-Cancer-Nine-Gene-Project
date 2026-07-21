#!/bin/bash
# monitor.sh —— 每 5 秒刷新 pipeline + GPU 状态
# 用法: bash pipeline/week_1_baseline/monitor.sh
# 停止: Ctrl+C

ROOT="/home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project"
while true; do
    clear
    echo "═══════════════════════════════════════════════════"
    echo "  Week-1 Baseline 监控"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════"

    echo ""
    echo "▶ Case 进度"
    FEAT_DIR="$ROOT/pipeline/week_1_baseline/outputs/features_per_case"
    BAG_DIR="$ROOT/pipeline/week_1_baseline/outputs/bags_per_case"
    PRED="$ROOT/pipeline/week_1_baseline/outputs/predictions.csv"
    n_feat=$(ls "$FEAT_DIR" 2>/dev/null | wc -l)
    n_bag=$(ls "$BAG_DIR" 2>/dev/null | wc -l)
    echo "  step1 已抽 case: $n_feat / 83"
    echo "  step2 已打 bag:  $n_bag"
    [ -f "$PRED" ] && echo "  step3 ✓ predictions.csv 已生成" || echo "  step3 ⏳ 训练中"

    echo ""
    echo "▶ 后台进程"
    ps -p 937159 > /dev/null 2>&1 && echo "  ✓ 主进程 937159 活着" || echo "  ✗ 主进程 937159 死了"
    ps aux | grep -E "step1_extract|step2_pack|step3_train|step4_eval" | grep -v grep | \
        awk '{printf "  - PID=%s  CPU=%s%%  CPU_TIME=%s\n", $2, $3, $10}' | head -3

    echo ""
    echo "▶ GPU 状态"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader 2>/dev/null | awk -F',' '{printf "  GPU%s: used=%s  free=%s  util=%s\n", $1, $2, $3, $4}'

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Ctrl+C 退出 · 每 5 秒刷新"
    sleep 5
done