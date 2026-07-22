#!/bin/bash
# monitor_train.sh —— 训练过程实时监控仪表盘
# 用法: bash pipeline/week_1_baseline/monitor_train.sh
# 显示:
#   - 当前 fold / epoch / case 进度(全局百分比 + ETA)
#   - 最近 15 个 epoch 的 loss + AUC 趋势(纯 ASCII 折线图)
#   - 9 基因当前 best AUC(从 step3.log 里 grep 出来)
#   - GPU/CPU/内存
#   - step3.log 最新 8 行
# 停止: Ctrl+C

ROOT="/home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project"
LOG_DIR="$ROOT/pipeline/week_1_baseline/logs"

# 找最新的 step3 log
get_latest_log() {
    ls -t "$LOG_DIR"/step3_*.log 2>/dev/null | head -1
}

# ASCII 折线图(给定一组数字,横轴时间,纵轴值)
ascii_plot() {
    local label="$1"
    local -a vals=("${@:2}")
    local n=${#vals[@]}
    [ "$n" -eq 0 ] && return
    # 取最近 15 个
    local start=0
    [ "$n" -gt 15 ] && start=$((n - 15))
    local max=0
    for v in "${vals[@]:$start}"; do
        v=${v%.*}
        [ "$v" -gt "$max" ] && max=$v
    done
    [ "$max" -lt 1 ] && max=1
    echo "  $label  (最近 $((n - start)) 个值, max=${max})"
    local i=$start
    while [ "$i" -lt "$n" ]; do
        local v=${vals[$i]}
        # 量化到 1..10
        local bar_len=$(awk -v x="$v" -v m="$max" 'BEGIN{printf "%d", (x/m)*10+0.5}')
        [ "$bar_len" -lt 1 ] && bar_len=1
        local bar=$(printf '█%.0s' $(seq 1 $bar_len))
        local indent=""
        [ "$i" -lt 10 ] && indent="0"
        printf "    ep%d: %s %.3f\n" "$((i+1))" "$bar" "$v"
        i=$((i+1))
    done
}

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Week-1 Baseline 训练实时监控                                  ║"
    echo "║  时间: $(date '+%Y-%m-%d %H:%M:%S')                                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    STEP3_LOG=$(get_latest_log)
    if [ -z "$STEP3_LOG" ]; then
        echo ""
        echo "  ⚠️  没有 step3_*.log,等待中..."
        sleep 5
        continue
    fi

    # 1) 全局进度 (fold / epoch / case / 全局百分)
    echo ""
    echo "▶ 全局进度"
    CUR_FOLD=$(grep -oE 'Fold [0-9]+/[0-9]+' "$STEP3_LOG" | tail -1)
    CUR_CASE=$(grep -oE 'case [0-9]+/[0-9]+  loss_so_far=' "$STEP3_LOG" | tail -1 | awk '{print $2}')
    CUR_PCT=$(grep -oE '全局 [0-9]+/[0-9]+ \([0-9.]+\)' "$STEP3_LOG" | tail -1)
    EP_AUC=$(grep -oE 'ep[0-9 ]+: loss=[0-9.]+ val_mean_auc=[0-9.]+ \([0-9.]+s\)' "$STEP3_LOG" | tail -3)
    [ -n "$CUR_FOLD" ] && echo "  fold:       $CUR_FOLD"
    [ -n "$CUR_CASE" ] && echo "  当前 case:  $CUR_CASE"
    [ -n "$CUR_PCT" ] && echo "  全局:       $CUR_PCT"

    # 2) ETA 估算(根据最近几个 epoch 的平均耗时)
    LAST_3=$(grep -oE '\([0-9.]+s\)' "$STEP3_LOG" | tail -3 | grep -oE '[0-9.]+' | awk '{s+=$1} END{if(NR>0) printf "%.0f", s/NR; else print "0"}')
    TOTAL_EPOCHS=50
    N_FOLDS=5
    CUR_FOLD_NUM=$(echo "$CUR_FOLD" | grep -oE 'Fold [0-9]+' | grep -oE '[0-9]+')
    CUR_FOLD_NUM=${CUR_FOLD_NUM:-1}
    CUR_EP=$(grep -oE 'ep[0-9 ]+:' "$STEP3_LOG" | tail -1 | grep -oE '[0-9]+')
    CUR_EP=${CUR_EP:-0}
    if [ "${LAST_3:-0}" -gt 0 ]; then
        REMAIN_EP=$(( (N_FOLDS - CUR_FOLD_NUM) * TOTAL_EPOCHS + (TOTAL_EPOCHS - CUR_EP) ))
        ETA_SEC=$(( LAST_3 * REMAIN_EP ))
        ETA_H=$(( ETA_SEC / 3600 ))
        ETA_M=$(( (ETA_SEC % 3600) / 60 ))
        printf "  ETA:        ~%d 小时 %d 分钟  (avg epoch %.0fs × %d 剩余 epoch)\n" "$ETA_H" "$ETA_M" "${LAST_3:-0}" "$REMAIN_EP"
    fi

    # 3) loss 趋势(从 ep 摘要提取)
    echo ""
    echo "▶ 训练 loss 趋势"
    LOSSES=$(grep -oE 'ep[0-9 ]+: loss=[0-9.]+' "$STEP3_LOG" | grep -oE '[0-9.]+$' | head -30)
    if [ -n "$LOSSES" ]; then
        # 转成数组
        IFS=$'\n' read -d '' -r -a LOSS_ARR < <(echo "$LOSSES")
        ascii_plot "loss" "${LOSS_ARR[@]}"
    else
        echo "  (尚未开始第一个 epoch)"
    fi

    # 4) val AUC 趋势
    echo ""
    echo "▶ 验证 mean AUC 趋势"
    AUCS=$(grep -oE 'val_mean_auc=[0-9.]+' "$STEP3_LOG" | grep -oE '[0-9.]+$' | head -30)
    if [ -n "$AUCS" ]; then
        IFS=$'\n' read -d '' -r -a AUC_ARR < <(echo "$AUCS")
        ascii_plot "AUC" "${AUC_ARR[@]}"
    else
        echo "  (尚未开始第一次 eval)"
    fi

    # 5) GPU / CPU / 内存
    echo ""
    echo "▶ 资源"
    PROC=$(ps aux | grep -E "step3_train_deepgem_style" | grep -v grep | head -1)
    if [ -n "$PROC" ]; then
        PID=$(echo "$PROC" | awk '{print $2}')
        CPU=$(echo "$PROC" | awk '{print $3}')
        CPU_T=$(echo "$PROC" | awk '{print $10}')
        RSS=$(echo "$PROC" | awk '{print $6}' | awk '{printf "%.1f GB", $1/1024/1024}')
        printf "  step3: PID=%s  CPU=%s%%  CPU_TIME=%s  RSS=%s\n" "$PID" "$CPU" "$CPU_T" "$RSS"
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | awk -F',' '{printf "  GPU%s: util=%s mem=%s/%s\n", $1, $2, $3, $4}'
    free -h | awk '/^Mem:/ {printf "  内存: used=%s total=%s\n", $3, $2}'

    # 6) 最新 8 行日志
    echo ""
    echo "▶ 最新日志 (tail -8)"
    tail -8 "$STEP3_LOG" 2>/dev/null | sed 's/^/  /'

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Ctrl+C 退出 · 每 5 秒刷新 · 当前监控: $(basename "$STEP3_LOG")"
    sleep 5
done
