#!/bin/bash
# run_step3_console.sh —— 前台跑 step3,屏幕实时看到打印
#
# 用法(在自己终端跑,会一直占着屏幕):
#   bash pipeline/week_1_baseline/run_step3_console.sh
#
# 停止: Ctrl+C
# 日志: scripts 会用 tee 写到 logs/step3_<时间戳>.log (脚本完整记录,可事后 grep)
#
# 关键:`script` 给 Python 一个伪 tty,Python 自动 line-buffer,每个 print 立刻显示

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

LOG_DIR="pipeline/week_1_baseline/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/step3_${STAMP}.log"

echo "================================================================"
echo " Step3 训练开始 (前台运行 — 屏幕实时打印)"
echo " 时间: $STAMP"
echo " 模式: CPU(用 GPU 跑改 USE_GPU=1 GPU=0 bash ...)"
echo " 日志: $LOG (脚本完整记录)"
echo " 停止: Ctrl+C"
echo "================================================================"
echo

# ── 用 `script` 让 Python 拿到一个 pty,自动 line-buffer ──
# 不再用 tee/stdbuf(那些会在 conda run 子 shell 里失效)
# 直接 python ... | tee 一行解决,而且不靠 SIGUSR1

USE_GPU="${USE_GPU:-0}"

if [ "$USE_GPU" = "1" ]; then
    GPU_ID="${GPU:-0}"
    echo "[$(date +%H:%M:%S)] GPU 模式: device=cuda:$GPU_ID"
    echo "[$(date +%H:%M:%S)] 当前 GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null
    echo
    DEVICE_ARG="--device cuda"
    CUDA_PREFIX="CUDA_VISIBLE_DEVICES=$GPU_ID"
else
    echo "[$(date +%H:%M:%S)] CPU 模式 (USE_GPU=0)"
    DEVICE_ARG="--device cpu"
    CUDA_PREFIX='CUDA_VISIBLE_DEVICES=""'
fi
echo

# 关键:`script` 强制 line-buffer + tee 自动写盘
# `script -q -c "CMD" /dev/null` = 创建 pty 但不写 typescript 文件
# 然后我们再把这个 stream pipe 给 tee,tee 既显示又写盘

script -q -c "$CUDA_PREFIX conda run -n UNI python -u pipeline/week_1_baseline/step3_train_deepgem_style.py --folds 5 --epochs 50 --patience 15 --hidden-dim 512 $DEVICE_ARG" /dev/null \
    2>&1 | tee "$LOG"

echo
echo "================================================================"
echo " Step3 完成"
echo " 预测:   pipeline/week_1_baseline/outputs/predictions.csv"
echo "================================================================"

# Step 4 评估
echo
echo "--- Step 4 评估 ---"
script -q -c "$CUDA_PREFIX conda run -n UNI python -u pipeline/week_1_baseline/step4_eval.py" /dev/null \
    2>&1 | tee -a "$LOG"

echo "全部完成 ✓"
