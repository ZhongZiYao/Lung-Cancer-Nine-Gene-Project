#!/bin/bash
# run_step3_tmux.sh —— 用 tmux 跑 step3,可分离可再连
# 适用:想关掉终端训练不被打断,但随时能 tmux attach 看进度
#
# 用法:
#   bash pipeline/week_1_baseline/run_step3_tmux.sh
#   # 训练中:另开终端,跑  tmux attach -t step3  看进度
#   # 离开 tmux 不打断训练: Ctrl+B 然后按 D
#   # 关掉所有: tmux kill-session -t step3

set -e
cd /home/wangweipingustb/ZhongZiyao/Lung-Cancer-Nine-Gene-Project

SESSION="step3"

# 如果已存在,提示
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION' 已存在,正在 attach..."
    tmux attach -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -c "$(pwd)"
tmux send-keys -t "$SESSION" "bash pipeline/week_1_baseline/run_step3_console.sh" Enter
tmux attach -t "$SESSION"
