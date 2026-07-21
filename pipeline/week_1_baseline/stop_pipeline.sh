#!/bin/bash
# stop_pipeline.sh —— 停止后台跑的 pipeline
PIDS=$(pgrep -f "step[0-9]_" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Stopping PIDs: $PIDS"
    kill $PIDS 2>/dev/null
    sleep 2
    pgrep -f "step[0-9]_" 2>/dev/null && {
        echo "还在跑,强杀"
        pkill -9 -f "step[0-9]_" 2>/dev/null
    }
fi
# 也杀 conda run 进程
CONDA_PIDS=$(pgrep -f "conda run" 2>/dev/null)
if [ -n "$CONDA_PIDS" ]; then
    echo "Stopping conda: $CONDA_PIDS"
    kill $CONDA_PIDS 2>/dev/null
fi
echo "Done"