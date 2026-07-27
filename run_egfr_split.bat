@echo off
setlocal
cd /d "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
"C:\Users\ziyao\.conda\envs\deepgem\python.exe" -u pipeline\week_1_baseline\step3_egfr_split.py --epochs 80 --patience 15 > _egfr_split.log 2>&1
echo EXIT=%ERRORLEVEL% >> _egfr_split.log
