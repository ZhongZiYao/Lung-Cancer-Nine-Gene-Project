@echo off
setlocal
cd /d "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
"C:\Users\ziyao\.conda\envs\deepgem\python.exe" -u pipeline\week_1_baseline\step3_egfr_loo.py > _egfr_loo.log 2>&1
echo EXIT=%ERRORLEVEL% >> _egfr_loo.log
