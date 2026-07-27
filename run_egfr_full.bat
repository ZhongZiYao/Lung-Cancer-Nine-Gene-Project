@echo off
setlocal
cd /d "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
"C:\Users\ziyao\.conda\envs\deepgem\python.exe" -u pipeline\week_1_baseline\step3_egfr_only.py --bags-dir pipeline\week_1_baseline\outputs\bags_v3 > _egfr_full_train.log 2>&1
echo EXIT=%ERRORLEVEL% >> _egfr_full_train.log
