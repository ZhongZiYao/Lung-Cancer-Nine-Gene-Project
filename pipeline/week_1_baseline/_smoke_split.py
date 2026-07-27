"""冒烟测试 split 脚本(用 --max-cases=20)"""
import subprocess
res = subprocess.run([
    r"C:\Users\ziyao\.conda\envs\deepgem\python.exe", "-u",
    r"pipeline\week_1_baseline\step3_egfr_split.py",
    "--epochs", "10", "--patience", "5",
], capture_output=True, text=True)
print("STDOUT:")
print(res.stdout[-2000:])
print("\nSTDERR:")
print(res.stderr[-1000:])
print(f"\nexit code: {res.returncode}")