# -*- coding: utf-8 -*-
"""13_seed_variance.py — 量化「同配置换随机种子」的波动幅度
目的: 把「创新模块的 Delta」和「纯随机波动的 Delta」放在同一把尺子上量
"""
import os, sys, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
p7 = os.path.join(R,"pipeline","data_prep","07_nnmil_v2.py")
env = dict(os.environ, PYTHONIOENCODING="utf-8")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]

def run(flags, seed):
    r = subprocess.run([sys.executable, p7, "--cohort","NSCLC","--epochs","25",
                        "--s-inst","512","--lr","3e-4","--seed",str(seed)]+flags,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    for ln in r.stdout.splitlines():
        if "mean_all" in ln:
            return float(ln.split("mean_all=")[1].split()[0])
    return None

print("=== base 配置, 5 个随机种子 ===")
base=[]
for s in [0,1,2,3,4]:
    v = run([], s); base.append(v); print(f"  seed={s}  mean_all={v}", flush=True)
b=np.array(base)
print(f"\n  base 跨种子: mean={b.mean():.4f}  std={b.std():.4f}  极差={b.max()-b.min():.4f}")
print(f"  >>> 纯随机波动幅度 (max-min) = {b.max()-b.min():.4f}")

print("\n=== 各创新模块 3 个种子 ===")
res={}
for name,fl in [("mixup",["--mixup"]),("dsmil",["--dsmil"]),("plscale",["--plscale"])]:
    vs=[run(fl,s) for s in [0,1,2]]
    res[name]=vs
    print(f"  {name:<10} {[round(v,4) for v in vs]}  mean={np.mean(vs):.4f} std={np.std(vs):.4f}", flush=True)

print(f"\n=== 结论 ===")
print(f"  base 均值           = {b.mean():.4f}")
for k,v in res.items():
    d = np.mean(v)-b.mean()
    print(f"  {k:<10} Delta = {d:+.4f}   vs 随机波动 std={b.std():.4f}  ->  {'在噪声内' if abs(d)<2*b.std() else '超出噪声'}")
print(f"\n  判定阈值: |Delta| < 2*base跨种子std ({2*b.std():.4f}) 时, Delta 不可与随机波动区分")
