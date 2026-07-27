"""快速查看 bags_v3 / features_v3 中 npz 的结构"""
import numpy as np
from pathlib import Path

for name in ["bags_v3", "features_v3"]:
    base = Path(f"pipeline/week_1_baseline/outputs/{name}")
    if not base.exists():
        print(f"[{name}] 目录不存在")
        continue
    files = sorted(base.glob("*.npz"))
    print(f"\n===== {name}: {len(files)} files =====")
    good, bad = [], []
    sample_keys = None
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            keys = list(d.keys())
            if sample_keys is None:
                sample_keys = keys
            good.append(f.name)
        except Exception as e:
            bad.append((f.name, type(e).__name__))
    print(f"  good={len(good)}, bad={len(bad)}")
    if sample_keys:
        print(f"  sample keys (from {good[0]}): {sample_keys}")
    if bad:
        print(f"  first 5 bad:")
        for n, e in bad[:5]:
            print(f"    {n}: {e}")
    if good:
        f0 = base / good[0]
        d = np.load(f0, allow_pickle=True)
        print(f"  [{good[0]}] shapes:")
        for k in d.keys():
            arr = d[k]
            try:
                print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
            except Exception:
                print(f"    {k}: <obj> {arr}")