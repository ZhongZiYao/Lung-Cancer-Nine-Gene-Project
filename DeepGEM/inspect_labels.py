"""Inspect WSI and patch level labels in DeepGEM data."""
import pickle
import numpy as np
import os

ROOT = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\DeepGEM"

# === 1. WSI-level label: sample.csv ===
print("=" * 70)
print("1. WSI 级标签：sample.csv")
print("=" * 70)
with open(os.path.join(ROOT, "sample_data", "sample.csv"), "r") as f:
    print(f.read())

# === 2. WSI-level pkl (combined_feat) ===
print("=" * 70)
print("2. WSI 级 pkl: combined_feat/tcga-49-6743-01z-00-dx2.pkl")
print("=" * 70)
with open(os.path.join(ROOT, "sample_data", "combined_feat",
                       "tcga-49-6743-01z-00-dx2.pkl"), "rb") as f:
    obj = pickle.load(f)
print(f"  type: {type(obj)}")
print(f"  len: {len(obj)} 个 patch entry")
print()
print("  === 第 0 个 patch entry ===")
patch0 = obj[0]
print(f"    type: {type(patch0)}")
print(f"    keys: {list(patch0.keys())}")
print(f"    feat_name: {patch0['feat_name']}")
print(f"    'val' shape: {patch0['val'].shape}, dtype: {patch0['val'].dtype}")
print(f"    'val' 前 5 维: {patch0['val'][:5]}")
print()
# Check if 'tr' exists (only present in training, not inference)
if 'tr' in patch0:
    print(f"    'tr' shape: {patch0['tr'].shape}, dtype: {patch0['tr'].dtype}")
    print(f"    'tr' 前 5 维: {patch0['tr'][0, :5]}")
    print(f"    tr == val ?  {np.allclose(patch0['tr'][0], patch0['val'][0])}")
else:
    print("    (没有 'tr' 字段 — 这是推理/inference 阶段生成的 pkl)")
    print("    (训练时 step3 会同时存 'tr' 和 'val'，但推理时只存 'val')")
print()

# === 3. patch-level pkl (sample_data/feat) ===
print("=" * 70)
print("3. patch 级 pkl: feat/tcga-05-4250-01z-00-dx1/10_10-...")
print("=" * 70)
sample_patch = os.path.join(ROOT, "sample_data", "feat",
                            "tcga-05-4250-01z-00-dx1",
                            "10_10-tile-r22400-c22400-1120x1120.pkl")
with open(sample_patch, "rb") as f:
    obj = pickle.load(f)
print(f"  type: {type(obj)}")
print(f"  keys: {list(obj.keys())}")
if 'tr' in obj:
    print(f"  'tr'  shape: {obj['tr'].shape}, dtype: {obj['tr'].dtype}")
    print(f"  'tr'  前 5 维: {obj['tr'][0, :5]}")
print()
print("  'val' shape:", obj['val'].shape)
print("  'val' 前 5 维:", obj['val'][:5])
print()
print("  ⚠️  注意：单独 patch pkl 里没有 'label' 字段！")
print("     'label' 来自 WSI 级的 sample.csv / pickle")
print()

# === 4. dict_name2imgs.pkl (背景过滤后的 patch 列表) ===
print("=" * 70)
print("4. 背景过滤索引: dict_name2imgs_path_rmbg.pkl")
print("=" * 70)
with open(os.path.join(ROOT, "sample_data", "dict_name2imgs_path_rmbg.pkl"), "rb") as f:
    d = pickle.load(f)
print(f"  type: {type(d)}")
print(f"  keys: {list(d.keys())}")
for k, v in d.items():
    print(f"  {k}: {len(v)} 个 patch 路径")
    print(f"    前 3 个: {v[:3]}")
