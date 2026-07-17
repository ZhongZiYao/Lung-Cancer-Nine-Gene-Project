"""Print all 16 checkpoints' parameter dicts."""
import pickle
import os

print('=' * 80)
print('所有 16 个 checkpoint 的 parameter 字段 (feature_len = 每张 WSI 最大 patch 数)')
print('=' * 80)
for ckpt in sorted(os.listdir('checkpoints/DeepGEM')):
    with open(f'checkpoints/DeepGEM/{ckpt}', 'rb') as f:
        d = pickle.load(f)
    p = d['parameter']
    print(f'{ckpt:55s} hidden_dim={p["hidden_dim"]:>4}  feat_len={p["feature_len"]:>4}  '
          f'patch_lamda={p["patch_lamda"]:>5}  depth={p["depth"]}')

print()
for ckpt in sorted(os.listdir('checkpoints/DeepGEM_TCGA')):
    with open(f'checkpoints/DeepGEM_TCGA/{ckpt}', 'rb') as f:
        d = pickle.load(f)
    p = d['parameter']
    print(f'TCGA/{ckpt:50s} hidden_dim={p["hidden_dim"]:>4}  feat_len={p["feature_len"]:>4}  '
          f'keys={list(p.keys())}')
