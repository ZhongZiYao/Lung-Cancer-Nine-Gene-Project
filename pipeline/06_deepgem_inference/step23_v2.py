"""step23_v2: build combined_feat/<case_id>__<slice_uid>.pkl + forward per-slice bag.

每个 SM 切片是一个独立 bag，feature 矩阵是 (n_patches_in_slice, 768)。
case-level label 继承同 case 的 EGFR 状态（一个 case 的所有 slice 共享同一 label）。

输入:  --feat    deepgem_feat/<case>/<SM_slice>/<x>_<y>.pkl
输出:  --combined-feat combined_feat/<case>__<slice>.pkl
       CSV:        case_id, slice_uid, true_label, prob_positive
       stats:      case-level AUC (mean prob over slices)
                   slice-level AUC (per-slice prob)

对比 v1:
 - v1: combined_feat/<case>.pkl = all slices concatenated, 1 bag per case
 - v2: combined_feat/<case>__<slice>.pkl = 1 bag per slice, 多 slice 多 bag
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys

# Force utf-8 (Windows gbk workaround)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Monkey-patch torch storage load for CPU/CUDA mismatch
import torch
_orig_load_from_bytes = torch.storage._load_from_bytes
def _cpu_load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
torch.storage._load_from_bytes = _cpu_load_from_bytes

from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\DeepGEM\main')
from util.misc import NestedTensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'))
    ap.add_argument('--labels', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'))
    ap.add_argument('--gene', default='EGFR')
    ap.add_argument('--combined-feat', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_combined_feat'))
    ap.add_argument('--internal-pickle', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_internal.pickle'))
    ap.add_argument('--checkpoint', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\DeepGEM\checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_EGFR.pickle'))
    ap.add_argument('--out', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\egfr_pred.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    label_map = dict(zip(df['case_id'], df[args.gene].astype(int)))

    # ---- Discover (case_id, slice_uid) pairs from feat dir ----
    print(f'[INFO] scanning {args.feat}')
    args.combined_feat.mkdir(parents=True, exist_ok=True)
    bag_records = []  # list of (bag_name, case_id, slice_uid, label, n_patches)
    for case_dir in sorted(args.feat.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if case_id not in label_map:
            continue
        label = int(label_map[case_id])
        # each subdir of case_dir = one SM slice
        slice_dirs = [p for p in case_dir.iterdir() if p.is_dir()]
        if not slice_dirs:
            continue
        for slice_dir in slice_dirs:
            slice_uid = slice_dir.name
            bag_name = f'{case_id}__{slice_uid}'
            # Aggregate patches from this slice
            per_patch = []
            for pkl in sorted(slice_dir.rglob('*.pkl')):
                with open(pkl, 'rb') as f:
                    d = pickle.load(f)
                if isinstance(d, list):
                    per_patch.extend(d)
                else:
                    per_patch.append(d)
            if not per_patch:
                continue
            # write combined_feat/<bag_name>.pkl
            out = args.combined_feat / f'{bag_name}.pkl'
            with open(out, 'wb') as f:
                pickle.dump(per_patch, f)
            bag_records.append({'bag': bag_name, 'case_id': case_id,
                                'slice_uid': slice_uid, 'label': label,
                                'n_patches': len(per_patch),
                                'feat_fp': str(out)})
    if not bag_records:
        sys.exit(f'[FATAL] no bags')
    bag_df = pd.DataFrame(bag_records)
    print(f'\n[INFO] {len(bag_df)} bags from {bag_df.case_id.nunique()} cases')
    print(bag_df.groupby('case_id').agg({'bag':'count', 'label':'first'}).to_string())

    # ---- Internal.pickle: bag_ids + labels ----
    payload = {args.wsi_type if hasattr(args, 'wsi_type') else 'ExcisionalBiopsy': {
        args.gene: {
            'test': {
                'pid': bag_df.bag.tolist(),  # bag name acts as "pid"
                'label': bag_df.label.tolist(),
            }
        }
    }}
    with open(args.internal_pickle, 'wb') as f:
        pickle.dump(payload, f)
    print(f'\n[OK] {len(bag_df)} bags  pos={int(bag_df.label.sum())}  neg={len(bag_df)-int(bag_df.label.sum())}')
    print(f'[OK] internal.pickle → {args.internal_pickle}')

    # ---- DeepGEM forward ----
    class DeepGEMDataset(torch.utils.data.Dataset):
        def __init__(self, info, feature_len):
            self.records = list(zip(info['pid'], info['feat_fp'], info['label']))
            self.feature_len = feature_len
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            bag, fp, label = self.records[idx]
            with open(fp, 'rb') as f:
                patch_feat = pickle.load(f)
            ret_feat = []
            for each_ob in patch_feat:
                if isinstance(each_ob, dict):
                    if 'val' in each_ob:
                        ret_feat.append(each_ob['val'].reshape(-1))
                    elif 'tr' in each_ob:
                        c = each_ob['tr']
                        ret_feat.append(c[np.random.choice(c.shape[0], 1).item()].reshape(-1))
            if not ret_feat:
                ret_feat = [np.zeros(768, dtype=np.float32)]
            ret_feat = np.stack(ret_feat)
            mask = np.ones(len(ret_feat))
            if len(ret_feat) > self.feature_len:
                ret_feat = ret_feat[:self.feature_len]
                mask = mask[:self.feature_len]
            elif len(ret_feat) < self.feature_len:
                res = self.feature_len - len(ret_feat)
                ret_feat = np.concatenate([ret_feat, np.zeros((res, ret_feat.shape[1]))])
                mask = np.concatenate([mask, np.zeros(res)])
            return (torch.from_numpy(ret_feat).float(),
                    torch.from_numpy(mask).bool(),
                    {'label': torch.tensor(label).long().reshape(-1).repeat(self.feature_len),
                     'pid': bag, 'case_id': bag.split('__')[0]})

    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    parameter = ckpt['parameter']
    state_dict = ckpt['checkpoint']
    print(f'[INFO] ckpt parameter: {parameter}')

    from models.model_deepgem import DeepGEM
    model = DeepGEM(num_classes=2, patch_dim=768,
                    dim=parameter['hidden_dim'], depth=parameter['depth']).to('cpu').eval()
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()
                 if not k.endswith(('total_params','total_ops'))}
    miss, unex = model.load_state_dict(new_state, strict=False)
    if not miss and not unex:
        print('[OK] DeepGEM model loaded: missing=0 unexpected=0 [OK]')
    else:
        print(f'[WARN] miss={len(miss)} unex={len(unex)}')

    info = {'pid': bag_df.bag.tolist(),
            'feat_fp': bag_df.feat_fp.tolist(),
            'label': bag_df.label.tolist()}
    ds = DeepGEMDataset(info, feature_len=parameter['feature_len'])
    print(f'[INFO] {len(ds)} bags to forward')

    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            feat, mask, target = ds[idx]
            bag = target['pid']
            case_id = target['case_id']
            label = int(target['label'][0].item())
            f_b = feat.unsqueeze(0); m_b = mask.unsqueeze(0)
            nt = NestedTensor(f_b, m_b)
            logits = model(nt)
            if isinstance(logits, tuple): logits = logits[0]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            print(f'  {bag:60s}  label={label}  P(pos)={probs[1]:.4f}')
            rows.append({'bag': bag, 'case_id': case_id,
                        'slice_uid': bag.split('__', 1)[1],
                        'true_label': label,
                        'prob_negative': float(probs[0]),
                        'prob_positive': float(probs[1])})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f'\n[OK] → {args.out}')

    # ---- Both slice-level and case-level AUC ----
    from sklearn.metrics import roc_auc_score
    try:
        slice_auc = roc_auc_score(out_df.true_label, out_df.prob_positive)
    except Exception:
        slice_auc = float('nan')
    print(f'\n=== Slice-level AUC (each SM slice 1 bag) = {slice_auc:.4f}')

    # Case-level: mean prob over slices per case
    case_df = out_df.groupby('case_id').agg({'true_label':'first', 'prob_positive':'mean'}).reset_index()
    case_df['true_label'] = case_df['true_label'].astype(int)
    print('\ncase-level aggregation:')
    print(case_df.to_string(index=False))
    try:
        case_auc = roc_auc_score(case_df.true_label, case_df.prob_positive)
    except Exception:
        case_auc = float('nan')
    print(f'\n=== Case-level AUC (mean over slices) = {case_auc:.4f}')

    print(f'\n[OK] → {args.out}')


if __name__ == '__main__':
    main()
