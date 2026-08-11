"""step23_v3: build case-level bag + DeepGEM forward  (跟原始 test.py 兼容的 v2 升级)"""
from __future__ import annotations
import argparse, io, os, pickle, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
_orig = torch.storage._load_from_bytes
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)

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
    ap.add_argument('--combined-feat', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_combined_feat_v3_case'))
    ap.add_argument('--internal-pickle', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_internal_v3_case.pickle'))
    ap.add_argument('--checkpoint', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\DeepGEM\checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_EGFR.pickle'))
    ap.add_argument('--out', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\egfr_pred_v3_case.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    label_map = dict(zip(df['case_id'], df[args.gene].astype(int)))

    # --- Step A: build case-level bag (concatenate all slices per case) ---
    print(f'[INFO] building case-level bags from {args.feat}')
    args.combined_feat.mkdir(parents=True, exist_ok=True)
    pids, labels = [], []
    for case_dir in sorted(args.feat.iterdir()):
        if not case_dir.is_dir(): continue
        case_id = case_dir.name
        if case_id not in label_map: continue
        label = int(label_map[case_id])
        all_patch_feats = []  # raw 768-dim list of dicts {val, feat_name}
        for slice_dir in sorted(case_dir.iterdir()):
            if not slice_dir.is_dir(): continue
            for pkl in sorted(slice_dir.rglob('*.pkl')):
                with open(pkl, 'rb') as f:
                    d = pickle.load(f)
                if isinstance(d, list):
                    all_patch_feats.extend(d)
                else:
                    all_patch_feats.append(d)
        if not all_patch_feats: continue
        out = args.combined_feat / f'{case_id}.pkl'
        with open(out, 'wb') as f:
            pickle.dump(all_patch_feats, f)
        pids.append(case_id)
        labels.append(label)
        print(f'  {case_id}: {len(all_patch_feats)} patches (label={label})')

    # internal.pickle in DeepGEM-native schema
    payload = {'ExcisionalBiopsy': {args.gene: {'test': {'pid': pids, 'label': labels}}}}
    with open(args.internal_pickle, 'wb') as f:
        pickle.dump(payload, f)
    print(f'\n[OK] {len(pids)} cases built as DeepGEM-native bags')

    # --- Step B: DeepGEMDataset (DeepGEM-native style, 500-pad, just like test.py) ---
    class DeepGEMDataset(torch.utils.data.Dataset):
        def __init__(self, info, feature_len):
            self.wsi_loc = [(info['pid'][i], info['feat_fp'][i], info['label'][i])
                            for i in range(len(info['pid']))]
            self.feature_len = feature_len
        def __len__(self): return len(self.wsi_loc)
        def __getitem__(self, idx):
            feat_len = self.feature_len
            pid, fp, label = self.wsi_loc[idx]
            with open(fp, 'rb') as f:
                patch_feat = pickle.load(f)
            ret_feat = []
            for each_ob in patch_feat:
                if isinstance(each_ob, dict) and 'val' in each_ob:
                    ret_feat.append(each_ob['val'].reshape(-1))
            ret_feat = np.stack(ret_feat)
            mask = np.ones(len(ret_feat))
            if len(ret_feat) > feat_len:
                ret_feat = ret_feat[:feat_len]
                mask = mask[:feat_len]
            elif len(ret_feat) < feat_len:
                res = feat_len - len(ret_feat)
                ret_feat = np.concatenate([ret_feat, np.zeros((res, ret_feat.shape[1]))])
                mask = np.concatenate([mask, np.zeros(res)])
            return (torch.from_numpy(ret_feat).float(),
                    torch.from_numpy(mask).bool(),
                    {'label': torch.tensor(label).long().reshape(-1).repeat(feat_len),
                     'pid': pid})

    # --- Step C: DeepGEM model + load weights ---
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    parameter = ckpt['parameter']
    state_dict = ckpt['checkpoint']
    print(f'[INFO] ckpt: feature_len={parameter["feature_len"]}, hidden_dim={parameter["hidden_dim"]}, depth={parameter["depth"]}')

    from models.model_deepgem import DeepGEM
    model = DeepGEM(num_classes=2, patch_dim=768,
                    dim=parameter['hidden_dim'], depth=parameter['depth']).to('cpu').eval()
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()
                 if not k.endswith(('total_params','total_ops'))}
    miss, unex = model.load_state_dict(new_state, strict=False)
    print(f'[OK] DeepGEM loaded: missing={len(miss)} unex={len(unex)}')

    info = {'pid': pids,
            'feat_fp': [str(args.combined_feat / f'{p}.pkl') for p in pids],
            'label': labels}
    ds = DeepGEMDataset(info, feature_len=parameter['feature_len'])
    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            feat, mask, target = ds[idx]
            pid = target['pid']
            label = int(target['label'][0].item())
            f_b = feat.unsqueeze(0); m_b = mask.unsqueeze(0)
            nt = NestedTensor(f_b, m_b)
            logits = model(nt)
            if isinstance(logits, tuple): logits = logits[0]
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            rows.append({'case_id': pid, 'true_label': label,
                         'prob_negative': float(probs[0]),
                         'prob_positive': float(probs[1]),
                         'n_bag_patches': int(mask.sum().item())})
            print(f'  {pid:18s} label={label}  '
                  f'P(neg)={probs[0]:.6f}  P(pos)={probs[1]:.6f}  '
                  f'bag={int(mask.sum().item())} patches')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f'\n[OK] → {args.out}')

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(out_df.true_label, out_df.prob_positive)
    except Exception as e:
        auc = float('nan')
    print(f'\n=== Case-level AUC (DeepGEM-native bag, 500-pad) = {auc:.4f} ===')

    pos_df = out_df[out_df.true_label == 1]
    neg_df = out_df[out_df.true_label == 0]
    print(f'  EGFR+ cases (n={len(pos_df)}) mean P(pos) = {pos_df.prob_positive.mean():.6f}')
    print(f'  EGFR- cases (n={len(neg_df)}) mean P(pos) = {neg_df.prob_positive.mean():.6f}')

if __name__ == '__main__':
    main()
