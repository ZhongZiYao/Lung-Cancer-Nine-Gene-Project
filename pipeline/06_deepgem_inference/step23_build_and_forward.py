"""Use DeepGEM's official model_deepgem.py + monkey-patched torch storage loader + CTransPath features."""
from __future__ import annotations

import argparse
import os, sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Monkey-patch torch storage load to force CPU (BEFORE importing torch models) ----
import torch
_orig_load_from_bytes = torch.storage._load_from_bytes
def _cpu_load_from_bytes(b):
    import io
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
torch.storage._load_from_bytes = _cpu_load_from_bytes

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

# ---- Add DeepGEM/main to path so we can import their models ----
sys.path.insert(0, r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\DeepGEM\main')
# Use real NestedTensor from util.misc (defines .tensors / .mask that model.forward expects)
from util.misc import NestedTensor


# ---- DeepGEMDataset (兼容 DeepGEMDataset.build) ----
class DeepGEMDataset(data_utils.Dataset):
    def __init__(self, data_info, feature_len):
        self.wsi_loc = []
        for i in range(len(data_info['pid'])):
            self.wsi_loc.append((data_info['pid'][i], data_info['feat_fp'][i], data_info['label'][i]))
        self.feature_len = feature_len

    def __len__(self):
        return len(self.wsi_loc)

    def __getitem__(self, index):
        feat_len = self.feature_len
        image_id, patch_feat_fp, label = self.wsi_loc[index]
        with open(patch_feat_fp, "rb") as f:
            patch_feat = pickle.load(f)
        ret_feat, ret_feat_name = [], []
        for each_ob in patch_feat:
            if 'val' in each_ob:
                select_feat = each_ob['val']
            elif 'tr' in each_ob:
                c_feat = each_ob['tr']
                idx = np.random.choice(c_feat.shape[0], 1).item()
                select_feat = c_feat[idx]
            else:
                continue
            select_feat = select_feat.reshape(-1)
            ret_feat.append(select_feat)
            ret_feat_name.append(each_ob.get('feat_name', image_id))
        ret_feat = np.stack(ret_feat)
        mask = np.ones(len(ret_feat))
        if len(ret_feat) > feat_len:
            ret_feat = ret_feat[:feat_len]
            mask = mask[:feat_len]
        cur_len = len(ret_feat)
        if cur_len < feat_len:
            residual = feat_len - cur_len
            append_data = np.zeros((residual, ret_feat.shape[1]))
            ret_feat = np.concatenate([ret_feat, append_data])
            mask = np.concatenate([mask, np.zeros(residual)])
        # DeepGEM forward expects NestedTensor: (tensors, mask)
        # mask: True = valid
        return (torch.from_numpy(ret_feat).float(),
                torch.from_numpy(mask).bool(),
                {'label': torch.tensor(label).long().reshape(-1).repeat(feat_len),
                 'pid': image_id})


def _wrap_as_nestedtensor(feat_b, mask_b):
    """Create a dummy object that mimics NestedTensor (.tensors, .mask)."""
    class NT:
        pass
    nt = NT()
    nt.tensors = feat_b
    nt.mask = mask_b
    return nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'))
    ap.add_argument('--labels', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'))
    ap.add_argument('--combined-feat', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_combined_feat'))
    ap.add_argument('--internal-pickle', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_internal.pickle'))
    ap.add_argument('--checkpoint', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\DeepGEM\checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_EGFR.pickle'))
    ap.add_argument('--out', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\egfr_pred.csv'))
    args = ap.parse_args()

    # ---- Build combined_feat + internal.pickle ----
    df = pd.read_csv(args.labels)
    label_map = dict(zip(df['case_id'], df['EGFR'].astype(int)))
    case_ids = sorted(p.name for p in args.feat.iterdir() if p.is_dir())
    print(f'[INFO] {len(case_ids)} cases from {args.feat}')

    args.combined_feat.mkdir(parents=True, exist_ok=True)
    pids, labels = [], []
    for case_id in case_ids:
        per_patch = []
        for pkl in sorted((args.feat / case_id).rglob('*.pkl')):
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, list):
                per_patch.extend(d)
            else:
                per_patch.append(d)
        if not per_patch:
            continue
        if case_id not in label_map:
            continue
        out = args.combined_feat / f'{case_id}.pkl'
        with open(out, 'wb') as f:
            pickle.dump(per_patch, f)
        pids.append(case_id)
        labels.append(int(label_map[case_id]))
        print(f'  {case_id}: {len(per_patch)} patches  label={labels[-1]}')

    if not pids:
        sys.exit('[FATAL] no cases with both features and labels')
    payload = {'ExcisionalBiopsy': {'EGFR': {'test': {'pid': pids, 'label': labels}}}}
    with open(args.internal_pickle, 'wb') as f:
        pickle.dump(payload, f)
    print(f'\n[OK] {len(pids)} cases  pos={sum(labels)}  neg={len(labels)-sum(labels)}')

    # ---- Load checkpoint with monkey-patched CPU storage loader ----
    with open(args.checkpoint, 'rb') as f:
        ckpt = pickle.load(f)
    parameter = ckpt['parameter']
    state_dict = ckpt['checkpoint']
    print(f'[INFO] parameter: {parameter}')
    print(f'[INFO] # state tensors: {len(state_dict)}')

    # ---- Build DeepGEM model directly ----
    from models.model_deepgem import DeepGEM as _DeepGEM
    model = _DeepGEM(
        num_classes=2,
        patch_dim=768,
        dim=parameter['hidden_dim'],
        depth=parameter['depth'],
    ).to('cpu').eval()

    # Strip 'module.' prefix and load
    new_state = {}
    for k, v in state_dict.items():
        if k.endswith('total_params') or k.endswith('total_ops'):
            continue
        new_state[k.replace('module.', '')] = v
    miss, unex = model.load_state_dict(new_state, strict=False)
    if miss or unex:
        print(f'[WARN] ckpt miss={len(miss)} unex={len(unex)}')
        if miss: print(f'  miss sample: {miss[:5]}')
        if unex: print(f'  unex sample: {unex[:5]}')
    else:
        print('[OK] DeepGEM model loaded: missing=0 unexpected=0 [OK]')

    # ---- Forward ----
    info = payload['ExcisionalBiopsy']['EGFR']['test']
    info['feat_fp'] = [str(args.combined_feat / f'{pid}.pkl') for pid in pids]
    ds = DeepGEMDataset(info, feature_len=parameter['feature_len'])
    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            feat, mask, target = ds[idx]
            pid = target['pid']
            label = int(target['label'][0].item())
            f_b = feat.unsqueeze(0)  # [1, 500, 768]
            m_b = mask.unsqueeze(0)  # [1, 500]
            nt = NestedTensor(f_b, m_b)
            logits = model(nt)
            print(f'  [debug] {pid}  logits raw={logits.detach().cpu().numpy() if hasattr(logits, "detach") else "?"}')
            if isinstance(logits, tuple):
                logits = logits[0]
                print(f'  [debug] {pid}  tuple first={logits.detach().cpu().numpy()}')
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            print(f'  [debug] {pid}  softmax={probs}')
            rows.append({'case_id': pid, 'true_label': label,
                         'prob_negative': float(probs[0]),
                         'prob_positive': float(probs[1])})
            print(f'  {pid:18s}  label={label}  P(pos)={probs[1]:.4f}')
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f'\n[OK] → {args.out}')

    pos_df = out_df[out_df.true_label == 1]
    neg_df = out_df[out_df.true_label == 0]
    print(f'\n=== SUMMARY ===')
    print(f'EGFR+ cases (n={len(pos_df)})  mean P(positive) = {pos_df.prob_positive.mean() if len(pos_df) else float("nan"):.4f}')
    print(f'EGFR- cases (n={len(neg_df)})  mean P(positive) = {neg_df.prob_positive.mean() if len(neg_df) else float("nan"):.4f}')
    if len(pos_df) and len(neg_df):
        if pos_df.prob_positive.mean() > neg_df.prob_positive.mean():
            print('[PASS] sanity OK: positive mean > negative mean')
        else:
            print('[FAIL] sanity FAIL')
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(out_df.true_label, out_df.prob_positive)
            print(f'AUC = {auc:.3f}  (N={len(out_df)} limited)')
        except Exception as e:
            print(f'AUC skipped: {e}')


if __name__ == '__main__':
    main()
