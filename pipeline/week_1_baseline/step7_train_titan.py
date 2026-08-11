# -*- coding: utf-8 -*-
"""
step7_train_titan.py — Train 9-gene classifier on TITAN pre-extracted slide embeddings

Uses TITAN pre-extracted 768-dim TCGA slide embeddings from TCGA_TITAN_features.pkl
Mean-pool multiple slides per case. Train simple MLP for each gene.
"""
import json
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']
TITAN_PKL = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\TITAN_model\TCGA_TITAN_features.pkl'
LABELS_CSV = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter6_titan')


def flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def load_titan_embeddings(pkl_path):
    """Load TITAN embeddings, aggregate multiple slides per case via mean pooling"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    filenames = data['filenames']
    case_to_indices = defaultdict(list)
    for i, fn in enumerate(filenames):
        m = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', fn)
        if m:
            case_to_indices[m.group(0)].append(i)
    # Mean pool
    case_emb = {cid: embeddings[idxs].mean(axis=0) for cid, idxs in case_to_indices.items()}
    return case_emb


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}
    return labels


def compute_pos_weight(cases, labels, gene, max_w=10.0):
    n_pos = sum(1 for c in cases if labels[c][gene] == 1)
    n_neg = len(cases) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return min(n_neg / max(n_pos, 1), max_w)


class SimpleMLP(nn.Module):
    """Simple MLP for slide-level classification"""
    def __init__(self, in_dim=768, hidden=128, num_genes=9, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_genes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_fold(train_cases, val_cases, X, Y, gene, hidden, lr, epochs, patience,
                    pos_w_max, device, seed):
    """Single fold training"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build X, Y tensors from cases
    train_X = torch.from_numpy(np.stack([X[i] for i in train_cases])).float().to(device)
    val_X = torch.from_numpy(np.stack([X[i] for i in val_cases])).float().to(device)
    train_Y = torch.from_numpy(np.stack([Y[i] for i in train_cases])).float().to(device)
    val_Y = torch.from_numpy(np.stack([Y[i] for i in val_cases])).float().to(device)

    g_idx = GENES.index(gene)
    n_pos = int(train_Y[:, g_idx].sum().item())
    n_neg = len(train_cases) - n_pos
    pos_w = min(n_neg / max(n_pos, 1), pos_w_max) if n_pos > 0 else 1.0
    pos_w_tensor = torch.tensor([pos_w], dtype=torch.float32).to(device)

    model = SimpleMLP(in_dim=X.shape[1], hidden=hidden, num_genes=9).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(train_X)
        loss = F.binary_cross_entropy_with_logits(logits, train_Y, pos_weight=pos_w_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_preds = val_logits.sigmoid().cpu().numpy()
        try:
            auc = roc_auc_score(val_Y.cpu().numpy()[:, g_idx], val_preds[:, g_idx])
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
        if ep % 5 == 0 or ep == epochs:
            flush_print('    ep%d: loss=%.3f val_auc=%.3f best=%.3f' % (ep, loss.item(), auc, best_auc))
        if no_improve >= patience:
            flush_print('    early stop at ep%d' % ep)
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(val_X).sigmoid().cpu().numpy()
    y_true = val_Y.cpu().numpy()[:, g_idx]
    try:
        final_auc = roc_auc_score(y_true, val_preds[:, g_idx])
    except Exception:
        final_auc = 0.5
    y_pred = (val_preds[:, g_idx] > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'auc': final_auc, 'acc': acc, 'f1': f1, 'best_epoch': best_epoch, 'pos_w': pos_w}


def main():
    device = torch.device('cpu')
    flush_print('[TITAN] loading pre-extracted embeddings...')
    case_emb = load_titan_embeddings(TITAN_PKL)
    labels = load_labels(LABELS_CSV)

    # Build dataset: only cases with both TITAN features AND 9-gene labels
    valid_cases = sorted(set(case_emb.keys()) & set(labels.keys()))
    n_with_labels = sum(1 for c in valid_cases if all(labels[c][g] in (0, 1) for g in GENES))
    valid_cases = [c for c in valid_cases if all(labels[c][g] in (0, 1) for g in GENES)]
    flush_print(f'[TITAN] {len(valid_cases)} cases have both features and valid labels')

    # Build X (N, 768), Y (N, 9)
    X = np.stack([case_emb[c] for c in valid_cases])
    Y = np.stack([[labels[c][g] for g in GENES] for c in valid_cases])
    flush_print(f'[TITAN] X.shape={X.shape}, Y.shape={Y.shape}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3-fold CV per gene
    from sklearn.model_selection import StratifiedKFold
    all_results = {}
    for gene in GENES:
        g_idx = GENES.index(gene)
        y = Y[:, g_idx]
        n_pos = int(y.sum())
        flush_print(f'\n=== {gene} (pos={n_pos}/{len(y)}) ===')
        if n_pos < 5:
            flush_print(f'  SKIP too few positives')
            all_results[gene] = {'mean': {'auc': 0.5, 'acc': 0, 'f1': 0}, 'std': {'auc': 0, 'acc': 0, 'f1': 0}}
            continue

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            flush_print(f'  Fold {fold+1}/3: train={len(train_idx)} val={len(val_idx)}')
            r = train_one_fold(train_idx.tolist(), val_idx.tolist(), X, Y, gene,
                               hidden=128, lr=1e-3, epochs=30, patience=8,
                               pos_w_max=10.0, device=device, seed=42 + fold)
            flush_print(f'  [f{fold}] AUC={r["auc"]:.3f} ACC={r["acc"]:.3f} F1={r["f1"]:.3f} @ep{r["best_epoch"]}')
            fold_results.append(r)
        mean = {k: float(np.mean([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
        std = {k: float(np.std([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
        flush_print(f'  >>> {gene}: AUC={mean["auc"]:.3f}±{std["auc"]:.3f} ACC={mean["acc"]:.3f} F1={mean["f1"]:.3f}')
        all_results[gene] = {'mean': mean, 'std': std}

    # Summary
    flush_print('\n' + '=' * 70)
    flush_print('TITAN slide embedding 9-gene results')
    flush_print('=' * 70)
    flush_print('%-8s %-14s %-8s %-8s' % ('Gene', 'AUC', 'ACC', 'F1'))
    sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['mean']['auc'])
    for gene, res in sorted_results:
        m = res['mean']
        s = res['std']
        auc_str = '%.3f+-%.3f' % (m['auc'], s['auc'])
        flush_print('%-8s %-14s %-8.3f %-8.3f' % (gene, auc_str, m['acc'], m['f1']))

    # Save
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump({g: {'mean': r['mean'], 'std': r['std']} for g, r in all_results.items()}, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))


if __name__ == '__main__':
    main()