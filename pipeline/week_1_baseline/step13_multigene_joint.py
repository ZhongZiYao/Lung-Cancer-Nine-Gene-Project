# -*- coding: utf-8 -*-
"""
step13_multigene_joint.py — Multi-gene joint training

Strategy:
1. Single model with 9 outputs (sigmoid per gene)
2. Loss = sum of per-gene BCE losses (with per-gene pos_weight)
3. 5-fold CV + 3-seed ensemble
4. Compare with per-gene baseline
"""
import json
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']
TITAN_PKL = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\TITAN_model\TCGA_TITAN_features.pkl'
UNI_DIR = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'
LABELS_CSV = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter12_multigene_joint')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
NUM_GENES = 9


def flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def load_titan_embeddings(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    case_to_indices = defaultdict(list)
    for i, fn in enumerate(data['filenames']):
        m = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', fn)
        if m:
            case_to_indices[m.group(0)].append(i)
    return {cid: data['embeddings'][idxs].mean(axis=0) for cid, idxs in case_to_indices.items()}


def load_uni_features(uni_dir):
    uni = {}
    for p in Path(uni_dir).glob('*.npz'):
        z = np.load(p, allow_pickle=True)
        uni[p.stem] = z['features'].mean(axis=0)
    return uni


class MultiGeneFusionMLP(nn.Module):
    """
    Multi-task MLP: outputs 9 gene probabilities simultaneously
    Backbone: 1792 -> 512 -> 256 -> 128
    9 heads: 128 -> 1 each
    """
    def __init__(self, in_dim=1792, hidden_dims=[512, 256, 128], num_genes=9, dropout=0.5):
        super().__init__()
        # Shared backbone
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev),
                nn.Dropout(dropout),
                nn.Linear(prev, h),
                nn.GELU(),
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)
        # Per-gene heads
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(num_genes)])

    def forward(self, x):
        feats = self.backbone(x)
        return torch.cat([h(feats) for h in self.heads], dim=-1)  # (B, num_genes)


def best_threshold_metrics(y_true, y_proba):
    from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                                 precision_score, recall_score, confusion_matrix)
    best_f1 = -1
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95, 0.02):
        y_pred = (y_proba > thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    y_pred = (y_proba > best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(best_thr),
        'acc': float(accuracy_score(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(tp / max(tp + fn, 1)),
        'specificity': float(tn / max(tn + fp, 1)),
        'f1': float(best_f1),
    }


def compute_pos_weights(y_train, max_w=10.0):
    """Compute per-gene pos_weight"""
    pos_weights = []
    for g in range(y_train.shape[1]):
        n_pos = y_train[:, g].sum()
        n_neg = len(y_train) - n_pos
        if n_pos == 0:
            pos_weights.append(1.0)
        else:
            pos_weights.append(min(n_neg / max(n_pos, 1), max_w))
    return torch.tensor(pos_weights, dtype=torch.float32)


def train_joint(X_titan_tr, X_uni_tr, Y_tr, X_titan_va, X_uni_va, Y_va,
                 hidden_dims, lr, epochs, patience, max_pos_w, device, seed):
    """Joint training across all 9 genes"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_titan_tr = torch.from_numpy(X_titan_tr).float().to(device)
    X_uni_tr = torch.from_numpy(X_uni_tr).float().to(device)
    Y_tr = torch.from_numpy(Y_tr).float().to(device)
    X_titan_va = torch.from_numpy(X_titan_va).float().to(device)
    X_uni_va = torch.from_numpy(X_uni_va).float().to(device)
    Y_va = torch.from_numpy(Y_va).float().to(device)

    pos_w = compute_pos_weights(Y_tr.cpu().numpy(), max_w=max_pos_w).to(device)
    in_dim = X_titan_tr.shape[1] + X_uni_tr.shape[1]
    model = MultiGeneFusionMLP(in_dim=in_dim, hidden_dims=hidden_dims, num_genes=NUM_GENES).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Joint loss: mean of per-gene BCE
    def joint_loss(logits, targets):
        losses = []
        for g in range(NUM_GENES):
            losses.append(F.binary_cross_entropy_with_logits(
                logits[:, g], targets[:, g], pos_weight=pos_w[g]
            ))
        return torch.stack(losses).mean()

    best_mean_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(torch.cat([X_titan_tr, X_uni_tr], dim=-1))
        loss = joint_loss(logits, Y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.cat([X_titan_va, X_uni_va], dim=-1))
            val_p = val_logits.sigmoid().cpu().numpy()
        # Mean AUC across genes
        aucs = []
        for g in range(NUM_GENES):
            try:
                if Y_va[:, g].sum() >= 1 and Y_va[:, g].sum() < len(Y_va):
                    aucs.append(roc_auc_score(Y_va[:, g].cpu().numpy(), val_p[:, g]))
            except Exception:
                pass
        mean_auc = float(np.mean(aucs)) if aucs else 0.5
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = model(torch.cat([X_titan_va, X_uni_va], dim=-1)).sigmoid().cpu().numpy()
    return val_p, best_mean_auc


def evaluate_joint(X_titan, X_uni, Y, hidden_dims=[512, 256, 128], lr=1e-3,
                    n_folds=5, seeds=SEEDS):
    """Run 5-fold CV + multi-seed ensemble for multi-gene joint training"""
    flush_print(f'\n[MultiGene] hidden={hidden_dims} lr={lr}')

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    # Use TP53 as primary stratify (most balanced)
    y_strat = Y[:, GENES.index('TP53')]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = {g: [] for g in GENES}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_titan, y_strat)):
        X_titan_tr, X_uni_tr, Y_tr = X_titan[train_idx], X_uni[train_idx], Y[train_idx]
        X_titan_va, X_uni_va, Y_va = X_titan[val_idx], X_uni[val_idx], Y[val_idx]
        seed_preds = []
        for seed in seeds:
            pred, _ = train_joint(
                X_titan_tr, X_uni_tr, Y_tr, X_titan_va, X_uni_va, Y_va,
                hidden_dims=hidden_dims, lr=lr, epochs=50, patience=10,
                max_pos_w=10.0, device=torch.device('cpu'), seed=seed,
            )
            seed_preds.append(pred)
        val_p_ensemble = np.mean(seed_preds, axis=0)
        # Per-gene metrics
        for g in GENES:
            g_idx = GENES.index(g)
            y_true = Y_va[:, g_idx]
            if y_true.sum() < 1 or y_true.sum() == len(y_true):
                continue
            metrics = best_threshold_metrics(y_true, val_p_ensemble[:, g_idx])
            metrics['auc'] = float(roc_auc_score(y_true, val_p_ensemble[:, g_idx]))
            fold_metrics[g].append(metrics)
        flush_print(f'  f{fold}: done')

    # Aggregate
    results = {}
    for g in GENES:
        if not fold_metrics[g]:
            results[g] = {'mean': {'auc': 0.5}, 'std': {}, 'fold_metrics': []}
            continue
        keys = ['auc', 'acc', 'balanced_acc', 'precision', 'recall', 'specificity', 'f1']
        mean = {k: float(np.mean([m[k] for m in fold_metrics[g]])) for k in keys}
        std = {k: float(np.std([m[k] for m in fold_metrics[g]])) for k in keys}
        results[g] = {'mean': mean, 'std': std, 'fold_metrics': fold_metrics[g]}
        flush_print(f'  >>> {g}: AUC={mean["auc"]:.3f}+/-{std["auc"]:.3f} BalAcc={mean["balanced_acc"]:.3f} F1={mean["f1"]:.3f}')
    return results


def main():
    flush_print('Loading TITAN + UNI features...')
    case_emb_titan = load_titan_embeddings(TITAN_PKL)
    case_emb_uni = load_uni_features(UNI_DIR)
    df = pd.read_csv(LABELS_CSV)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}

    valid_cases = sorted([c for c in case_emb_titan if c in case_emb_uni and c in labels and
                          all(labels[c][g] in (0, 1) for g in GENES)])
    flush_print(f'Cases: {len(valid_cases)}')

    X_titan = np.stack([case_emb_titan[c] for c in valid_cases])
    X_uni = np.stack([case_emb_uni[c] for c in valid_cases])
    Y = np.stack([[labels[c][g] for g in GENES] for c in valid_cases])

    from sklearn.preprocessing import StandardScaler
    X_titan = StandardScaler().fit_transform(X_titan)
    X_uni = StandardScaler().fit_transform(X_uni)

    # Multi-gene joint training
    results = evaluate_joint(X_titan, X_uni, Y, hidden_dims=[512, 256, 128], lr=1e-3)

    # Save
    summary_data = {g: {'mean': r['mean'], 'std': r['std']} for g, r in results.items()}
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)

    # Final table
    flush_print('\n\n' + '=' * 100)
    flush_print('MULTI-GENE JOINT TRAINING (5-fold + 3-seed ensemble)')
    flush_print('=' * 100)
    flush_print('%-8s %-12s %-10s %-10s %-10s %-10s %-10s' %
                ('Gene', 'AUC', 'Acc', 'BalAcc', 'Precision', 'Recall', 'F1'))
    rows = []
    for g in GENES:
        m = results[g]['mean']
        if m.get('auc', 0.5) == 0.5:
            flush_print('%-8s (skip)' % g)
            continue
        flush_print('%-8s %-12s %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f' % (
            g,
            '%.3f+/-%.3f' % (m['auc'], results[g]['std']['auc']),
            m['acc'], m['balanced_acc'],
            m['precision'], m['recall'], m['f1']
        ))
        rows.append((g, m['auc'], results[g]['std']['auc']))

    # Comparison with previous best (from step12)
    flush_print('\n\n' + '=' * 100)
    flush_print('Comparison: Joint vs Separate (DeepConcat from step12)')
    flush_print('=' * 100)
    with open(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter11_deep_mlp_attn_fuse\summary.json') as f:
        prev = json.load(f)
    flush_print('%-8s %-12s %-12s %-10s' % ('Gene', 'Separate', 'Joint', 'Diff'))
    rows.sort(key=lambda x: -x[1])
    for gene, joint_auc, joint_std in rows:
        sep = prev.get(f'DeepConcat_{gene}', {}).get('mean', {}).get('auc', 0.5)
        diff = joint_auc - sep
        flag = ' JOINT' if diff > 0.01 else (' SEP' if diff < -0.01 else '')
        flush_print('%-8s %-12.3f %-12.3f %+.3f%s' % (gene, sep, joint_auc, diff, flag))


if __name__ == '__main__':
    main()