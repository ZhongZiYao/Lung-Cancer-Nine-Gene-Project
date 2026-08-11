# -*- coding: utf-8 -*-
"""
step10_combined_classes.py — Merge rare genes into combined classes

Strategy:
1. Original 9-gene classification (baseline)
2. Combined "any_rare" (NRAS | ERBB2 | ALK | ROS1 | PIK3CA) - rare driver
3. 3-class: EGFR vs KRAS vs rare_combined
4. 5-fold CV + 3-seed ensemble (same as step9)
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
RARE_GENES = ['NRAS', 'ERBB2', 'ALK', 'ROS1', 'PIK3CA']  # 5 rare drivers

TITAN_PKL = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\TITAN_model\TCGA_TITAN_features.pkl'
LABELS_CSV = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter9_combined_classes')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]


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
    case_emb = {cid: data['embeddings'][idxs].mean(axis=0) for cid, idxs in case_to_indices.items()}
    return case_emb


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=128, num_outputs=1, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, x):
        return self.net(x)


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


def train_one_model(X_train, y_train, X_val, y_val, hidden, lr, epochs, patience,
                     pos_w_max, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_X = torch.from_numpy(X_train).float().to(device)
    val_X = torch.from_numpy(X_val).float().to(device)
    train_y = torch.from_numpy(y_train).float().to(device)
    val_y = torch.from_numpy(y_val).float().to(device)

    n_pos = int(train_y.sum().item())
    n_neg = len(X_train) - n_pos
    pos_w = min(n_neg / max(n_pos, 1), pos_w_max) if n_pos > 0 else 1.0
    pos_w_t = torch.tensor([pos_w]).to(device)

    model = SimpleMLP(in_dim=X_train.shape[1], hidden=hidden, num_outputs=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(train_X).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            val_p = model(val_X).sigmoid().squeeze(-1).cpu().numpy()
        try:
            auc = roc_auc_score(val_y.cpu().numpy(), val_p)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
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
        val_p = model(val_X).sigmoid().squeeze(-1).cpu().numpy()
    return val_p, best_auc


def run_strategy(name, y_labels, X, n_folds=5, seeds=SEEDS):
    """Run one classification strategy"""
    flush_print(f'\n========== Strategy: {name} (pos={int(y_labels.sum())}/{len(y_labels)}) ==========')
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n_pos = int(y_labels.sum())
    if n_pos < 5:
        flush_print('  SKIP too few positives')
        return {'mean': {'auc': 0.5}, 'std': {}, 'fold_metrics': []}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics_list = []
    oof_pred = np.zeros(len(y_labels))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labels)):
        X_train, y_train = X[train_idx], y_labels[train_idx]
        X_val, y_val = X[val_idx], y_labels[val_idx]
        seed_preds = []
        for seed in seeds:
            pred, _ = train_one_model(
                X_train, y_train, X_val, y_val,
                hidden=128, lr=1e-3, epochs=50, patience=10,
                pos_w_max=10.0, device=torch.device('cpu'), seed=seed,
            )
            seed_preds.append(pred)
        val_p_ensemble = np.mean(seed_preds, axis=0)
        oof_pred[val_idx] = val_p_ensemble
        metrics = best_threshold_metrics(y_val, val_p_ensemble)
        metrics['auc'] = float(roc_auc_score(y_val, val_p_ensemble))
        flush_print(f'  f{fold}: AUC={metrics["auc"]:.3f} BalAcc={metrics["balanced_acc"]:.3f} F1={metrics["f1"]:.3f}')
        fold_metrics_list.append(metrics)

    keys = ['auc', 'acc', 'balanced_acc', 'precision', 'recall', 'specificity', 'f1']
    mean = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics_list])) for k in keys}
    flush_print(f'  >>> {name}: AUC={mean["auc"]:.3f}±{std["auc"]:.3f} BalAcc={mean["balanced_acc"]:.3f} F1={mean["f1"]:.3f}')
    return {'mean': mean, 'std': std, 'fold_metrics': fold_metrics_list, 'oof_pred': oof_pred}


def main():
    flush_print('Loading TITAN embeddings...')
    case_emb = load_titan_embeddings(TITAN_PKL)
    df = pd.read_csv(LABELS_CSV)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}

    valid_cases = sorted([c for c in case_emb if c in labels and
                          all(labels[c][g] in (0, 1) for g in GENES)])
    flush_print(f'Cases: {len(valid_cases)}')
    X = np.stack([case_emb[c] for c in valid_cases])
    Y = np.stack([[labels[c][g] for g in GENES] for c in valid_cases])

    results = {}

    # Strategy 1: 9 individual genes (baseline, like step9)
    flush_print('\n=== Strategy 1: 9 individual genes ===')
    for gene in GENES:
        g_idx = GENES.index(gene)
        y = Y[:, g_idx]
        r = run_strategy(gene, y, X)
        results[gene] = r

    # Strategy 2: any_rare (NRAS | ERBB2 | ALK | ROS1 | PIK3CA)
    flush_print('\n=== Strategy 2: Combined rare classes ===')
    y_any_rare = (Y[:, GENES.index('NRAS')] | Y[:, GENES.index('ERBB2')] |
                  Y[:, GENES.index('ALK')] | Y[:, GENES.index('ROS1')] |
                  Y[:, GENES.index('PIK3CA')])
    r = run_strategy('any_rare', y_any_rare, X)
    results['any_rare'] = r

    # Strategy 3: 3-class binary (EGFR vs KRAS vs others)
    flush_print('\n=== Strategy 3: 3-class binary (EGFR/KRAS/rare_or_other) ===')
    # Multiclass via one-vs-rest encoded as 3 binary tasks
    # (We'll just train 3 separate binary classifiers)
    y_egfr = Y[:, GENES.index('EGFR')]
    y_kras = Y[:, GENES.index('KRAS')]
    # For "rare" we use any_rare excluding EGFR/KRAS positives
    y_rare_only = ((y_any_rare == 1) & (y_egfr == 0) & (y_kras == 0)).astype(int)
    r = run_strategy('rare_only', y_rare_only, X)
    results['rare_only'] = r

    # Final summary
    flush_print('\n' + '=' * 90)
    flush_print('Final comparison: 9 genes + combined classes')
    flush_print('=' * 90)
    flush_print('%-15s %-12s %-10s %-10s %-10s %-10s %-10s' %
                ('Strategy', 'AUC', 'Acc', 'BalAcc', 'Precision', 'Recall', 'F1'))
    sorted_res = sorted(results.items(), key=lambda x: -x[1]['mean'].get('auc', 0))
    for name, r in sorted_res:
        m = r['mean']
        std = r['std']
        if m.get('auc', 0.5) == 0.5:
            flush_print('%-15s (skip)' % name)
            continue
        flush_print('%-15s %-12s %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f' % (
            name,
            '%.3f±%.3f' % (m['auc'], std['auc']),
            m['acc'],
            m['balanced_acc'],
            m['precision'],
            m['recall'],
            m['f1'],
        ))

    # Save summary
    summary_data = {name: {'mean': r['mean'], 'std': r['std']} for name, r in results.items()}
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    items_sorted = [(name, r['mean'].get('auc', 0), r['std'].get('auc', 0)) for name, r in results.items()]
    items_sorted.sort(key=lambda x: -x[1])
    names = [x[0] for x in items_sorted]
    aucs = [x[1] for x in items_sorted]
    stds = [x[2] for x in items_sorted]
    colors = ['#2ecc71' if a >= 0.85 else '#27ae60' if a >= 0.75 else '#3498db' if a >= 0.65 else '#e74c3c' for a in aucs]
    bars = ax.bar(names, aucs, yerr=stds, color=colors, capsize=5)
    ax.axhline(0.5, color='gray', linestyle='--', label='Random')
    ax.axhline(0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    ax.set_ylabel('AUC')
    ax.set_title('TITAN 9-gene + Combined classes (5-fold CV + 3-seed ensemble)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{a:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'combined_classes_auc.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('AUC bar chart saved to %s' % str(FIG_DIR / 'combined_classes_auc.png'))


if __name__ == '__main__':
    main()