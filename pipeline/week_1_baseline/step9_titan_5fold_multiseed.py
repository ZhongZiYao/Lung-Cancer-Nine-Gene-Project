# -*- coding: utf-8 -*-
"""
step9_titan_5fold_multiseed.py — TITAN slide embedding with 5-fold CV + multi-seed ensemble

Improvements over step7/step8:
1. 5-fold StratifiedKFold (vs 3-fold)
2. Multi-seed ensemble (3 seeds per fold)
3. Probability averaging across 15 models per gene
4. Best-F1 threshold + balanced_accuracy + precision/recall
5. PR/ROC curves per gene
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
LABELS_CSV = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter8_titan_5fold_multiseed')
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
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    return {
        'threshold': float(best_thr),
        'acc': float(accuracy_score(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(sensitivity),
        'specificity': float(specificity),
        'f1': float(best_f1),
    }


def train_one_model(X_train, Y_train, X_val, Y_val, gene_idx, hidden, lr, epochs,
                     patience, pos_w_max, device, seed):
    """Train one MLP, return oof prob for val"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_X = torch.from_numpy(X_train).float().to(device)
    val_X = torch.from_numpy(X_val).float().to(device)
    train_Y = torch.from_numpy(Y_train).float().to(device)
    val_Y = torch.from_numpy(Y_val).float().to(device)

    n_pos = int(train_Y[:, gene_idx].sum().item())
    n_neg = len(X_train) - n_pos
    pos_w = min(n_neg / max(n_pos, 1), pos_w_max) if n_pos > 0 else 1.0
    pos_w_t = torch.tensor([pos_w]).to(device)

    model = SimpleMLP(in_dim=X_train.shape[1], hidden=hidden, num_genes=9).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_X), train_Y, pos_weight=pos_w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            val_p = model(val_X).sigmoid().cpu().numpy()[:, gene_idx]
        try:
            auc = roc_auc_score(val_Y.cpu().numpy()[:, gene_idx], val_p)
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
        val_p = model(val_X).sigmoid().cpu().numpy()[:, gene_idx]
    return val_p, best_auc


def main():
    device = torch.device('cpu')
    flush_print('[TITAN-5fold-multiseed] loading embeddings...')
    case_emb = load_titan_embeddings(TITAN_PKL)
    df = pd.read_csv(LABELS_CSV)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}

    valid_cases = sorted([c for c in case_emb if c in labels and
                          all(labels[c][g] in (0, 1) for g in GENES)])
    flush_print(f'[TITAN] {len(valid_cases)} cases')
    X = np.stack([case_emb[c] for c in valid_cases])
    Y = np.stack([[labels[c][g] for g in GENES] for c in valid_cases])

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

    n_folds = 5
    all_results = {}
    oof_predictions = {g: np.zeros(len(valid_cases)) for g in GENES}
    oof_labels = {g: Y[:, GENES.index(g)] for g in GENES}

    for gene in GENES:
        g_idx = GENES.index(gene)
        y = Y[:, g_idx]
        n_pos = int(y.sum())
        flush_print(f'\n=== {gene} (pos={n_pos}/{len(y)}) ===')
        if n_pos < 5:
            flush_print('  SKIP')
            all_results[gene] = {'mean': {'auc': 0.5}, 'std': {}, 'fold_metrics': []}
            continue

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics_list = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            # Multi-seed ensemble
            seed_preds = []
            seed_aucs = []
            for seed in SEEDS:
                pred, auc_val = train_one_model(
                    X_train, Y_train, X_val, Y_val, g_idx,
                    hidden=128, lr=1e-3, epochs=50, patience=10,
                    pos_w_max=10.0, device=device, seed=seed,
                )
                seed_preds.append(pred)
                seed_aucs.append(auc_val)
            # Average predictions across seeds
            val_p_ensemble = np.mean(seed_preds, axis=0)
            y_true = Y_val[:, g_idx]
            # Metrics
            metrics = best_threshold_metrics(y_true, val_p_ensemble)
            metrics['auc'] = float(roc_auc_score(y_true, val_p_ensemble))
            mean_seed_auc = float(np.mean(seed_aucs))
            flush_print(f'  f{fold}: AUC={metrics["auc"]:.3f} (seeds={mean_seed_auc:.3f}) BalAcc={metrics["balanced_acc"]:.3f} P={metrics["precision"]:.3f} R={metrics["recall"]:.3f} F1={metrics["f1"]:.3f} thr={metrics["threshold"]:.2f}')
            fold_metrics_list.append(metrics)
            # Store OOF
            for i, vi in enumerate(val_idx):
                oof_predictions[gene][vi] = val_p_ensemble[i]

        # Aggregate
        keys = ['auc', 'acc', 'balanced_acc', 'precision', 'recall', 'specificity', 'f1']
        mean = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
        std = {k: float(np.std([m[k] for m in fold_metrics_list])) for k in keys}
        flush_print(f'  >>> {gene}: AUC={mean["auc"]:.3f}±{std["auc"]:.3f} BalAcc={mean["balanced_acc"]:.3f} F1={mean["f1"]:.3f}')
        all_results[gene] = {'mean': mean, 'std': std, 'fold_metrics': fold_metrics_list}

    # Final table
    flush_print('\n' + '=' * 90)
    flush_print('TITAN 9-gene final results (5-fold CV, multi-seed, sorted by AUC)')
    flush_print('=' * 90)
    flush_print('%-8s %-12s %-10s %-10s %-10s %-10s %-10s %-10s' %
                ('Gene', 'AUC', 'Acc', 'BalAcc', 'Precision', 'Recall', 'Specificity', 'F1'))
    sorted_res = sorted(all_results.items(), key=lambda x: -x[1]['mean'].get('auc', 0))
    for gene, res in sorted_res:
        m = res['mean']
        std = res['std']
        if 'auc' not in m or m['auc'] == 0.5:
            flush_print('%-8s %s' % (gene, '(skip)'))
            continue
        flush_print('%-8s %-12s %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f' % (
            gene,
            '%.3f±%.3f' % (m['auc'], std['auc']),
            m['acc'],
            m['balanced_acc'],
            m['precision'],
            m['recall'],
            m['specificity'],
            m['f1'],
        ))

    # Save summary
    summary_data = {g: {'mean': r['mean'], 'std': r['std']} for g, r in all_results.items()}
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))

    # Save OOF predictions
    np.savez(OUT_DIR / 'oof_predictions.npz',
             oof_predictions=oof_predictions, oof_labels=oof_labels,
             valid_cases=np.array(valid_cases))

    # Plot ROC curves
    n_genes = len(GENES)
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.array(axes).flatten()
    for i, gene in enumerate(GENES):
        ax = axes[i]
        if gene not in oof_predictions:
            ax.set_title(f'{gene} (skipped)')
            continue
        y_true = oof_labels[gene]
        y_pred = oof_predictions[gene]
        if y_pred.max() > y_pred.min():
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        else:
            y_pred_norm = y_pred
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_norm)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{gene} (pos={int(y_true.sum())})')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    for j in range(n_genes, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'roc_curves_5fold.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('ROC curves saved to %s' % str(FIG_DIR / 'roc_curves_5fold.png'))

    # Plot PR curves
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.array(axes).flatten()
    for i, gene in enumerate(GENES):
        ax = axes[i]
        if gene not in oof_predictions:
            ax.set_title(f'{gene} (skipped)')
            continue
        y_true = oof_labels[gene]
        y_pred = oof_predictions[gene]
        if y_pred.max() > y_pred.min():
            y_pred_norm = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        else:
            y_pred_norm = y_pred
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_pred_norm)
            ap = average_precision_score(y_true, y_pred_norm)
            ax.plot(rec, prec, color='blue', lw=2, label=f'AP={ap:.3f}')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{gene} (pos={int(y_true.sum())})')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
    for j in range(n_genes, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pr_curves_5fold.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('PR curves saved to %s' % str(FIG_DIR / 'pr_curves_5fold.png'))

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    genes_sorted = sorted(all_results.keys(), key=lambda g: -all_results[g]['mean'].get('auc', 0))
    aucs = [all_results[g]['mean'].get('auc', 0) for g in genes_sorted]
    stds = [all_results[g]['std'].get('auc', 0) for g in genes_sorted]
    colors = ['#2ecc71' if a >= 0.85 else '#27ae60' if a >= 0.75 else '#3498db' if a >= 0.65 else '#e74c3c' for a in aucs]
    bars = ax.bar(genes_sorted, aucs, yerr=stds, color=colors, capsize=5)
    ax.axhline(0.5, color='gray', linestyle='--', label='Random (0.5)')
    ax.axhline(0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    ax.set_ylabel('AUC')
    ax.set_title('TITAN 9-gene AUC (5-fold CV + 3-seed ensemble, mean ± std)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{a:.3f}', ha='center', fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'auc_bar_5fold.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('AUC bar chart saved to %s' % str(FIG_DIR / 'auc_bar_5fold.png'))


if __name__ == '__main__':
    main()