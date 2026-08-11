# -*- coding: utf-8 -*-
"""
step11_feature_fusion.py — TITAN + UNI feature fusion

Combines:
- TITAN slide embedding (768-dim): global slide-level features
- UNI patch features (1024-dim): mean-pooled patch-level features

Total: 1792-dim per case
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
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter10_feature_fusion')
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


def load_uni_features(uni_dir):
    """Load UNI patch features, mean pool to 1024-dim per case"""
    uni = {}
    for p in Path(uni_dir).glob('*.npz'):
        z = np.load(p, allow_pickle=True)
        feats = z['features']
        uni[p.stem] = feats.mean(axis=0)
    return uni


class FusionMLP(nn.Module):
    def __init__(self, in_dim=1792, hidden=256, num_outputs=1, dropout=0.5):
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
        'n_pos': int(y_true.sum()),
    }


def train_one_model(X_train, y_train, X_val, y_val, in_dim, hidden, lr, epochs,
                     patience, pos_w_max, device, seed):
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

    model = FusionMLP(in_dim=in_dim, hidden=hidden, num_outputs=1).to(device)
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


def evaluate_strategy(X, Y, gene, strategy_name, hidden=256, lr=1e-3, n_folds=5, seeds=SEEDS):
    """Run 5-fold CV + multi-seed ensemble for one gene"""
    g_idx = GENES.index(gene)
    y = Y[:, g_idx]
    n_pos = int(y.sum())
    flush_print(f'\n[{strategy_name}] {gene} (pos={n_pos})')
    if n_pos < 5:
        flush_print('  SKIP')
        return {'mean': {'auc': 0.5}, 'std': {}, 'fold_metrics': []}

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        seed_preds = []
        for seed in seeds:
            pred, _ = train_one_model(
                X_train, y_train, X_val, y_val,
                in_dim=X.shape[1],
                hidden=hidden, lr=lr, epochs=50, patience=10,
                pos_w_max=10.0, device=torch.device('cpu'), seed=seed,
            )
            seed_preds.append(pred)
        val_p_ensemble = np.mean(seed_preds, axis=0)
        metrics = best_threshold_metrics(y_val, val_p_ensemble)
        metrics['auc'] = float(roc_auc_score(y_val, val_p_ensemble))
        flush_print(f'  f{fold}: AUC={metrics["auc"]:.3f} BalAcc={metrics["balanced_acc"]:.3f} F1={metrics["f1"]:.3f}')
        fold_metrics_list.append(metrics)

    keys = ['auc', 'acc', 'balanced_acc', 'precision', 'recall', 'specificity', 'f1']
    mean = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics_list])) for k in keys}
    flush_print(f'  >>> {gene}: AUC={mean["auc"]:.3f}±{std["auc"]:.3f} BalAcc={mean["balanced_acc"]:.3f} F1={mean["f1"]:.3f}')
    return {'mean': mean, 'std': std, 'fold_metrics': fold_metrics_list}


def main():
    flush_print('Loading TITAN + UNI features...')
    case_emb_titan = load_titan_embeddings(TITAN_PKL)
    case_emb_uni = load_uni_features(UNI_DIR)
    flush_print(f'TITAN cases: {len(case_emb_titan)}')
    flush_print(f'UNI cases: {len(case_emb_uni)}')

    df = pd.read_csv(LABELS_CSV)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}

    # Cases that have BOTH TITAN and UNI features and labels
    valid_cases = sorted([c for c in case_emb_titan if c in case_emb_uni and c in labels and
                          all(labels[c][g] in (0, 1) for g in GENES)])
    flush_print(f'Cases with both features: {len(valid_cases)}')

    # Build features
    X_titan = np.stack([case_emb_titan[c] for c in valid_cases])
    X_uni = np.stack([case_emb_uni[c] for c in valid_cases])
    X_fused = np.concatenate([X_titan, X_uni], axis=1)
    Y = np.stack([[labels[c][g] for g in GENES] for c in valid_cases])
    flush_print(f'X_titan: {X_titan.shape}, X_uni: {X_uni.shape}, X_fused: {X_fused.shape}')

    # Normalize features (z-score)
    from sklearn.preprocessing import StandardScaler
    scaler_titan = StandardScaler().fit(X_titan)
    X_titan_norm = scaler_titan.transform(X_titan)
    scaler_uni = StandardScaler().fit(X_uni)
    X_uni_norm = scaler_uni.transform(X_uni)
    X_fused_norm = np.concatenate([X_titan_norm, X_uni_norm], axis=1)
    flush_print(f'After normalization: TITAN {X_titan_norm.shape}, UNI {X_uni_norm.shape}, Fused {X_fused_norm.shape}')

    results = {}

    # Strategy 1: TITAN only (baseline)
    flush_print('\n\n========== Strategy 1: TITAN only (baseline) ==========')
    for gene in GENES:
        r = evaluate_strategy(X_titan_norm, Y, gene, 'TITAN', hidden=128)
        results[f'TITAN_{gene}'] = r

    # Strategy 2: UNI only
    flush_print('\n\n========== Strategy 2: UNI only ==========')
    for gene in GENES:
        r = evaluate_strategy(X_uni_norm, Y, gene, 'UNI', hidden=256)
        results[f'UNI_{gene}'] = r

    # Strategy 3: Fused (TITAN + UNI)
    flush_print('\n\n========== Strategy 3: Fused TITAN + UNI ==========')
    for gene in GENES:
        r = evaluate_strategy(X_fused_norm, Y, gene, 'Fused', hidden=256)
        results[f'Fused_{gene}'] = r

    # Final comparison
    flush_print('\n\n' + '=' * 100)
    flush_print('Final comparison: TITAN vs UNI vs Fused')
    flush_print('=' * 100)
    flush_print('%-15s %-12s %-12s %-12s %-12s %-12s %-10s' % ('Gene', 'TITAN', 'UNI', 'Fused', 'Best', 'Diff', 'Gene'))
    rows = []
    for gene in GENES:
        t = results[f'TITAN_{gene}']['mean'].get('auc', 0.5)
        u = results[f'UNI_{gene}']['mean'].get('auc', 0.5)
        f = results[f'Fused_{gene}']['mean'].get('auc', 0.5)
        if max(t, u, f) == 0.5:
            continue
        best = max(t, u, f)
        diff = f - t
        flag = '🎯' if f > t and f > u else ''
        rows.append((gene, t, u, f, best, diff, flag))

    rows.sort(key=lambda x: -x[4])
    for gene, t, u, f, best, diff, flag in rows:
        flush_print('%-15s %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f %-10s' %
                    (gene, t, u, f, best, diff, flag))

    # Save summary
    summary_data = {name: {'mean': r['mean'], 'std': r['std']} for name, r in results.items()}
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))

    # Bar chart comparing strategies per gene
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(rows))
    width = 0.27
    titan_vals = [r[1] for r in rows]
    uni_vals = [r[2] for r in rows]
    fused_vals = [r[3] for r in rows]
    genes_list = [r[0] for r in rows]
    ax.bar(x - width, titan_vals, width, label='TITAN', color='#3498db')
    ax.bar(x, uni_vals, width, label='UNI', color='#e67e22')
    ax.bar(x + width, fused_vals, width, label='Fused (TITAN+UNI)', color='#2ecc71')
    ax.axhline(0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels(genes_list)
    ax.set_ylabel('AUC')
    ax.set_title('TITAN vs UNI vs Fused (5-fold + 3-seed)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fusion_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('Comparison chart saved to %s' % str(FIG_DIR / 'fusion_comparison.png'))


if __name__ == '__main__':
    main()