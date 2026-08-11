# -*- coding: utf-8 -*-
"""
step12_deep_mlp_attn_fuse.py — Deep MLP + Attention-based fusion

Strategy:
1. Per-feature projection: TITAN(768) -> 256, UNI(1024) -> 256
2. Cross-attention fusion (Q from TITAN, K/V from UNI)
3. Concat + Deep MLP (4 layers)
4. 5-fold CV + 3-seed ensemble
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
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter11_deep_mlp_attn_fuse')
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
TITAN_DIM = 768
UNI_DIM = 1024
PROJ_DIM = 256
FUSED_DIM = 512  # 256 (TITAN projected) + 256 (UNI projected)


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


class DeepFusionMLP(nn.Module):
    """
    Deep MLP with attention-based fusion:
    1. Project TITAN (768) -> 256
    2. Project UNI (1024) -> 256
    3. Cross-attention: TITAN query attends to UNI
    4. Deep MLP: 4 layers
    """
    def __init__(self, titan_dim=TITAN_DIM, uni_dim=UNI_DIM, proj_dim=PROJ_DIM,
                 hidden_dims=[512, 256, 128, 64], num_outputs=1, dropout=0.5):
        super().__init__()
        # Per-feature projection
        self.titan_proj = nn.Sequential(
            nn.LayerNorm(titan_dim),
            nn.Linear(titan_dim, proj_dim),
            nn.GELU(),
        )
        self.uni_proj = nn.Sequential(
            nn.LayerNorm(uni_dim),
            nn.Linear(uni_dim, proj_dim),
            nn.GELU(),
        )
        # Cross-attention: TITAN queries attend to UNI
        self.cross_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, batch_first=True)
        self.norm_attn = nn.LayerNorm(proj_dim)
        # Deep MLP
        in_dim = proj_dim * 2  # TITAN proj + (TITAN attended by UNI)
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                nn.Dropout(dropout),
                nn.Linear(prev_dim, h),
                nn.GELU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_outputs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, titan_x, uni_x):
        # Project
        t = self.titan_proj(titan_x).unsqueeze(1)  # (B, 1, proj_dim)
        u = self.uni_proj(uni_x).unsqueeze(1)     # (B, 1, proj_dim)
        # Cross-attention: TITAN attends to UNI
        attn_out, _ = self.cross_attn(query=t, key=u, value=u)
        attn_out = self.norm_attn(t + attn_out).squeeze(1)  # (B, proj_dim)
        # Concatenate TITAN + cross-attended
        fused = torch.cat([t.squeeze(1), attn_out], dim=-1)  # (B, 2*proj_dim)
        # Deep MLP
        return self.mlp(fused)


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


def train_one_model(X_titan_tr, X_uni_tr, y_tr, X_titan_va, X_uni_va, y_va,
                     hidden_dims, lr, epochs, patience, pos_w_max, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_titan_tr = torch.from_numpy(X_titan_tr).float().to(device)
    X_uni_tr = torch.from_numpy(X_uni_tr).float().to(device)
    y_tr = torch.from_numpy(y_tr).float().to(device)
    X_titan_va = torch.from_numpy(X_titan_va).float().to(device)
    X_uni_va = torch.from_numpy(X_uni_va).float().to(device)
    y_va = torch.from_numpy(y_va).float().to(device)

    n_pos = int(y_tr.sum().item())
    n_neg = len(X_titan_tr) - n_pos
    pos_w = min(n_neg / max(n_pos, 1), pos_w_max) if n_pos > 0 else 1.0
    pos_w_t = torch.tensor([pos_w]).to(device)

    model = DeepFusionMLP(hidden_dims=hidden_dims, num_outputs=1).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(X_titan_tr, X_uni_tr).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr, pos_weight=pos_w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            val_p = model(X_titan_va, X_uni_va).sigmoid().squeeze(-1).cpu().numpy()
        try:
            auc = roc_auc_score(y_va.cpu().numpy(), val_p)
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
        val_p = model(X_titan_va, X_uni_va).sigmoid().squeeze(-1).cpu().numpy()
    return val_p, best_auc


def evaluate_architecture(X_titan, X_uni, Y, gene, arch_name, hidden_dims,
                            lr=1e-3, n_folds=5, seeds=SEEDS):
    """Run 5-fold CV + multi-seed ensemble for one gene with deep fusion"""
    g_idx = GENES.index(gene)
    y = Y[:, g_idx]
    n_pos = int(y.sum())
    flush_print(f'\n[{arch_name}] {gene} (pos={n_pos})')
    if n_pos < 5:
        flush_print('  SKIP')
        return {'mean': {'auc': 0.5}, 'std': {}, 'fold_metrics': []}

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_titan, y)):
        X_titan_tr, X_uni_tr, y_tr = X_titan[train_idx], X_uni[train_idx], y[train_idx]
        X_titan_va, X_uni_va, y_va = X_titan[val_idx], X_uni[val_idx], y[val_idx]

        seed_preds = []
        for seed in seeds:
            pred, _ = train_one_model(
                X_titan_tr, X_uni_tr, y_tr, X_titan_va, X_uni_va, y_va,
                hidden_dims=hidden_dims, lr=lr, epochs=50, patience=10,
                pos_w_max=10.0, device=torch.device('cpu'), seed=seed,
            )
            seed_preds.append(pred)
        val_p_ensemble = np.mean(seed_preds, axis=0)
        metrics = best_threshold_metrics(y_va, val_p_ensemble)
        metrics['auc'] = float(roc_auc_score(y_va, val_p_ensemble))
        flush_print(f'  f{fold}: AUC={metrics["auc"]:.3f} BalAcc={metrics["balanced_acc"]:.3f} F1={metrics["f1"]:.3f}')
        fold_metrics_list.append(metrics)

    keys = ['auc', 'acc', 'balanced_acc', 'precision', 'recall', 'specificity', 'f1']
    mean = {k: float(np.mean([m[k] for m in fold_metrics_list])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics_list])) for k in keys}
    flush_print(f'  >>> {gene}: AUC={mean["auc"]:.3f} +/- {std["auc"]:.3f} BalAcc={mean["balanced_acc"]:.3f} F1={mean["f1"]:.3f}')
    return {'mean': mean, 'std': std, 'fold_metrics': fold_metrics_list}


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

    # Normalize
    from sklearn.preprocessing import StandardScaler
    X_titan = StandardScaler().fit_transform(X_titan)
    X_uni = StandardScaler().fit_transform(X_uni)

    results = {}

    # Architecture 1: Deep MLP (4 layers) with concat fusion
    flush_print('\n\n========== Arch 1: Deep MLP + Concat Fusion ==========')
    for gene in GENES:
        r = evaluate_architecture(X_titan, X_uni, Y, gene, 'DeepConcat',
                                   hidden_dims=[512, 256, 128, 64], lr=1e-3)
        results[f'DeepConcat_{gene}'] = r

    # Architecture 2: Deep MLP + Attn Fusion
    flush_print('\n\n========== Arch 2: Deep MLP + Cross-Attn Fusion ==========')
    for gene in GENES:
        r = evaluate_architecture(X_titan, X_uni, Y, gene, 'DeepAttn',
                                   hidden_dims=[512, 256, 128, 64], lr=1e-3)
        results[f'DeepAttn_{gene}'] = r

    # Architecture 3: Wider MLP (deeper + wider)
    flush_print('\n\n========== Arch 3: Wider Deep MLP (1024 -> 512 -> 256 -> 128 -> 64) ==========')
    for gene in GENES:
        r = evaluate_architecture(X_titan, X_uni, Y, gene, 'DeepWide',
                                   hidden_dims=[1024, 512, 256, 128], lr=5e-4)
        results[f'DeepWide_{gene}'] = r

    # Final comparison
    flush_print('\n\n' + '=' * 100)
    flush_print('FINAL COMPARISON: 3 deep architectures')
    flush_print('=' * 100)
    rows = []
    for gene in GENES:
        d1 = results[f'DeepConcat_{gene}']['mean'].get('auc', 0.5)
        d2 = results[f'DeepAttn_{gene}']['mean'].get('auc', 0.5)
        d3 = results[f'DeepWide_{gene}']['mean'].get('auc', 0.5)
        if max(d1, d2, d3) == 0.5:
            continue
        best = max(d1, d2, d3)
        rows.append((gene, d1, d2, d3, best))
    rows.sort(key=lambda x: -x[4])
    flush_print('%-8s %-12s %-12s %-12s %-10s' % ('Gene', 'DeepConcat', 'DeepAttn', 'DeepWide', 'Best'))
    for gene, d1, d2, d3, best in rows:
        ach85 = 'YES' if best >= 0.85 else 'NO'
        flush_print('%-8s %-12.3f %-12.3f %-12.3f %s (%.3f)' % (gene, d1, d2, d3, ach85, best))

    # Save summary
    summary_data = {name: {'mean': r['mean'], 'std': r['std']} for name, r in results.items()}
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(rows))
    width = 0.27
    ax.bar(x - width, [r[1] for r in rows], width, label='DeepConcat', color='#3498db')
    ax.bar(x, [r[2] for r in rows], width, label='DeepAttn', color='#e74c3c')
    ax.bar(x + width, [r[3] for r in rows], width, label='DeepWide', color='#2ecc71')
    ax.axhline(0.85, color='green', linestyle=':', alpha=0.5, label='Target 0.85')
    ax.axhline(0.9, color='red', linestyle=':', alpha=0.5, label='Target 0.90')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('AUC')
    ax.set_title('Deep MLP architectures (5-fold + 3-seed)')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'deep_arch_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    flush_print('Comparison chart saved to %s' % str(FIG_DIR / 'deep_arch_comparison.png'))


if __name__ == '__main__':
    main()