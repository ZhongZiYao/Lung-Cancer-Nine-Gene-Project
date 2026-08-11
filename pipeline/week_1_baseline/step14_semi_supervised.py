# -*- coding: utf-8 -*-
"""
step14_semi_supervised.py — Semi-supervised learning using TITAN's 9,547 TCGA-LUAD cases

Strategy:
1. Train teacher on 383 labeled TCGA-LUAD cases (TITAN covered)
2. Predict 9,547 unlabeled cases (TITAN all)
3. High-confidence predictions become pseudo-labels
4. Train student on labeled + pseudo-labeled
5. Validate on 5-fold CV of labeled cases
"""
import json
import pickle
import re
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
OUT_DIR = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\week_1_baseline\outputs\iter13_semi_supervised')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
HIGH_CONF = 0.85
LOW_CONF = 0.15
TITAN_DIM = 768
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
    return {cid: data['embeddings'][idxs].mean(axis=0) for cid, idxs in case_to_indices.items()}, data['filenames']


class MLP(nn.Module):
    def __init__(self, in_dim=TITAN_DIM, hidden=128, num_outputs=NUM_GENES, dropout=0.5):
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


def train_teacher(X_labeled, Y_labeled, X_val, Y_val, hidden, lr, epochs, patience,
                    pos_w_max, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_X = torch.from_numpy(X_labeled).float().to(device)
    train_Y = torch.from_numpy(Y_labeled).float().to(device)
    val_X = torch.from_numpy(X_val).float().to(device)
    val_Y = torch.from_numpy(Y_val).float().to(device)
    pos_w = torch.tensor([5.0] * NUM_GENES).to(device)
    model = MLP(in_dim=X_labeled.shape[1], hidden=hidden, num_outputs=NUM_GENES).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_mean_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score
    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(train_X)
        loss = F.binary_cross_entropy_with_logits(logits, train_Y, pos_weight=pos_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            val_p = model(val_X).sigmoid().cpu().numpy()
        # Mean AUC
        aucs = []
        for g in range(NUM_GENES):
            try:
                if val_Y[:, g].sum() >= 1 and val_Y[:, g].sum() < len(val_Y):
                    aucs.append(roc_auc_score(val_Y[:, g].numpy(), val_p[:, g]))
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
    return model, best_mean_auc


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        pred = model(X_t).sigmoid().cpu().numpy()
    return pred


def train_student_with_pseudo(X_labeled, Y_labeled, X_pseudo, Y_pseudo, X_val, Y_val,
                                hidden, lr, epochs, patience, pos_w_max, device, seed,
                                pseudo_weight=0.3):
    """Train student with both labeled and pseudo-labeled data"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_X = torch.from_numpy(X_labeled).float().to(device)
    train_Y = torch.from_numpy(Y_labeled).float().to(device)
    pseudo_X = torch.from_numpy(X_pseudo).float().to(device)
    pseudo_Y = torch.from_numpy(Y_pseudo).float().to(device)
    val_X = torch.from_numpy(X_val).float().to(device)
    val_Y = torch.from_numpy(Y_val).float().to(device)
    pos_w = torch.tensor([5.0] * NUM_GENES).to(device)

    model = MLP(in_dim=X_labeled.shape[1], hidden=hidden, num_outputs=NUM_GENES).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_mean_auc = 0.0
    best_state = None
    no_improve = 0
    from sklearn.metrics import roc_auc_score
    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        # Labeled loss (full weight)
        logits_l = model(train_X)
        loss_l = F.binary_cross_entropy_with_logits(logits_l, train_Y, pos_weight=pos_w)
        # Pseudo loss (lower weight)
        logits_p = model(pseudo_X)
        loss_p = F.binary_cross_entropy_with_logits(logits_p, pseudo_Y, pos_weight=pos_w)
        loss = loss_l + pseudo_weight * loss_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            val_p = model(val_X).sigmoid().cpu().numpy()
        aucs = []
        for g in range(NUM_GENES):
            try:
                if val_Y[:, g].sum() >= 1 and val_Y[:, g].sum() < len(val_Y):
                    aucs.append(roc_auc_score(val_Y[:, g].numpy(), val_p[:, g]))
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
    return model, best_mean_auc


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


def main():
    flush_print('[Semi-supervised] Loading TITAN features...')
    case_emb, filenames = load_titan_embeddings(TITAN_PKL)
    df = pd.read_csv(LABELS_CSV)
    labels = {row['case_id']: {g: int(row[g]) for g in GENES} for _, row in df.iterrows()}

    labeled_cases = sorted([c for c in case_emb if c in labels and
                            all(labels[c][g] in (0, 1) for g in GENES)])
    flush_print(f'Labeled cases: {len(labeled_cases)}')

    # All TCGA cases (including unlabeled)
    all_cases = sorted(case_emb.keys())
    unlabeled_cases = [c for c in all_cases if c not in labeled_cases]
    flush_print(f'Unlabeled cases (potential pseudo-labels): {len(unlabeled_cases)}')

    X_labeled = np.stack([case_emb[c] for c in labeled_cases])
    Y_labeled = np.stack([[labels[c][g] for g in GENES] for c in labeled_cases])
    X_unlabeled = np.stack([case_emb[c] for c in unlabeled_cases])

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.concatenate([X_labeled, X_unlabeled], axis=0))
    X_labeled = scaler.transform(X_labeled)
    X_unlabeled = scaler.transform(X_unlabeled)

    # 5-fold CV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    y_strat = Y_labeled[:, GENES.index('TP53')]  # Use TP53 for stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_baseline = {g: [] for g in GENES}
    results_semi = {g: [] for g in GENES}

    device = torch.device('cpu')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_strat)):
        flush_print(f'\n=== Fold {fold+1}/5 ===')
        X_train_l = X_labeled[train_idx]
        Y_train_l = Y_labeled[train_idx]
        X_val = X_labeled[val_idx]
        Y_val = Y_labeled[val_idx]

        # Step 1: Train teacher on labeled
        teacher, _ = train_teacher(X_train_l, Y_train_l, X_val, Y_val,
                                     hidden=128, lr=1e-3, epochs=50, patience=10,
                                     pos_w_max=10.0, device=device, seed=42+fold)

        # Step 2: Predict on unlabeled
        flush_print(f'  Predicting {len(X_unlabeled)} unlabeled cases...')
        pseudo_pred = predict(teacher, X_unlabeled, device)

        # Step 3: Filter high-confidence pseudo-labels
        # Keep only cases where at least one gene has high or low confidence
        n_kept = 0
        X_pseudo_list = []
        Y_pseudo_list = []
        for i in range(len(X_unlabeled)):
            high_conf_mask = (pseudo_pred[i] > HIGH_CONF) | (pseudo_pred[i] < LOW_CONF)
            if high_conf_mask.sum() >= 3:  # At least 3 genes have high confidence
                X_pseudo_list.append(X_unlabeled[i])
                # Hard pseudo-labels
                y_pseudo = (pseudo_pred[i] > 0.5).astype(np.float32)
                Y_pseudo_list.append(y_pseudo)
                n_kept += 1
        flush_print(f'  Kept {n_kept}/{len(X_unlabeled)} pseudo-labeled cases')

        if n_kept > 0:
            X_pseudo = np.stack(X_pseudo_list)
            Y_pseudo = np.stack(Y_pseudo_list)
        else:
            X_pseudo = np.zeros((0, X_labeled.shape[1]))
            Y_pseudo = np.zeros((0, NUM_GENES))

        # Step 4: Evaluate baseline (no semi-supervised)
        # Multi-seed
        for seed in SEEDS:
            _, _ = train_teacher(X_train_l, Y_train_l, X_val, Y_val,
                                   hidden=128, lr=1e-3, epochs=50, patience=10,
                                   pos_w_max=10.0, device=device, seed=seed)
            val_p = predict(_, X_val, device) if False else None
        # Just one model for baseline (using teacher which is trained)
        baseline_pred = predict(teacher, X_val, device)
        for g in GENES:
            g_idx = GENES.index(g)
            y_true = Y_val[:, g_idx]
            if y_true.sum() < 1 or y_true.sum() == len(y_true):
                continue
            metrics = best_threshold_metrics(y_true, baseline_pred[:, g_idx])
            metrics['auc'] = float(roc_auc_score(y_true, baseline_pred[:, g_idx]))
            results_baseline[g].append(metrics)

        # Step 5: Train student with pseudo-labels
        seed_preds = []
        for seed in SEEDS:
            student, _ = train_student_with_pseudo(
                X_train_l, Y_train_l, X_pseudo, Y_pseudo, X_val, Y_val,
                hidden=128, lr=1e-3, epochs=50, patience=10,
                pos_w_max=10.0, device=device, seed=seed,
                pseudo_weight=0.3,
            )
            val_p = predict(student, X_val, device)
            seed_preds.append(val_p)
        val_p_semi = np.mean(seed_preds, axis=0)
        for g in GENES:
            g_idx = GENES.index(g)
            y_true = Y_val[:, g_idx]
            if y_true.sum() < 1 or y_true.sum() == len(y_true):
                continue
            metrics = best_threshold_metrics(y_true, val_p_semi[:, g_idx])
            metrics['auc'] = float(roc_auc_score(y_true, val_p_semi[:, g_idx]))
            results_semi[g].append(metrics)

    # Final summary
    flush_print('\n' + '=' * 100)
    flush_print('Semi-supervised results: Baseline vs Semi-supervised')
    flush_print('=' * 100)
    flush_print('%-8s %-12s %-12s %-10s' % ('Gene', 'Baseline', 'Semi-sup', 'Diff'))

    final = {}
    for g in GENES:
        if not results_baseline[g] or not results_semi[g]:
            continue
        base_auc = float(np.mean([m['auc'] for m in results_baseline[g]]))
        semi_auc = float(np.mean([m['auc'] for m in results_semi[g]]))
        diff = semi_auc - base_auc
        flag = ' <--' if diff > 0.005 else (' -->' if diff < -0.005 else '')
        flush_print('%-8s %-12.3f %-12.3f %+.3f%s' % (g, base_auc, semi_auc, diff, flag))
        final[g] = {
            'baseline': {'mean': {'auc': base_auc}, 'std': {}},
            'semi': {'mean': {'auc': semi_auc}, 'std': {}},
            'diff': diff,
        }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(final, f, indent=2)
    flush_print('\nResults saved to %s' % str(OUT_DIR / 'summary.json'))


if __name__ == '__main__':
    main()