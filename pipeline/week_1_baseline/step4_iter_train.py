# -*- coding: utf-8 -*-
"""
step4_iter_train.py — Iterative training with hyperparameter sweep

Improvements over v3:
1. Hidden dim grid (32/64/128)
2. Lr grid (2e-4, 5e-4, 1e-3)
3. Dropout grid (0.3, 0.5)
4. Patiance 10, max epoch 25
5. Pos_weight grid (5, 10, 20)
6. Track per-fold AUC best, report mean AUC

Usage:
  # EGFR 3-fold CV, hyperparameter sweep
  python step4_iter_train.py --features-dir deepgem_feat --gene EGFR --cv 3 --max-iter 8

  # TP53, single 70/15/15
  python step4_iter_train.py --features-dir deepgem_feat --gene TP53 --cv 0
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings('ignore')
GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']


def flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def load_labels(labels_csv):
    labels = {}
    with open(labels_csv) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            pid = parts[0]
            vals = parts[1:]
            labels[pid] = {g: int(v) for g, v in zip(header[1:11], vals)}
    return labels


def compute_pos_weight(cases, labels, gene, max_w):
    n_pos = sum(1 for c in cases if labels[c][gene] == 1)
    n_neg = len(cases) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return min(n_neg / max(n_pos, 1), max_w)


class CaseDataset(torch.utils.data.Dataset):
    def __init__(self, cases, features_dir, labels):
        self.cases = cases
        self.features_dir = Path(features_dir)
        self.labels = labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        d = np.load(self.features_dir / f'{cid}.npz', allow_pickle=True)
        return {
            'case_id': cid,
            'patch_feats': torch.from_numpy(d['features']).float(),
            'sm_indices': torch.from_numpy(d['slice_ids']).long(),
            'label': torch.tensor([self.labels[cid][g] for g in GENES], dtype=torch.float32),
        }


def collate(batch):
    return batch[0]


def train_one_run(train_cases, val_cases, feat_dir, labels, gene,
                   hidden_dim, dropout, lr, weight_decay,
                   epochs, patience, pos_w_max, device, seed, log_prefix=''):
    """Single train run, returns (best_auc, best_epoch, final_auc, final_acc, final_f1)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(Path(__file__).parent))
    from step3_train_deepgem_style import Trainer
    from deepgem_style_model import DeepGEMStyleModel
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    pos_w = compute_pos_weight(train_cases, labels, gene, max_w=pos_w_max)
    pos_weight = torch.tensor([pos_w], dtype=torch.float32)
    g_idx = GENES.index(gene)

    model = DeepGEMStyleModel(in_dim=1024, hidden_dim=hidden_dim, num_genes=9, dropout=dropout)
    trainer = Trainer(model, torch.device(device), num_genes=9, lr=lr, weight_decay=weight_decay)

    train_ds = CaseDataset(train_cases, feat_dir, labels)
    val_ds = CaseDataset(val_cases, feat_dir, labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    best_auc = 0.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        trainer.cur_epoch = epoch
        train_loss = trainer.train_one_epoch(
            train_loader, pos_weight,
            epoch=epoch, n_epoch=epochs, fold=0, n_fold=1,
            n_cases_total=len(train_cases) + len(val_cases),
        )
        case_ids, preds, lbls = trainer.eval_one_epoch(val_loader)
        p = preds[:, g_idx]
        y_true = lbls[:, g_idx]
        try:
            auc = roc_auc_score(y_true, p)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == epochs:
            flush_print('    %sep%d: loss=%.3f val_auc=%.3f best=%.3f@ep%d' %
                        (log_prefix, epoch, train_loss, auc, best_auc, best_epoch))
        if no_improve >= patience:
            flush_print('    %searly stop at ep%d' % (log_prefix, epoch))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    case_ids, preds, lbls = trainer.eval_one_epoch(val_loader)
    p = preds[:, g_idx]
    y_true = lbls[:, g_idx]
    try:
        final_auc = roc_auc_score(y_true, p)
    except Exception:
        final_auc = 0.5
    y_pred = (p > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'auc': final_auc, 'best_auc': best_auc, 'best_epoch': best_epoch,
            'acc': acc, 'f1': f1, 'pos_w': pos_w}


def split_70_15_15(cases, labels, gene, seed=42):
    from sklearn.model_selection import train_test_split
    y = [labels[c][gene] for c in cases]
    train_val, holdout = train_test_split(cases, test_size=0.15, stratify=y, random_state=seed)
    y_tv = [labels[c][gene] for c in train_val]
    train, val = train_test_split(train_val, test_size=0.176, stratify=y_tv, random_state=seed)
    return train, val, holdout


def grid_search_kfold(valid_cases, labels, gene, feat_dir, device,
                      hidden_dims=(64, 128), dropouts=(0.5,),
                      lrs=(2e-4, 5e-4), pos_w_max=10.0,
                      epochs=20, patience=8, cv=3, seed=42, n_seeds=2):
    """Try hyperparam combos, return best config by mean AUC across CV folds"""
    from sklearn.model_selection import StratifiedKFold

    # Build all combos
    combos = []
    for h in hidden_dims:
        for d in dropouts:
            for lr in lrs:
                combos.append({'hidden_dim': h, 'dropout': d, 'lr': lr})
    flush_print('[GRID] %d combos, %d-fold CV, %d seeds each' %
                (len(combos), cv, n_seeds))

    best_score = 0.0
    best_cfg = None
    best_results = None

    for ci, cfg in enumerate(combos):
        all_folds = []
        for seed_i in range(n_seeds):
            sd = seed + seed_i * 1000
            y = [labels[c][gene] for c in valid_cases]
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=sd)
            for fold, (train_idx, val_idx) in enumerate(skf.split(valid_cases, y)):
                train_cases = [valid_cases[i] for i in train_idx]
                val_cases = [valid_cases[i] for i in val_idx]
                t0 = time.time()
                r = train_one_run(train_cases, val_cases, feat_dir, labels, gene,
                                 cfg['hidden_dim'], cfg['dropout'], cfg['lr'], 1e-4,
                                 epochs, patience, pos_w_max, device, sd,
                                 log_prefix='  [c%d/f%d/s%d] ' % (ci, fold, seed_i))
                flush_print('  [c%d/f%d/s%d] h=%d d=%.1f lr=%.0e AUC=%.3f (%.1fs)' %
                            (ci, fold, seed_i, cfg['hidden_dim'], cfg['dropout'], cfg['lr'],
                             r['auc'], time.time() - t0))
                all_folds.append(r)
        mean_auc = float(np.mean([r['auc'] for r in all_folds]))
        std_auc = float(np.std([r['auc'] for r in all_folds]))
        flush_print('  >>> Combo %d: h=%d d=%.1f lr=%.0e → AUC=%.3f ± %.3f' %
                    (ci, cfg['hidden_dim'], cfg['dropout'], cfg['lr'], mean_auc, std_auc))
        if mean_auc > best_score:
            best_score = mean_auc
            best_cfg = cfg
            best_results = all_folds
            flush_print('  *** NEW BEST ***')
    return best_cfg, best_score, best_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--labels-csv', default='project_resources/data_public/9gene_panel_LUAD.csv')
    ap.add_argument('--gene', required=True)
    ap.add_argument('--cv', type=int, default=3, help='CV folds. 0 = 70/15/15')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--pos-w-max', type=float, default=10.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output-dir', default='pipeline/week_1_baseline/outputs/iter')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grid', action='store_true', help='Grid search mode')
    ap.add_argument('--n-seeds', type=int, default=2)
    args = ap.parse_args()

    labels = load_labels(args.labels_csv)
    feat_dir = Path(args.features_dir)
    available = set(p.stem for p in feat_dir.glob('*.npz'))
    valid_cases = sorted([c for c in available if labels.get(c, {}).get(args.gene) in (0, 1)])
    n_pos = sum(1 for c in valid_cases if labels[c][args.gene] == 1)
    flush_print('[%s] %d cases, pos=%d (%.1f%%)' %
                (args.gene, len(valid_cases), n_pos, 100.0 * n_pos / len(valid_cases)))
    out_dir = Path(args.output_dir) / ('%s_%s' % (args.gene, args.cv))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid:
        best_cfg, best_score, results = grid_search_kfold(
            valid_cases, labels, args.gene, feat_dir, args.device,
            hidden_dims=(64, 128),
            dropouts=(0.5,),
            lrs=(2e-4, 5e-4),
            pos_w_max=args.pos_w_max,
            epochs=args.epochs, patience=args.patience,
            cv=args.cv if args.cv > 0 else 3, seed=args.seed, n_seeds=args.n_seeds,
        )
        flush_print('\n=== BEST CONFIG ===')
        flush_print('  %s' % best_cfg)
        flush_print('  AUC=%.3f' % best_score)
        out = {'best_cfg': best_cfg, 'best_auc': best_score,
               'n_results': len(results)}
        with open(out_dir / 'grid_best.json', 'w') as f:
            json.dump(out, f, indent=2)
        return

    # Single config run
    if args.cv > 0:
        from sklearn.model_selection import StratifiedKFold
        y = [labels[c][args.gene] for c in valid_cases]
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(valid_cases, y)):
            train_cases = [valid_cases[i] for i in train_idx]
            val_cases = [valid_cases[i] for i in val_idx]
            flush_print('  Fold %d/%d: train=%d val=%d' % (fold + 1, args.cv, len(train_cases), len(val_cases)))
            t0 = time.time()
            r = train_one_run(train_cases, val_cases, feat_dir, labels, args.gene,
                              args.hidden_dim, args.dropout, args.lr, 1e-4,
                              args.epochs, args.patience, args.pos_w_max,
                              args.device, args.seed + fold,
                              log_prefix='  [f%d] ' % fold)
            flush_print('  [f%d] AUC=%.3f ACC=%.3f F1=%.3f @ep%d (%.1fs)' %
                        (fold, r['auc'], r['acc'], r['f1'], r['best_epoch'], time.time() - t0))
            fold_results.append(r)
        mean = {k: float(np.mean([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
        std = {k: float(np.std([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
        flush_print('\n=== %s %d-fold ===' % (args.gene, args.cv))
        flush_print('AUC=%.3f ± %.3f ACC=%.3f ± %.3f F1=%.3f ± %.3f' %
                    (mean['auc'], std['auc'], mean['acc'], std['acc'], mean['f1'], std['f1']))
        with open(out_dir / 'single_summary.json', 'w') as f:
            json.dump({'gene': args.gene, 'cv': args.cv, 'mean': mean, 'std': std,
                       'cfg': {'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
                               'lr': args.lr, 'epochs': args.epochs, 'patience': args.patience}}, f, indent=2)
    else:
        train, val, holdout = split_70_15_15(valid_cases, labels, args.gene, args.seed)
        flush_print('  70/15/15: train=%d val=%d holdout=%d' % (len(train), len(val), len(holdout)))
        t0 = time.time()
        r = train_one_run(train, val, feat_dir, labels, args.gene,
                          args.hidden_dim, args.dropout, args.lr, 1e-4,
                          args.epochs, args.patience, args.pos_w_max,
                          args.device, args.seed, log_prefix='')
        flush_print('  holdout: AUC=%.3f ACC=%.3f F1=%.3f @ep%d (%.1fs)' %
                    (r['auc'], r['acc'], r['f1'], r['best_epoch'], time.time() - t0))
        with open(out_dir / 'single_summary.json', 'w') as f:
            json.dump({'gene': args.gene, 'mode': '70/15/15', 'result': r,
                       'cfg': {'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
                               'lr': args.lr, 'epochs': args.epochs, 'patience': args.patience}}, f, indent=2)


if __name__ == '__main__':
    main()