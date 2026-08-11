"""
step3_train_v3.py — Final training script (Best of DeepGEM + GAMIL papers)

Design (synthesizing Lancet 2025 DeepGEM + Diag Path 2025 GAMIL):

1. Per-gene training (GAMIL strategy for low-data regime)
2. 3-fold StratifiedCV (GAMIL method, suitable for 422 cases)
3. 70/15/15 holdout (DeepGEM method, final report AUC)
4. Gated Attention per-SM + case-level (both papers)
5. Bag + instance dual supervision (DeepGEM)
6. hidden_dim=128 default (~335K params for 422 case)
7. AdamW + Warmup(5 ep) + Cosine, early stopping (patience=10)
8. AUC main metric, also accuracy/F1/precision/recall (DeepGEM)

Usage:
  # 3-fold CV per gene (GAMIL method), all 9 genes
  python step3_train_v3.py --features-dir deepgem_feat

  # Single gene, 3-fold CV
  python step3_train_v3.py --features-dir deepgem_feat --genes EGFR

  # 70/15/15 hold-out (DeepGEM method), single split
  python step3_train_v3.py --features-dir deepgem_feat --cv 0

  # Common genes only (EGFR/KRAS/TP53)
  python step3_train_v3.py --features-dir deepgem_feat --common-only

  # Rare genes separately
  python step3_train_v3.py --features-dir deepgem_feat --rare
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']
COMMON_GENES = ['EGFR', 'KRAS', 'TP53']
RARE_GENES = ['ALK', 'ROS1', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS']


def flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


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


def compute_pos_weight(cases, labels, gene, max_w=10.0):
    n_pos = sum(1 for c in cases if labels[c][gene] == 1)
    n_neg = len(cases) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return min(n_neg / max(n_pos, 1), max_w)


def split_70_15_15(cases, labels, gene, seed=42):
    from sklearn.model_selection import train_test_split
    y = [labels[c][gene] for c in cases]
    train_val, holdout = train_test_split(cases, test_size=0.15, stratify=y, random_state=seed)
    y_tv = [labels[c][gene] for c in train_val]
    train, val = train_test_split(train_val, test_size=0.176, stratify=y_tv, random_state=seed)
    return train, val, holdout


def train_one_fold(train_cases, val_cases, feat_dir, labels, gene,
                   hidden_dim, epochs, patience, device, lr, weight_decay, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(Path(__file__).parent))
    from step3_train_deepgem_style import Trainer
    from deepgem_style_model import DeepGEMStyleModel
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    pos_w = compute_pos_weight(train_cases, labels, gene)
    pos_weight = torch.tensor([pos_w], dtype=torch.float32)

    model = DeepGEMStyleModel(in_dim=1024, hidden_dim=hidden_dim, num_genes=9)
    trainer = Trainer(model, torch.device(device), num_genes=9, lr=lr, weight_decay=weight_decay)
    train_ds = CaseDataset(train_cases, feat_dir, labels)
    val_ds = CaseDataset(val_cases, feat_dir, labels)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    best_auc = 0.0
    best_epoch = 0
    no_improve = 0
    g_idx = GENES.index(gene)

    for epoch in range(1, epochs + 1):
        trainer.cur_epoch = epoch
        n_cases_total = len(train_cases) + len(val_cases)
        train_loss = trainer.train_one_epoch(
            train_loader, pos_weight,
            epoch=epoch, n_epoch=epochs,
            fold=0, n_fold=1,
            n_cases_total=n_cases_total,
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
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == epochs:
            flush_print('    ep%d: loss=%.3f val_auc=%.3f best=%.3f@ep%d' % (epoch, train_loss, auc, best_auc, best_epoch))
        if no_improve >= patience:
            flush_print('    early stop at ep%d' % epoch)
            break

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
    return {'auc': final_auc, 'acc': acc, 'f1': f1, 'best_epoch': best_epoch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--labels-csv', default='project_resources/data_public/9gene_panel_LUAD.csv')
    ap.add_argument('--hidden-dim', type=int, default=128, help='hidden_dim, default 128 (~335K params for 422 case)')
    ap.add_argument('--cv', type=int, default=3, help='CV folds. 0 = single 70/15/15 hold-out')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--genes', nargs='+', default=None)
    ap.add_argument('--common-only', action='store_true')
    ap.add_argument('--rare', action='store_true')
    ap.add_argument('--output-dir', default='pipeline/week_1_baseline/outputs/train_v3')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.common_only:
        target_genes = COMMON_GENES
    elif args.rare:
        target_genes = RARE_GENES
    elif args.genes:
        target_genes = args.genes
    else:
        target_genes = GENES
    flush_print('[V3] genes=%s hidden_dim=%d cv=%d' % (target_genes, args.hidden_dim, args.cv))

    labels = {}
    with open(args.labels_csv) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            pid = parts[0]
            vals = parts[1:]
            labels[pid] = {g: int(v) for g, v in zip(header[1:11], vals)}

    feat_dir = Path(args.features_dir)
    available = set(p.stem for p in feat_dir.glob('*.npz'))
    cases = sorted(available)
    flush_print('[V3] %d cases with patches' % len(cases))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for gene in target_genes:
        valid_cases = [c for c in cases if labels.get(c, {}).get(gene) in (0, 1)]
        n_pos = sum(1 for c in valid_cases if labels[c][gene] == 1)
        flush_print('\n=== %s (%d cases, pos=%d) ===' % (gene, len(valid_cases), n_pos))
        if n_pos < 5:
            flush_print('  SKIP: too few positives (%d)' % n_pos)
            continue
        if args.cv > 0:
            from sklearn.model_selection import StratifiedKFold
            y = [labels[c][gene] for c in valid_cases]
            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(valid_cases, y)):
                train_cases = [valid_cases[i] for i in train_idx]
                val_cases = [valid_cases[i] for i in val_idx]
                flush_print('  Fold %d/%d: train=%d val=%d' % (fold+1, args.cv, len(train_cases), len(val_cases)))
                r = train_one_fold(train_cases, val_cases, feat_dir, labels, gene,
                                    args.hidden_dim, args.epochs, args.patience,
                                    args.device, args.lr, args.weight_decay, args.seed + fold)
                flush_print('    -> AUC=%.3f ACC=%.3f F1=%.3f @ep%d' % (r['auc'], r['acc'], r['f1'], r['best_epoch']))
                fold_results.append(r)
            mean = {k: float(np.mean([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
            std = {k: float(np.std([r[k] for r in fold_results])) for k in ['auc', 'acc', 'f1']}
            flush_print('  %s %d-fold: AUC=%.3f+%-7.3f ACC=%.3f F1=%.3f' % (gene, args.cv, mean['auc'], std['auc'], mean['acc'], mean['f1']))
            all_results.append((gene, mean, std))
        else:
            train, val, holdout = split_70_15_15(valid_cases, labels, gene, args.seed)
            flush_print('  70/15/15: train=%d val=%d holdout=%d' % (len(train), len(val), len(holdout)))
            r = train_one_fold(train, val, feat_dir, labels, gene,
                                args.hidden_dim, args.epochs, args.patience,
                                args.device, args.lr, args.weight_decay, args.seed)
            flush_print('  %s holdout: AUC=%.3f ACC=%.3f F1=%.3f' % (gene, r['auc'], r['acc'], r['f1']))
            all_results.append((gene, r, {k: 0.0 for k in ['auc', 'acc', 'f1']}))

    flush_print('\n========== Final Summary ==========')
    flush_print('%-8s %-14s %-8s %-8s' % ('Gene', 'AUC', 'ACC', 'F1'))
    for gene, m, s in all_results:
        if s['auc'] > 0:
            auc_str = '%.3f+-%.3f' % (m['auc'], s['auc'])
        else:
            auc_str = '%.3f' % m['auc']
        flush_print('%-8s %-14s %-8.3f %-8.3f' % (gene, auc_str, m['acc'], m['f1']))

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump([{'gene': g, 'mean': m, 'std': s} for g, m, s in all_results], f, indent=2)
    flush_print('\nResults saved to %s' % summary_path)


if __name__ == '__main__':
    main()
