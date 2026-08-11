"""step3_train_v2.py - Direct reader of step1 npz, no step2 needed"""
from __future__ import annotations
import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', required=True)
    ap.add_argument('--labels-csv', default='project_resources/data_public/9gene_panel_LUAD.csv')
    ap.add_argument('--gene', choices=GENES, default='EGFR')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--hidden-dim', type=int, default=128, help='hidden_dim, default 128 for 422 case')
    args = ap.parse_args()

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
    cases = [c for c in cases if c in labels and labels[c][args.gene] in (0, 1)]
    flush_print(f'[Train v2] {len(cases)} cases, gene={args.gene}, device={args.device}')

    sys.path.insert(0, str(Path(__file__).parent))
    from step3_train_deepgem_style import Trainer
    from deepgem_style_model import DeepGEMStyleModel

    from sklearn.model_selection import StratifiedKFold
    labels_gene = [labels[c][args.gene] for c in cases]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    device = torch.device(args.device)
    fold_aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(cases, labels_gene)):
        flush_print(f'\n=== Fold {fold+1}/{args.n_folds} ===')
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        flush_print(f'  train: {len(train_cases)}, val: {len(val_cases)}')

        n_pos = sum(labels[c][args.gene] for c in train_cases)
        n_neg = len(train_cases) - n_pos
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)
        flush_print(f'  pos={n_pos}, neg={n_neg}')

        model = DeepGEMStyleModel(in_dim=1024, hidden_dim=args.hidden_dim, num_genes=9)
        print(f'  model: hidden_dim={args.hidden_dim}, total params={sum(p.numel() for p in model.parameters()):,}')

        train_ds = CaseDataset(train_cases, feat_dir, labels)
        val_ds = CaseDataset(val_cases, feat_dir, labels)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

        trainer = Trainer(model, device, num_genes=9, lr=args.lr)

        best_auc = 0.0
        for epoch in range(1, args.epochs + 1):
            w_inst = trainer.get_w_inst(epoch)
            trainer.train_one_epoch(train_loader, pos_weight, epoch=epoch, n_epoch=args.epochs, fold=fold, n_fold=args.n_folds, w_inst=w_inst)
            if epoch % 5 == 0 or epoch == args.epochs:
                val_metrics = trainer.eval_one_epoch(val_loader, pos_weight)
                auc = val_metrics.get('auc', 0)
                flush_print(f'  epoch {epoch}/{args.epochs}  val_auc={auc:.4f}')
                if auc > best_auc:
                    best_auc = auc
        fold_aucs.append(best_auc)
        flush_print(f'  fold {fold+1} best AUC: {best_auc:.4f}')

    flush_print(f'\n=== Summary ===')
    flush_print(f'5-fold mean AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}')


if __name__ == '__main__':
    main()
