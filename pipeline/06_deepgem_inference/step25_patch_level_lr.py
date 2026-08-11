"""Patch-level EGFR 区分度验证

不训 model,不依赖 DeepGEM forward。
只用 CTransPath 768 维 patch 特征 + sklearn Logistic Regression,
看 patch-level 768-dim 是否区分 EGFR+ vs EGFR- case-level label。

输入:
  --feat   : deepgem_feat/<case>/<patch>.pkl 目录
  --labels : 9gene_panel_LUAD.csv (case-level EGFR)
输出:
  patch_level_pred.csv: 每 patch 的 LR 预测概率
  patch_level_auc.txt : EGFR patch-level ROC-AUC

注意: patch-level AUC 不直接对等于 case-level AUC,
但能给"patch 特征表达是否真的区分病理差异"一个 baseline 信号。
"""
from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force utf-8 (Windows gbk workaround)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'))
    ap.add_argument('--labels', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'))
    ap.add_argument('--out', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\feature_qa'))
    ap.add_argument('--max-patches-per-case', type=int, default=800)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df_lab = pd.read_csv(args.labels)
    label_map = dict(zip(df_lab['case_id'], df_lab['EGFR'].astype(int)))

    # ---- load all patches ----
    X_list, y_list, case_list = [], [], []
    rng = np.random.RandomState(0)
    for case_dir in sorted(args.feat.iterdir()):
        if not case_dir.is_dir():
            continue
        c = case_dir.name
        if c not in label_map:
            continue
        feats = []
        for pkl in sorted(case_dir.rglob('*.pkl')):
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, list):
                feats.extend(d)
            else:
                feats.append(d)
        feats = [f['val'].reshape(-1) for f in feats if isinstance(f, dict) and 'val' in f]
        if not feats:
            continue
        if len(feats) > args.max_patches_per_case:
            feats = [feats[i] for i in rng.choice(len(feats), args.max_patches_per_case, replace=False)]
        X_list.extend(feats)
        n = len(feats)
        y_list.extend([label_map[c]] * n)
        case_list.extend([c] * n)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    cases = np.array(case_list)
    print(f'[INFO] {X.shape[0]} patches, {len(set(cases))} cases, label balance: pos={int(y.sum())}, neg={int((y==0).sum())}')

    # ---- patch-level stratified 5-fold CV by case ----
    from sklearn.model_selection import StratifiedKFold
    print('\n[Patch-level Stratified 5-fold CV (case-level stratified) ]')
    case_unique = sorted(set(cases))
    n_class = max(np.bincount([label_map[c] for c in case_unique]))
    n_splits = min(5, n_class, len(case_unique))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    print(f'  using {n_splits}-fold (limited by smallest class)')
    y_score = np.zeros(len(cases))
    aucs_per_fold = []
    for fold_idx, (tr_case_idx, te_case_idx) in enumerate(skf.split(case_unique, np.array([label_map[c] for c in case_unique]))):
        tr_cases = set(case_unique[i] for i in tr_case_idx)
        te_cases = set(case_unique[i] for i in te_case_idx)
        train_mask = np.array([c in tr_cases for c in cases])
        test_mask = np.array([c in te_cases for c in cases])
        if len(set(y[train_mask])) < 2:
            print(f'  fold {fold_idx+1}: skip (single class in train)')
            continue
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(X[train_mask], y[train_mask])
        y_score[test_mask] = clf.predict_proba(X[test_mask])[:, 1]
        try:
            f_auc = roc_auc_score(y[test_mask], y_score[test_mask])
            aucs_per_fold.append(f_auc)
            print(f'  fold {fold_idx+1}: AUC={f_auc:.4f}  (held-out {len(te_cases)} cases, {int(test_mask.sum())} patches)')
        except Exception as e:
            print(f'  fold {fold_idx+1}: skip AUC ({e})')
    try:
        auc = roc_auc_score(y, y_score)
    except Exception as e:
        print(f'AUC err: {e}')
        auc = float('nan')
    print(f'\nPATCH-LEVEL 5-fold CV ROC-AUC (case-level stratified) = {auc:.4f}')
    print(f'   per-fold mean AUC = {np.mean(aucs_per_fold):.4f} ± {np.std(aucs_per_fold):.4f}')

    # Save patch-level predictions
    pred_df = pd.DataFrame({'case_id': cases, 'true_egfr': y, 'pred_prob': y_score})
    pred_df.to_csv(args.out / 'patch_level_pred.csv', index=False)

    # ---- Case-level AUC via patch-mean aggregation ----
    print('\n[Case-level AUC (from above 5-fold patch probs, mean over each case)]')
    case_scores = {c: float(y_score[cases == c].mean()) for c in case_unique}
    case_truths = {c: int(label_map[c]) for c in case_unique}
    rows = [{'case_id': c, 'true_egfr': case_truths[c],
             'mean_patch_prob_positive': round(case_scores[c], 4)} for c in case_unique]
    casedf = pd.DataFrame(rows)
    casedf.to_csv(args.out / 'patch_lr_case_level.csv', index=False)
    try:
        case_auc = roc_auc_score([case_truths[c] for c in case_unique],
                                  [case_scores[c] for c in case_unique])
    except Exception as e:
        case_auc = float('nan')
    print(f'CASE-LEVEL AUC (patch-mean from 5-fold) = {case_auc:.4f}')
    print('\ncase-level predictions (sorted):')
    for c in case_unique:
        print(f'  {c:18s}  true={case_truths[c]}  mean_patch_prob={case_scores[c]:.4f}')

    with open(args.out / 'patch_level_auc.txt', 'w') as f:
        f.write(f'patch-level 5-fold CV ROC-AUC = {auc:.4f}\n')
        f.write(f'case-level (patch-mean) ROC-AUC = {case_auc:.4f}\n')
        f.write(f'# cases used: {len(case_unique)}  ({int(sum(label_map[c] for c in case_unique))} EGFR+)\n')
        f.write(f'# total patches: {X.shape[0]}\n')
        f.write(f'per-fold AUC: ' + ', '.join([f'{a:.3f}' for a in aucs_per_fold]) + '\n')
    print(f'\n[OK] → {args.out}/patch_level_auc.txt')


if __name__ == '__main__':
    main()
