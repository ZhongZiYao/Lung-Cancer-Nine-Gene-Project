"""step24_verify_features.py
对已抽取的 CTransPath 768 维 patch 特征做无监督质量验证。
不训任何 DNN model——只用 sklearn (LR / PCA / t-SNE),全部 CPU 可跑。

3 个子任务:
 1. Linear probe: log-loss / AUC of EGFR status, 用全体 patch 投票
 2. Reconstruction: PCA 95% 累计方差所需维度 (低 = 表征高效)
 3. Per-case separation: 每个 case 自己的 patch cluster 有多大

输出: --out_dir  下:
   linear_probe.csv           (per-case + per-gene AUC)
   pca_95_dim.txt             (int)
   case_separation.csv        (per-case within-cluster avg distance)
   patches_tsne.png           (5 case patch-level t-SNE 可视化)

用法:
  python step24_verify_features.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE


FEAT_ROOT_DEFAULT = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\deepgem_feat'
LABELS_CSV_DEFAULT = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'
GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53', 'BRAF', 'PIK3CA', 'ERBB2', 'NRAS', 'RET', 'MET']


def load_all_features(feat_root: Path, label_map: dict, max_per_case: int = 1000):
    """读所有 case 的 patch .pkl 到 np array + 对应 case_id / gene label"""
    X_list, case_list, lab_list = [], [], []
    by_gene_lab = {g: ([], []) for g in GENES}
    for case_dir in sorted(feat_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if case_id not in label_map:
            continue
        feats = []
        for pkl in sorted(case_dir.rglob('*.pkl')):
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            if isinstance(d, list):
                feats.extend(d)
            else:
                feats.append(d)
        # list[dict{val, feat_name}]
        feat_vecs = [f['val'].reshape(-1) for f in feats if isinstance(f, dict) and 'val' in f]
        if not feat_vecs:
            continue
        # sample to speed up
        if len(feat_vecs) > max_per_case:
            idx = np.random.RandomState(0).choice(len(feat_vecs), max_per_case, replace=False)
            feat_vecs = [feat_vecs[i] for i in idx]
        X_list.extend(feat_vecs)
        case_list.extend([case_id] * len(feat_vecs))
        lab_list.extend([label_map[case_id]] * len(feat_vecs))
    X = np.stack(X_list).astype(np.float32)
    print(f'[INFO] loaded {X.shape[0]} patches x {X.shape[1]} dim, {len(set(case_list))} cases')
    return X, np.array(case_list), np.array(lab_list)


def task_linear_probe(X, case_list, label_map, out_csv: Path):
    """Per-gene: aggregate patches per case → train logistic regression leave-one-out → AUC."""
    print('\n[T1] Linear probe...')
    rows = []
    df_lab = pd.DataFrame(
        [(c, label_map[c]) for c in set(case_list)],
        columns=['case_id', 'EGFR'])  # 只演示 EGFR, 其他同理
    case2label = dict(zip(df_lab.case_id, df_lab.EGFR))

    # Aggregate per-case: mean of patch features
    cases = sorted(set(case_list))
    feat_per_case = np.zeros((len(cases), X.shape[1]), dtype=np.float32)
    labels = np.zeros(len(cases), dtype=np.int64)
    for i, c in enumerate(cases):
        mask = (case_list == c)
        feat_per_case[i] = X[mask].mean(0)
        labels[i] = case2label[c]
    # mean-centered
    feat_per_case -= feat_per_case.mean(0, keepdims=True)
    feat_per_case /= (feat_per_case.std(0, keepdims=True) + 1e-9)

    # Leave-one-out ROC-AUC
    for gene in GENES:
        labels_g = np.array([label_map[c] for c in cases])
        # majority-class AUC baseline
        pos = labels_g.sum()
        base_auc = max(pos, len(labels_g)-pos) / len(labels_g)
        y_score = np.zeros(len(cases))
        for i in range(len(cases)):
            tr_idx = [j for j in range(len(cases)) if j != i]
            te_idx = [i]
            if len(np.unique(labels_g[tr_idx])) < 2:
                y_score[i] = float(labels_g.mean())  # fallback
                continue
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(feat_per_case[tr_idx], labels_g[tr_idx])
            y_score[i] = clf.predict_proba(feat_per_case[[i]])[0, 1]
        try:
            auc = roc_auc_score(labels_g, y_score) if len(np.unique(labels_g)) > 1 else float('nan')
        except Exception:
            auc = float('nan')
        rows.append({'gene': gene, 'baseline_auc': round(base_auc, 3),
                     'linear_probe_auc_loo': round(auc, 3),
                     'n_pos': int(pos), 'n_neg': int(len(labels_g)-pos),
                     'n_cases': len(labels_g)})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f'[OK] → {out_csv}')


def task_pca_95(X, out_txt: Path):
    """PCA: how many dims to retain 95% variance?"""
    print('\n[T2] PCA 95% dim...')
    pca = PCA(n_components=min(128, X.shape[1]), random_state=0)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    out_txt.write_text(
        f'PCA cumulative explained variance (first 20 dims):\n'
        f'cum[:20] = {[round(float(x), 3) for x in cum[:20]]}\n'
        f'# dims for 95% var = {n95} (max 768)\n'
        f'# dims for 90% var = {int(np.searchsorted(cum, 0.90) + 1)}\n'
        f'# dims for 99% var = {int(np.searchsorted(cum, 0.99) + 1)}\n'
        f'total variance explained by 128-dim PCA = {cum[127]:.3f}\n'
    )
    print(out_txt.read_text())
    print(f'[OK] → {out_txt}')


def task_case_separation(X, case_list, out_csv: Path):
    """各 case 内部 patch 紧密度 (越小越均一) + case 之间 (centroid) 距离比 (大 = 好分)"""
    print('\n[T3] Case separation...')
    cases = sorted(set(case_list))
    centroids = np.zeros((len(cases), X.shape[1]), dtype=np.float32)
    for i, c in enumerate(cases):
        mask = (case_list == c)
        centroids[i] = X[mask].mean(0)
    # cosine sim between centroids
    norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    cn = centroids / norm
    sim = cn @ cn.T
    rows = []
    for i, c in enumerate(cases):
        mask = (case_list == c)
        # within-case avg cosine sim (after norm)
        Xn = X[mask]
        Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
        within = float(Xn.mean(0) @ (Xn.mean(0)))  # cohere cn of case
        between = float(sim[i].mean() - sim[i, i] / (len(cases) - 1))
        rows.append({'case_id': c, 'n_patches': int(mask.sum()),
                     'within_case_coherence': round(within, 3),
                     'avg_between_case_sim': round((sim[i].sum() - 1) / (len(cases)-1), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f'[OK] → {out_csv}')


def task_tsne_png(X, case_list, out_png: Path, max_n=2000):
    """5 case patch-level t-SNE visualization"""
    print('\n[T4] t-SNE (might be slow on CPU)...')
    rng = np.random.RandomState(0)
    n = min(max_n, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    sub = X[idx]
    cases_sub = case_list[idx]
    print(f'  using {n} patches, {len(set(cases_sub))} cases')
    # PCA first to 50 dims to speed up t-SNE
    pca = PCA(n_components=50, random_state=0)
    sub50 = pca.fit_transform(sub)
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0, n_jobs=-1)
    Z = tsne.fit_transform(sub50)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 7))
    cases_unique = sorted(set(cases_sub))
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(cases_unique):
        m = (cases_sub == c)
        plt.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.6, label=c, color=cmap(i % 10))
    plt.legend(loc='best', fontsize=9, frameon=False)
    plt.title(f't-SNE of {n} patches ({len(cases_unique)} cases)')
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[OK] → {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', type=Path, default=Path(FEAT_ROOT_DEFAULT))
    ap.add_argument('--labels', type=Path, default=Path(LABELS_CSV_DEFAULT))
    ap.add_argument('--out', type=Path, default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\feature_qa'))
    ap.add_argument('--max-patches-per-case', type=int, default=1500,
                    help='limit patches per case for speed (1500 is enough for stable stats)')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels)
    label_map = dict(zip(df['case_id'], df['EGFR'].astype(int)))

    X, case_arr, _ = load_all_features(args.feat, label_map, args.max_patches_per_case)

    task_pca_95(X, args.out / 'pca_95_dim.txt')
    task_linear_probe(X, case_arr, label_map, args.out / 'linear_probe.csv')
    task_case_separation(X, case_arr, args.out / 'case_separation.csv')
    try:
        task_tsne_png(X, case_arr, args.out / 'patches_tsne.png')
    except Exception as e:
        print(f'[WARN] t-SNE skipped: {e}')
    print(f'\nALL DONE. → {args.out}')


if __name__ == '__main__':
    main()
