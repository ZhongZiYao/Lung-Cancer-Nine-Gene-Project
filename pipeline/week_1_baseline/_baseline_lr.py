"""
baseline_lr.py
───────────────
最便宜的 baseline:用 case_feature [500] 直接训 Logistic Regression。
如果这个 baseline 也很差 → 问题在特征本身(数据问题)。
如果这个 baseline 不错 → 之前的 deep model 训练有问题。
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BAGS_DIR = Path(r"pipeline\week_1_baseline\outputs\bags_v3")
MANIFEST = Path(r"pipeline\week_1_baseline\case_manifest.csv")

# 1) 读 manifest
labels, any_pos = {}, {}
with open(MANIFEST) as f:
    rd = csv.DictReader(f)
    for r in rd:
        labels[r["case_id"]] = int(r["EGFR"])
        any_pos[r["case_id"]] = int(r["any_9gene_positive"])

# 2) 读 case_feature
cases, X, y_egfr, y_any = [], [], [], []
for npz in sorted(BAGS_DIR.glob("*.npz")):
    cid = npz.stem
    if cid not in labels:
        continue
    d = np.load(npz, allow_pickle=True)
    feat = d["case_feature"]  # [500]
    cases.append(cid)
    X.append(feat)
    y_egfr.append(labels[cid])
    y_any.append(any_pos[cid])

X = np.stack(X, axis=0)
y_egfr = np.array(y_egfr)
y_any = np.array(y_any)
print(f"X shape = {X.shape}")
print(f"EGFR+: {y_egfr.sum()}/{len(y_egfr)}, any_pos+: {y_any.sum()}/{len(y_any)}")

# 3) 标准化
X = StandardScaler().fit_transform(X)

# 4) 5-fold OOF AUC(对每个基因)
GENES = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF", "PIK3CA", "ERBB2", "NRAS"]
with open(MANIFEST) as f:
    rd = csv.DictReader(f)
    gene_label = {r["case_id"]: {g: int(r[g]) for g in GENES} for r in rd}

print(f"\n{'Gene':>8} | {'Pos':>4} | {'OOF AUC':>10}")
print("-" * 30)
for g in GENES:
    y = np.array([gene_label[c][g] for c in cases])
    if y.sum() < 2 or y.sum() > len(y) - 2:
        print(f"{g:>8} | {y.sum():>4} |   skip (too few)")
        continue
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, random_state=42)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"{g:>8} | {y.sum():>4} |   {auc:.3f}")

# 5) any_positive
y = y_any
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tr, te in skf.split(X, y):
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=2000, random_state=42)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
print(f"{'any_pos':>8} | {y.sum():>4} |   {roc_auc_score(y, oof):.3f}")
