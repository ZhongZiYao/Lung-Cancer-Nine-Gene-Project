"""
baseline_tp53.py
─────────────────
TP53 深度诊断:
  1. 多种分类器(LR / SVM / RF / MLP)对比
  2. 多种特征(case_feature / patch mean / patch max)
  3. 验证 5-fold vs LOO 稳定性
  4. 看是否能稳定 ≥ 0.65
"""
import csv
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BAGS_DIR = Path(r"pipeline\week_1_baseline\outputs\bags_v3")
MANIFEST = Path(r"pipeline\week_1_baseline\case_manifest.csv")

GENE = "TP53"

# 1) 读 manifest
labels, any_pos = {}, {}
with open(MANIFEST) as f:
    rd = csv.DictReader(f)
    for r in rd:
        labels[r["case_id"]] = int(r[GENE])
        any_pos[r["case_id"]] = int(r["any_9gene_positive"])

# 2) 读多种特征
cases = []
feat_case = []   # [500]
feat_patch_mean = []  # [1024]
feat_patch_max = []   # [1024]
y = []
for npz in sorted(BAGS_DIR.glob("*.npz")):
    cid = npz.stem
    if cid not in labels:
        continue
    d = np.load(npz, allow_pickle=True)
    cases.append(cid)
    feat_case.append(d["case_feature"])
    feat_patch_mean.append(d["patch_features"].mean(axis=0))
    feat_patch_max.append(d["patch_features"].max(axis=0))
    y.append(labels[cid])

feat_case = np.stack(feat_case)
feat_patch_mean = np.stack(feat_patch_mean)
feat_patch_max = np.stack(feat_patch_max)
y = np.array(y)
print(f"X case: {feat_case.shape}, y={GENE}+: {y.sum()}/{len(y)}\n")


def eval_clf(name, clf, X, y, splitter, n_repeat=5):
    aucs = []
    for seed in range(n_repeat):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed) \
            if splitter == "5fold" else LeaveOneOut()
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])
            Xte = sc.transform(X[te])
            c = clf.__class__(**clf.get_params())
            c.fit(Xtr, y[tr])
            if hasattr(c, "predict_proba"):
                oof[te] = c.predict_proba(Xte)[:, 1]
            else:
                oof[te] = c.decision_function(Xte)
        aucs.append(roc_auc_score(y, oof))
    aucs = np.array(aucs)
    print(f"  {name:>22s}  {splitter:>5s}  AUC = {aucs.mean():.3f} ± {aucs.std():.3f}  "
          f"({aucs.min():.3f}~{aucs.max():.3f})")
    return aucs.mean()


clfs = {
    "LR(C=0.1,balanced)": LogisticRegression(C=0.1, class_weight="balanced",
                                             max_iter=2000, random_state=42),
    "LR(C=1,balanced)":   LogisticRegression(C=1.0, class_weight="balanced",
                                             max_iter=2000, random_state=42),
    "LR(C=10)":           LogisticRegression(C=10.0, class_weight="balanced",
                                            max_iter=2000, random_state=42),
    "SVM-RBF(C=1)":       SVC(C=1.0, kernel="rbf", probability=True,
                              class_weight="balanced", random_state=42),
    "RF(300)":            RandomForestClassifier(n_estimators=300, max_depth=5,
                                                 class_weight="balanced",
                                                 random_state=42),
    "GBM(200)":           GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                    random_state=42),
    "MLP(64)":            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                        random_state=42, early_stopping=True),
    "MLP(256)":           MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                        random_state=42, early_stopping=True),
}

for feat_name, X in [("case_feature", feat_case),
                      ("patch_mean",  feat_patch_mean),
                      ("patch_max",   feat_patch_max)]:
    print(f"\n=== Feature: {feat_name} (dim={X.shape[1]}) ===")
    for name, clf in clfs.items():
        try:
            eval_clf(name, clf, X, y, "5fold", n_repeat=3)
        except Exception as e:
            print(f"  {name:>22s}  ERR: {e}")

# LOO 验证最稳的那个组合
print(f"\n=== LOO 验证(最稳定特征+模型) ===")
for clf_name, clf in [("LR(C=1)", clfs["LR(C=1,balanced)"]),
                       ("GBM(200)", clfs["GBM(200)"])]:
    oof = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(feat_case, y):
        sc = StandardScaler().fit(feat_case[tr])
        c = clf.__class__(**clf.get_params())
        c.fit(sc.transform(feat_case[tr]), y[tr])
        if hasattr(c, "predict_proba"):
            oof[te] = c.predict_proba(sc.transform(feat_case[te]))[:, 1]
        else:
            oof[te] = c.decision_function(sc.transform(feat_case[te]))
    auc = roc_auc_score(y, oof)
    print(f"  {clf_name:>12s}  case_feature LOO AUC = {auc:.3f}")
