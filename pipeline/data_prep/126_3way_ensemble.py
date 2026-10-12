"""126_3way_ensemble.py - 3-way ensemble of students + teacher OOF

Combines:
  - §48 ABMIL student OOF (3fs mean) = 0.8713
  - §124 TransformerMIL student OOF = 0.8704
  - §35 gigapTeacher OOF (3fs mean) = 0.7419 (teacher, not student)

Tests if adding the teacher's soft predictions on top of student ensemble improves mean AUC.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, balanced_accuracy_score

GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8

# Load
abmil_data = np.load(r'E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\s117b_3fold_oof.npz', allow_pickle=True)
Y = abmil_data['Y']
oof_abmil = (abmil_data['oof_fs101'] + abmil_data['oof_fs202'] + abmil_data['oof_fs303']) / 3

tf_data = np.load(r'E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\s124_transformer_mil.npz', allow_pickle=True)
oof_tf = tf_data['oof']

gigap_oof = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
common_t = list(gigap_oof["common"])
abmil_common = list(abmil_data['common'])
g_idx = {c: i for i, c in enumerate(common_t)}
remap = np.array([g_idx[c] for c in abmil_common])
oof_teacher = (gigap_oof["fs0"][remap] + gigap_oof["fs1"][remap] + gigap_oof["fs2"][remap]) / 3


def mean_auc_safe(Y, P):
    aucs = []
    for j in range(len(GENES)):
        npos = int(Y[:, j].sum())
        if npos >= MIN_POS and npos <= len(Y) - MIN_POS:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


strategies = {
    "ABMIL §48": oof_abmil,
    "Transformer §124": oof_tf,
    "Teacher §35 (3fs mean)": oof_teacher,
    "ABMIL + Transformer (mean)": (oof_abmil + oof_tf) / 2,
    "ABMIL + Teacher": 0.6 * oof_abmil + 0.4 * oof_teacher,
    "Transformer + Teacher": 0.6 * oof_tf + 0.4 * oof_teacher,
    "ABMIL + TF + Teacher (1/3 each)": (oof_abmil + oof_tf + oof_teacher) / 3,
    "ABMIL + TF + Teacher (0.4/0.3/0.3)": 0.4 * oof_abmil + 0.3 * oof_tf + 0.3 * oof_teacher,
}

print(f"{'strategy':<50}{'mean AUC':>10}{'STK11':>10}{'KRAS':>10}{'TP53':>10}")
results = {}
for name, oof in strategies.items():
    auc = mean_auc_safe(Y, oof)
    pg = {g: roc_auc_score(Y[:, gi], oof[:, gi]) for gi, g in enumerate(GENES)}
    results[name] = (auc, pg)
    print(f"{name:<50}{auc:>10.4f}{pg['STK11']:>10.4f}{pg['KRAS']:>10.4f}{pg['TP53']:>10.4f}")

best_name = max(results, key=lambda k: results[k][0])
print(f"\nBest: {best_name} = {results[best_name][0]:.4f}")
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s126_3way_ensemble.npz",
                    oof=strategies[best_name], Y=Y, common=np.array(abmil_common, dtype=object))
print(f"[saved] s126_3way_ensemble.npz")