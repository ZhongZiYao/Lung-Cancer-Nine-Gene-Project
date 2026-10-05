# -*- coding: utf-8 -*-
"""38_net_signal_corrected.py — 修正 case_id 映射后重算基因组基线与净信号

bug 7: 早期脚本用严格正则取 case_id, 丢掉了 90/438 例的临床元数据。
本脚本用 01b 的 split 口径重算, 并给出【图像侧不受影响】的说明。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
SEP = "=" * 92

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); K = len(GENES)
strata = np.clip(Y.sum(1), 0, 3).astype(int)
L = np.load(os.path.join(C, "oof_giga_5seed.npy"))
PRED = 1.0 / (1.0 + np.exp(-L))
print(f"[LUAD n={N}] 图像侧 5-seed 集成分数 (与 bug 7 无关, 图像模型不依赖临床协变量)\n")

COLS = ["_PATIENT", "ajcc_pathologic_tumor_stage", "age_at_initial_pathologic_diagnosis",
        "gender", "race"]
gp = pd.read_parquet(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                  "provgigapath_embeddings_with_metadata.parquet"), columns=COLS)

def tn_split(b):
    return "-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None

import re
def tn_re(b):
    m = re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(b))
    return f"TCGA-{m.group(1)}-{m.group(2)}" if m else None

oh = lambda v: pd.get_dummies(pd.Series(v).fillna("__NA__").astype(str)).values.astype(float)

def geno(fn):
    g = gp.copy(); g["case"] = g["_PATIENT"].map(fn)
    gg = g.dropna(subset=["case"]).groupby("case").first()
    STAGE = oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")))
    AGEv = df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float)
    AGE = np.nan_to_num(np.where(np.isfinite(AGEv), AGEv, np.nan), nan=0.0).reshape(-1, 1)
    SEX = oh(df.case_id.map(gg.get("gender")))
    RACE = oh(df.case_id.map(gg.get("race")))
    X = np.hstack([STAGE, SEX, RACE, AGE])
    cov = int(df.case_id.map(gg.get("gender")).notna().sum())
    aucs = []
    for j in range(K):
        Xf = np.column_stack([X] + [Y[:, k] for k in range(K) if k != j])
        o = np.zeros((N, K))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, strata):
            if len(set(Y[tr, j])) < 2:
                continue
            sc = StandardScaler().fit(Xf[tr])
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xf[tr]), Y[tr, j])
            o[te, j] = m.predict_proba(sc.transform(Xf[te]))[:, 1]
        aucs.append(roc_auc_score(Y[:, j], o[:, j]) if 0 < Y[:, j].sum() < N else np.nan)
    return np.array(aucs), X.shape[1], cov

print(SEP, flush=True)
print("=== 基因组基线 (leave-one-out + 临床协变量) 在两种映射口径下 ===", flush=True)
a_re, d_re, c_re = geno(tn_re)
a_sp, d_sp, c_sp = geno(tn_split)
print(f"{'口径':<16}{'临床覆盖':>10}{'特征维':>8}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}")
print(f"{'正则(旧, 有 bug)':<16}{c_re:>10}{d_re:>8}" + "".join(f"{x:>9.3f}" for x in a_re) + f"{a_re.mean():>9.3f}")
print(f"{'split(修正)':<16}{c_sp:>10}{d_sp:>8}" + "".join(f"{x:>9.3f}" for x in a_sp) + f"{a_sp.mean():>9.3f}")
print(f"{'基线差':<16}{c_sp-c_re:>+10}{d_sp-d_re:>+8}" + "".join(f"{x:>+9.3f}" for x in (a_sp - a_re)) + f"{a_sp.mean()-a_re.mean():>+9.3f}")

img = np.array([roc_auc_score(Y[:, j], PRED[:, j]) if 0 < Y[:, j].sum() < N else np.nan for j in range(K)])
print("\n" + SEP, flush=True)
print("=== 净信号 = 图像 AUC − 基因组基线 AUC ===", flush=True)
print(f"{'口径':<18}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}")
print(f"{'图像(5seed集成)':<18}" + "".join(f"{x:>9.3f}" for x in img) + f"{img.mean():>9.3f}")
print(f"{'净信号(旧正则)':<18}" + "".join(f"{x:>+9.3f}" for x in (img - a_re)) + f"{np.mean(img-a_re):>+9.3f}")
print(f"{'净信号(修正后)':<18}" + "".join(f"{x:>+9.3f}" for x in (img - a_sp)) + f"{np.mean(img-a_sp):>+9.3f}")
print(f"\n净信号被高估的幅度: {np.mean(img-a_re) - np.mean(img-a_sp):+.4f}", flush=True)
print("图像侧 AUC 完全不受影响(图像模型不输入任何临床协变量), 变的是分母", flush=True)
pd.DataFrame([dict(gene=g, img_auc=img[j], geno_re=a_re[j], geno_split=a_sp[j],
                   net_re=img[j]-a_re[j], net_split=img[j]-a_sp[j]) for j, g in enumerate(GENES)]
             ).to_csv(os.path.join(C, "net_signal_corrected.csv"), index=False)
print(f"\n[saved] {C}\\net_signal_corrected.csv")
print("DONE", flush=True)