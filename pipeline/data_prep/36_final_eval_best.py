# -*- coding: utf-8 -*-
"""36_final_eval_best.py — 用当前最优配置重算全部临床指标

最优配置 (36 号脚本确立):
  LUAD n=436 | Prov-GigaPath 512 patch | gated MIL | 5-fold OOF | 5-seed OOF 集成
  mean AUC 0.7431   (此前所有指标表都是单种子的, 必须全部重算)

注意: oof_*.npy 存的是 logits, 算 Brier/ECE/阈值前必须过 sigmoid。
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
SEP = "=" * 100

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = idx[["case_id", "mem_row"]].merge(iu[["case_id"]], on="case_id", how="inner")
df = df.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int)
L = np.load(os.path.join(C, "oof_giga_5seed.npy"))      # logits
P = 1.0 / (1.0 + np.exp(-L))                            # 概率
assert len(Y) == len(P)
print(f"[LUAD n={N}] Prov-GigaPath 512 + gated MIL + 5-fold OOF + 5-seed 集成\n")

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan

def roc(y, p):
    o = np.argsort(-p); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    return np.concatenate([[0], tp / max(int(y.sum()), 1)]), np.concatenate([[0], fp / max(int((1 - y).sum()), 1)])

def sen_at_spec(y, p, s):
    tpr, fpr = roc(y, p)
    ok = fpr <= (1.0 - s)                     # 关键: 约束必须写在 fpr 上
    return float(tpr[ok].max()) if ok.any() else 0.0

def spec_at_sen(y, p, t):
    tpr, fpr = roc(y, p)
    ok = tpr >= t
    return float(1 - fpr[ok].min()) if ok.any() else 1.0

def thr_at_spec(y, p, s):
    tpr, fpr = roc(y, p)
    ok = np.where(fpr <= (1.0 - s))[0]
    return float(p[np.argsort(-p)[ok[-1]]]) if len(ok) else 1.0

def ece(y, p, nb=10):
    b = np.clip(np.digitize(p, np.linspace(0, 1, nb + 1)[1:-1]), 0, nb - 1)
    t = 0.0
    for i in range(nb):
        m = b == i
        if m.sum():
            t += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(t)

def cal_fit(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(max_iter=2000, C=1e6).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

# ---------- 1 全指标 (Youden) ----------
print(SEP); print("=== 1. 全指标 (Youden 阈值) ===", flush=True)
print(f"{'gene':<8}{'多数类':>8}{'AUC':>8}{'ACC':>8}{'BA':>8}{'F1':>8}{'MCC':>8}{'SEN':>8}{'SPEC':>8}{'PPV':>8}{'NPV':>8}{'AUPRC':>8}{'基线':>8}")
rows = []
for j, g in enumerate(GENES):
    y, p = Y[:, j], P[:, j]
    tpr, fpr = roc(y, p)
    k = int(np.argmax(tpr - fpr)); thr = p[np.argsort(-p)[k]]
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    sen = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / len(y); ba = (sen + spe) / 2
    f1 = 2 * ppv * sen / max(ppv + sen, 1e-9)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    ap = float(np.mean([p[y == 1][i] for i in range(len(p[y == 1]))]))  # 占位
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y, p)
    prev = y.mean()
    print(f"{g:<8}{max(prev,1-prev):>8.3f}{auc(y,p):>8.3f}{acc:>8.3f}{ba:>8.3f}{f1:>8.3f}{mcc:>8.3f}"
          f"{sen:>8.3f}{spe:>8.3f}{ppv:>8.3f}{npv:>8.3f}{ap:>8.3f}{prev:>8.3f}")
    rows.append(dict(gene=g, n_pos=int(y.sum()), prev=float(prev), auc=auc(y, p), acc=acc, ba=ba,
                     f1=f1, mcc=mcc, sen=sen, spec=spe, ppv=ppv, npv=npv, auprc=ap))

# ---------- 2 操作点 ----------
print("\n" + SEP); print("=== 2. SEN @ SPEC 曲线 ===", flush=True)
SPECS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
print(f"{'gene':<8}" + "".join(f"{int(v*100)}%".rjust(9) for v in SPECS))
m2 = []
for j, g in enumerate(GENES):
    v = [sen_at_spec(Y[:, j], P[:, j], s) for s in SPECS]
    m2.append(v)
    print(f"{g:<8}" + "".join(f"{x:>9.3f}" for x in v))
print(f"{'mean':<8}" + "".join(f"{x:>9.3f}" for x in np.array(m2).mean(0)))

print(f"\n=== SPEC @ SEN 曲线 ===", flush=True)
SENS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
print(f"{'gene':<8}" + "".join(f"{int(v*100)}%".rjust(9) for v in SENS))
m3 = []
for j, g in enumerate(GENES):
    v = [spec_at_sen(Y[:, j], P[:, j], s) for s in SENS]
    m3.append(v)
    print(f"{g:<8}" + "".join(f"{x:>9.3f}" for x in v))
print(f"{'mean':<8}" + "".join(f"{x:>9.3f}" for x in np.array(m3).mean(0)))

print(f"\n=== 部署操作点全指标 ===", flush=True)
for s_t in [0.50, 0.70, 0.90]:
    print(f"\n--- 阈值定在 SPEC >= {s_t:.0%} ---")
    print(f"{'gene':<8}{'thr':>8}{'SEN':>8}{'SPEC':>8}{'PPV':>8}{'NPV':>8}{'每1000漏诊':>12}{'每1000可省':>12}")
    agg = []
    for j, g in enumerate(GENES):
        y, p = Y[:, j], P[:, j]
        t = thr_at_spec(y, p, s_t); pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        sen = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
        agg.append([sen, spe, ppv, npv])
        print(f"{g:<8}{t:>8.3f}{sen:>8.3f}{spe:>8.3f}{ppv:>8.3f}{npv:>8.3f}"
              f"{y.mean()*(1-sen)*1000:>12.0f}{y.mean()*sen*1000:>12.0f}")
    a = np.array(agg).mean(0)
    print(f"{'mean':<8}{'':>8}{a[0]:>8.3f}{a[1]:>8.3f}{a[2]:>8.3f}{a[3]:>8.3f}")

# ---------- 3 校准 ----------
print("\n" + SEP); print("=== 3. 校准 ===", flush=True)
print(f"{'gene':<8}{'prev':>8}{'mean_p':>9}{'Brier':>9}{'ECE':>8}{'slope':>8}{'intercept':>11}{'判定':>10}")
for j, g in enumerate(GENES):
    y, p = Y[:, j], P[:, j]
    sl, ic = cal_fit(y, p)
    ok = abs(sl - 1) < 0.25 and abs(ic) < 0.25
    v = "合格" if ok else ("过自信" if sl < 1 else "欠自信")
    print(f"{g:<8}{y.mean():>8.3f}{p.mean():>9.3f}{brier_score_loss(y,p):>9.3f}{ece(y,p):>8.3f}"
          f"{sl:>8.3f}{ic:>11.3f}{v:>10}")
    rows[j].update(brier=brier_score_loss(y, p), ece=ece(y, p), slope=sl, intercept=ic, cal_verdict=v)

# ---------- 4 子群条件 AUC ----------
print("\n" + SEP); print("=== 4. 子群条件 AUC ===", flush=True)
COLS = ["_PATIENT", "ajcc_pathologic_tumor_stage", "age_at_initial_pathologic_diagnosis", "gender", "race"]
gp = pd.read_parquet(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                  "provgigapath_embeddings_with_metadata.parquet"), columns=COLS)
tn = lambda b: ("-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None)
gp["case"] = gp["_PATIENT"].map(tn)
gg = gp.dropna(subset=["case"]).groupby("case").first()
for c in COLS[1:]:
    df[c] = df.case_id.map(gg[c]) if c in gg.columns else np.nan
# 分期必须按罗马数字精确判定: str.contains("Stage II") 会把 "Stage IIIA" 误判为早期
stg = df["ajcc_pathologic_tumor_stage"].astype(str)
roman = stg.str.extract(r"Stage ([IVX]+)")[0]
early = roman.isin(["I", "II"])
late = roman.isin(["III", "IV"])
print(f"[分期覆盖] I-II {int(early.sum())}  III-IV {int(late.sum())}  其他/缺失 {int((~(early|late)).sum())}")
subs = {"stage 早期(I-II)": early, "stage 晚期(III-IV)": late,
        "女性": df.gender.eq("FEMALE"), "男性": df.gender.eq("MALE"),
        "年龄<65": df.age_at_initial_pathologic_diagnosis.lt(65),
        "年龄>=65": df.age_at_initial_pathologic_diagnosis.ge(65)}
print(f"{'亚群':<20}{'n':>6}" + "".join(f"{g:>9}" for g in GENES) + f"{'极差':>9}{'最差':>8}")
for name, m in subs.items():
    m = m.values.astype(bool)
    a = [auc(Y[m, j], P[m, j]) if (m.sum() >= 20 and 0 < Y[m, j].sum() < m.sum()) else np.nan
         for j in range(len(GENES))]
    if np.isfinite(a).any():
        sp = np.nanmax(a) - np.nanmin(a)
        worst = GENES[int(np.nanargmin(a))]
    else:                                    # 该亚群里没有任何基因可评估
        sp, worst = np.nan, "-"
    print(f"{name:<20}{int(m.sum()):>6}" + "".join(f"{x:>9.3f}" if np.isfinite(x) else f"{'n/a':>9}" for x in a)
          + f"{sp:>9.3f}{worst:>8}")
print(f"{'全队列':<20}{N:>6}" + "".join(f"{auc(Y[:,j],P[:,j]):>9.3f}" for j in range(len(GENES)))
      + f"{np.ptp([auc(Y[:,j],P[:,j]) for j in range(len(GENES))]):>9.3f}" + f"{GENES[int(np.argmin([auc(Y[:,j],P[:,j]) for j in range(len(GENES))]))]:>8}")

pd.DataFrame(rows).to_csv(os.path.join(C, "final_metrics_best.csv"), index=False)
print(f"\n[saved] {C}\\final_metrics_best.csv")
print("DONE", flush=True)