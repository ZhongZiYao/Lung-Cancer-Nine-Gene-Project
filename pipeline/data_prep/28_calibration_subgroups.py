# -*- coding: utf-8 -*-
"""28_calibration_subgroups.py — 校准评估 + 子群条件指标

对应 Dawood et al. (Nat Biomed Eng 2026, doi 10.1038/s41551-026-01616-8) 议程第 5 条:
    "conditional metrics + subgroup calibration" —— 不看全队列平均 AUC,
    而看"给定临床阈值下, 每个亚群里还剩多少信号"。

AUC 是排序能力, 临床落地看的是: 阈值定在 SPEC=90% 时, 各亚组的 SEN/PPV 还剩多少。
同时审计 Dawood 用的那批病理捷径变量在公开 NSCLC 队列里到底存不存在。
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
SEP = "=" * 88

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
Y = df[GENES].values.astype(int)
L = np.load(os.path.join(C, "agg_gated_pure.npy"))   # 注意: 该文件存的是【logits】, 不是概率
# AUC / 排序指标对 sigmoid 单调变换不敏感, 但 Brier / ECE / 阈值 必须先过 sigmoid
P = 1.0 / (1.0 + np.exp(-L))
assert len(Y) == len(P), f"row mismatch {len(Y)} vs {len(P)}"
N = len(Y)
print(f"[LUAD n={N}] Prov-GigaPath 512patch + gated aggregator, 5-fold OOF")
print(SEP)

# ----------------------------------------------------------------------------
# 0) 临床元数据接入
# ----------------------------------------------------------------------------
COLS = ["_PATIENT", "age_at_initial_pathologic_diagnosis", "gender", "race",
        "ajcc_pathologic_tumor_stage", "clinical_stage", "histological_type",
        "histological_grade", "margin_status", "residual_tumor"]
gp = pd.read_parquet(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                  "provgigapath_embeddings_with_metadata.parquet"),
                     columns=COLS)
tn = lambda s: (lambda m: f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(
    re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(s)))
gp["case"] = gp["_PATIENT"].map(tn)
gg = gp.dropna(subset=["case"]).groupby("case").first()

BAD = {"", "nan", "[not available]", "[not evaluated]", "[unknown]", "[discrepancy]",
       "unknown", "not reported", "[not applicable]", "[not measured]",
       "[unknown]", "[pending]", "not applicable", "unknown; NOS"}
def clean(v):
    if isinstance(v, str):
        t = v.strip()
        if t.lower() in BAD:
            return np.nan
        return t
    return v

for c in COLS[1:]:
    df[c] = df.case_id.map(gg[c]) if c in gg.columns else pd.Series(np.nan, index=df.index)
    df[c] = df[c].map(clean)

print(f"队列 {N} 例, parquet 命中 {int(df.gender.notna().sum())} 例元数据\n")

print("=== 0. Dawood 捷径变量可用性审计 ===")
print("  Dawood 2026 的核心实验是'只用病理 grade 就能拿到 ~75% 准确率'。")
print("  在本公开 NSCLC 队列里逐个检查这些变量是否存在、是否有区分度:\n")
print(f"{'变量':<38}{'覆盖':>8}{'取值数':>8}   {'能否用于捷径审计':<18}")
rows_cov = []
for c in COLS[1:]:
    s = df[c]
    cov = int(s.notna().sum())
    nu = int(s.nunique(dropna=True))
    if cov == 0:
        verdict = "结构缺失(无数据)"
    elif nu <= 1:
        verdict = "无区分度(常量)"
    else:
        verdict = "可用"
    short = c if len(c) < 37 else c[:36]
    print(f"{short:<38}{cov:>4}/{N:<4}{nu:>8}   {verdict:<18}")
    rows_cov.append(dict(var=c, n_covered=cov, coverage=cov / N, n_unique=nu, verdict=verdict))

# ----------------------------------------------------------------------------
# 1) 临床捷径基线 (OOF, 只用可用的临床变量)
# ----------------------------------------------------------------------------
print("\n" + SEP)
print("=== 1. 临床变量基线 vs WSI 模型 (5-fold OOF LR) ===")
print("  可用变量仅 stage / age / sex / race。若临床基线已接近 WSI, 说明模型在抄近路。\n")

def oh(v):
    return pd.get_dummies(pd.Series(v).fillna("__NA__").astype(str)).values.astype(float)

AGE = df["age_at_initial_pathologic_diagnosis"].values.astype(float)
AGE = np.where(np.isfinite(AGE), AGE, np.nanmedian(AGE[np.isfinite(AGE)]))
AGE = ((AGE - np.nanmean(AGE)) / (np.nanstd(AGE) + 1e-9)).reshape(-1, 1)
STAGE = oh(df["ajcc_pathologic_tumor_stage"])
SEX = oh(df["gender"])
RACE = oh(df["race"])
CLIN = np.hstack([STAGE, SEX, RACE, AGE])
print(f"临床特征矩阵: {CLIN.shape[1]} 维 (stage={STAGE.shape[1]} sex={SEX.shape[1]} race={RACE.shape[1]} age=1)\n")

def oof_auc(X, y, seed=0):
    if not (0 < int(y.sum()) < len(y)):
        return np.nan
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, C=1.0).fit((X[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, oof)

print(f"{'gene':<8}{'n+':>5}{'prev':>8}{'临床基线':>10}{'WSI':>8}{'净信号':>10}")
clin_auc, wsi_auc = [], []
for j, g in enumerate(GENES):
    y = Y[:, j]
    a1 = oof_auc(CLIN, y)
    a2 = roc_auc_score(y, P[:, j]) if 0 < y.sum() < len(y) else np.nan
    clin_auc.append(a1); wsi_auc.append(a2)
    print(f"{g:<8}{int(y.sum()):>5}{y.mean():>8.3f}{a1:>10.3f}{a2:>8.3f}{a2 - a1:>+10.3f}")
print(f"{'mean':<8}{'':>5}{'':>8}{np.nanmean(clin_auc):>10.3f}{np.nanmean(wsi_auc):>8.3f}"
      f"{np.nanmean(wsi_auc) - np.nanmean(clin_auc):>+10.3f}")

# ----------------------------------------------------------------------------
# 2) 校准
# ----------------------------------------------------------------------------
print("\n" + SEP)
print("=== 2. 校准评估 ===")
print("  临床真正怕的不是 AUC 低, 是'模型说 0.8 风险时实际只有 0.4'。\n")

def ece(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    b = np.clip(np.digitize(p, bins[1:-1]), 0, nb - 1)
    tot = 0.0
    for i in range(nb):
        m = b == i
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)

def mce(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    b = np.clip(np.digitize(p, bins[1:-1]), 0, nb - 1)
    v = [abs(p[b == i].mean() - y[b == i].mean()) for i in range(nb) if (b == i).sum()]
    return float(max(v)) if v else 0.0

def cal_fit(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(max_iter=2000, C=1e6).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

print(f"{'gene':<8}{'prev':>8}{'mean_p':>9}{'Brier':>9}{'ECE':>8}{'MCE':>8}{'slope':>8}{'intercept':>11}{'校准判定':>12}")
cal_rows = []
for j, g in enumerate(GENES):
    y, p = Y[:, j], np.clip(P[:, j], 0, 1)
    br = brier_score_loss(y, p)
    e10, mx = ece(y, p), mce(y, p)
    sl, ic = cal_fit(y, p)
    ok = abs(sl - 1) < 0.25 and abs(ic) < 0.25
    cal_rows.append(dict(gene=g, prev=float(y.mean()), mean_pred=float(p.mean()), brier=float(br),
                         ece10=e10, mce10=mx, slope=sl, intercept=ic,
                         verdict="合格" if ok else ("过自信" if sl < 1 else "欠自信")))
    print(f"{g:<8}{y.mean():>8.3f}{p.mean():>9.3f}{br:>9.3f}{e10:>8.3f}{mx:>8.3f}{sl:>8.3f}{ic:>11.3f}"
          f"{('合格' if ok else ('过自信' if sl < 1 else '欠自信')):>12}")
mb = np.mean([r["brier"] for r in cal_rows]); me = np.mean([r["ece10"] for r in cal_rows])
msl = np.mean([r["slope"] for r in cal_rows]); mic = np.mean([r["intercept"] for r in cal_rows])
print(f"{'mean':<8}{'':>8}{'':>9}{mb:>9.3f}{me:>8.3f}{'':>8}{msl:>8.3f}{mic:>11.3f}")
print("\n可靠性曲线 (预测概率 -> 实际阳性率):")
for j, g in enumerate(GENES):
    y, p = Y[:, j], np.clip(P[:, j], 0, 1)
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    cells = []
    for i in range(10):
        m = b == i
        if m.sum() >= 5:
            cells.append(f"[{i/10:.1f}-{(i+1)/10:.1f}) n={m.sum():>3} 预测{p[m].mean():.2f} 实际{y[m].mean():.2f}")
    print(f"  {g:<7} " + (" | ".join(cells) if cells else "(预测概率分布过散, 无 >=5 例的箱)"))

# ----------------------------------------------------------------------------
# 3) 子群条件指标
# ----------------------------------------------------------------------------
print("\n" + SEP)
print("=== 3. 子群条件 AUC (条件指标) ===")
print("  同一模型在不同临床亚群里的排序能力。若某亚群掉到 0.5 以下, 部署即有害。\n")

stg = df["ajcc_pathologic_tumor_stage"].astype(str)
early = stg.str.contains(r"Stage I|Stage II", regex=True, na=False)
subs = {
    "stage 早期(I-II)": early,
    "stage 晚期(III-IV)": (~early) & stg.notna(),
    "女性": df["gender"].eq("FEMALE"),
    "男性": df["gender"].eq("MALE"),
    "年龄<65": df["age_at_initial_pathologic_diagnosis"].lt(65),
    "年龄>=65": df["age_at_initial_pathologic_diagnosis"].ge(65),
}
print(f"{'亚群':<20}{'n':>6}" + "".join(f"{g:>9}" for g in GENES) + f"{'极差':>9}{'最差基因':>11}")
sub_rows = []
for name, m in subs.items():
    m = m.values.astype(bool)
    aucs, ns = [], []
    for j, g in enumerate(GENES):
        y = Y[m, j]
        if len(y) < 20 or not (0 < y.sum() < len(y)):
            aucs.append(np.nan); ns.append(len(y))
        else:
            aucs.append(roc_auc_score(y, P[m, j])); ns.append(len(y))
    aucs = np.array(aucs, dtype=float)
    spread = np.nanmax(aucs) - np.nanmin(aucs) if np.isfinite(aucs).any() else np.nan
    worst = GENES[int(np.nanargmin(aucs))] if np.isfinite(aucs).any() else "-"
    print(f"{name:<20}{int(m.sum()):>6}" + "".join(f"{v:>9.3f}" for v in aucs) + f"{spread:>9.3f}{worst:>11}")
    for j, g in enumerate(GENES):
        sub_rows.append(dict(subgroup=name, n=int(m.sum()), n_pos=int(ns[j]), gene=g, auc=float(aucs[j])))
print(f"{'全队列':<20}{N:>6}" + "".join(f"{roc_auc_score(Y[:,j],P[:,j]):>9.3f}" for j in range(6))
      + f"{np.ptp([roc_auc_score(Y[:,j],P[:,j]) for j in range(6)]):>9.3f}{'':>11}")

# ----------------------------------------------------------------------------
# 4) 部署操作点: SPEC=90% 时的全指标
# ----------------------------------------------------------------------------
print("\n" + SEP)
print("=== 4. 部署操作点: 阈值定在不同特异度下 ===")
print("  分诊的真实问题是'愿意牺牲多少假阳, 换取多少真阳'。单一阈值无意义, 必须看整条曲线。\n")

def thr_at_spec(y, p, target_spec):
    # 返回"特异度 >= target_spec"前提下可用的最高阈值。ROC 上 fpr = 1 - spec。
    o = np.argsort(-p); ys = y[o]
    fpr = np.cumsum(1 - ys) / max(int((1 - ys).sum()), 1)
    ok = np.where(fpr <= (1.0 - target_spec))[0]
    if len(ok) == 0:
        return 1.0
    return float(p[o][ok[-1]])
SPEC_OP = [0.50, 0.70, 0.90]
op_rows = []
for s_t in SPEC_OP:
    print(f"--- 阈值设定为 SPEC >= {s_t:.0%} ---")
    print(f"{'gene':<8}{'prev':>8}{'thr':>8}{'SEN':>8}{'SPEC':>8}{'PPV':>8}{'NPV':>8}{'每1000例漏诊':>14}{'每1000例可省':>14}")
    for j, g in enumerate(GENES):
        y, p = Y[:, j], P[:, j]
        t = thr_at_spec(y, p, s_t)
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        sen = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
        print(f"{g:<8}{y.mean():>8.3f}{t:>8.3f}{sen:>8.3f}{spec:>8.3f}{ppv:>8.3f}{npv:>8.3f}"
              f"{y.mean()*(1-sen)*1000:>13.0f}{y.mean()*sen*1000:>14.0f}")
        op_rows.append(dict(spec_target=s_t, gene=g, prev=float(y.mean()), thr=float(t),
                            sen=float(sen), spec=float(spec), ppv=float(ppv), npv=float(npv),
                            missed_per_1000=float(y.mean()*(1-sen)*1000),
                            saved_per_1000=float(y.mean()*sen*1000)))
    ms = np.mean([[r["sen"], r["spec"], r["ppv"], r["npv"]]
                  for r in op_rows if r["spec_target"] == s_t], axis=0)
    print(f"{'mean':<8}{'':>8}{'':>8}{ms[0]:>8.3f}{ms[1]:>8.3f}{ms[2]:>8.3f}{ms[3]:>8.3f}\n")
# ----------------------------------------------------------------------------
out = os.path.join(C, "calibration_subgroup.csv")
pd.DataFrame(rows_cov).to_csv(out.replace(".csv", "_coverage.csv"), index=False)
pd.DataFrame(cal_rows).to_csv(out.replace(".csv", "_calibration.csv"), index=False)
pd.DataFrame(sub_rows).to_csv(out.replace(".csv", "_subgroup.csv"), index=False)
pd.DataFrame(op_rows).to_csv(out.replace(".csv", "_operating.csv"), index=False)
print(f"\n[saved] {C}\\calibration_subgroup_*.csv")
print("DONE")