# -*- coding: utf-8 -*-
"""31_cptac_diagnosis.py — 定位 CPTAC 外验掉点根因

调研发现的矛盾: 我们 CPTAC 0.533, 而公开 CPTAC-LUAD 线性探针约 0.74。
本脚本用三组对照把原因拆开:

  (A) 重建队列   修两个 bug: ①9 号脚本丢弃 13 例全野生型(真阴性缺失)
                        ②11 号脚本 dt.get(s,"LUSC") 把未知癌种静默默认成 LUSC
  (B) 内部自测   在 CPTAC 内部做 5-fold OOF —— 若也很低, 是标签/特征问题
  (C) 迁移对照   与 TCGA LUAD 内部同协议对比, 差值 = 域偏移成本

判读:
  B 高 + C 低  -> 纯域偏移/特征分布差, 方法学无解, 只能换编码器或做 stain 对齐
  B 低         -> 标签或特征本身坏了, 一切结论作废, 优先修数据
"""
import os, sys, re, json, glob, gzip
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
CT = os.path.join(R, "data", "CPTAC")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
POS = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
       "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}
UU = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)

man = json.load(open(os.path.join(CT, "cptac3_maf_manifest.json"), encoding="utf-8"))
fid2cases = {m["file_id"].lower(): m.get("case_ids", []) for m in man}
cm = {k.lower(): v for k, v in json.load(open(os.path.join(CT, "cptac_case_mapping.json"), encoding="utf-8")).items()}

# ---------- (A) 重建标签: 每个有 MAF 的 submitter 都要出现, 全野生型 = 全 0 ----------
lab = {s: {g: 0 for g in GENES} for s in []}
allsub = {}
for p in sorted(glob.glob(os.path.join(CT, "cptac_maf", "*.maf.gz"))):
    fid = UU.search(os.path.basename(p)).group(1).lower()
    cu = fid2cases.get(fid) or []
    subs = {cm[u.lower()]["submitter_id"] for u in cu
            if u.lower() in cm and cm[u.lower()].get("submitter_id")}
    if not subs:
        continue
    for s in subs:
        allsub.setdefault(s, set()).add(p)
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            hdr = line.rstrip("\n").split("\t"); break
    ci = {c: i for i, c in enumerate(hdr)}
    if "Hugo_Symbol" not in ci or "Variant_Classification" not in ci:
        continue
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < len(hdr):
                continue
            g = c[ci["Hugo_Symbol"]]
            if g not in GENES or c[ci["Variant_Classification"]] not in POS:
                continue
            for s in subs:
                lab.setdefault(s, {x: 0 for x in GENES})[g] = 1

for s in allsub:                      # 有 MAF 但无目标基因突变 -> 全 0 (真阴性)
    lab.setdefault(s, {g: 0 for g in GENES})

# 癌种: 只用 mapping 里明确写了 adenocarcinoma / squamous 的, 其余记 unknown
def dtype_of(s):
    for u, v in cm.items():
        if v.get("submitter_id") == s:
            d = (v.get("disease_type") or "").lower()
            if "adenocarcinoma" in d:
                return "LUAD"
            if "squamous" in d:
                return "LUSC"
            return "unknown"
    return "unknown"

rows = []
for s in sorted(allsub):
    rows.append(dict(submitter_id=s, disease=dtype_of(s), n_maf=len(allsub[s]),
                     **{g: lab[s][g] for g in GENES}))
meta = pd.DataFrame(rows)
print("=== (A) 重建 CPTAC 队列 (含真阴性) ===")
print(f"  有 MAF 的 submitter: {len(meta)}")
print(f"  癌种分布: {meta.disease.value_counts().to_dict()}")
print(f"  全野生型(10 基因全 0): {int((meta[GENES].sum(1)==0).sum())} 例   <== 09 号脚本把这类全丢了")
old = pd.read_csv(os.path.join(R, "data", "labels", "cptac_gene_labels.csv"))
print(f"  旧标签表 {len(old)} 例  ->  新标签表 {len(meta)} 例 (差 {len(meta)-len(old)})")
for proj in ["LUAD", "LUSC", "unknown"]:
    sub = meta[meta.disease == proj]
    if len(sub) == 0:
        continue
    print(f"    {proj:<8} n={len(sub):<4} prev: " + " ".join(f"{g}={sub[g].mean():.2f}" for g in GENES))

# ---------- 特征 join ----------
fi = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
XE = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"), mmap_mode="r")
fmap = {c: r for c, r in zip(fi.case_id, fi.mem_row)}
meta["feat"] = meta.submitter_id.map(lambda s: fmap.get(s, -1))
ev = meta[(meta.feat >= 0) & (meta.disease.isin(["LUAD", "LUSC"]))].reset_index(drop=True)
print(f"\n  癌种明确 + 有 UNI 特征 = {len(ev)} 例 (LUAD {int((ev.disease=='LUAD').sum())} / LUSC {int((ev.disease=='LUSC').sum())})")
for proj in ["LUAD", "LUSC"]:
    sub = ev[ev.disease == proj]
    if len(sub):
        print(f"    {proj} prev: " + " ".join(f"{g}={int(sub[g].sum())}" for g in GENES))

# ---------- (B) CPTAC 内部自测 + (C) TCGA 内部对照 ----------
def pool(rows_idx):
    out = np.zeros((len(rows_idx), XE.shape[2]), np.float32)
    for i, r in enumerate(rows_idx):
        out[i] = np.asarray(XE[r], np.float32).mean(0)
    return out

def oof_auc(X, y, seed=0):
    # 阳性数太少时 5-fold 分层会崩, 直接判为不可评估
    if not (10 <= int(y.sum()) < len(y) - 10):
        return np.nan
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        o[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, o)

for proj in ["LUAD", "LUSC", None]:
    sub = ev if proj is None else ev[ev.disease == proj].reset_index(drop=True)
    tag = proj or "ALL"
    if len(sub) < 40:
        continue
    Xp = pool(sub.feat.values)
    Yp = sub[GENES].values.astype(int)
    aucs = [oof_auc(Xp, Yp[:, j]) for j in range(len(GENES))]
    print(f"\n[B] CPTAC-{tag} 内部 5-fold OOF (mean-pool + LR, n={len(sub)})")
    print("   " + "".join(f"{g}={a:.3f}  " for g, a in zip(GENES, aucs)) + f" mean={np.nanmean(aucs):.3f}")

# TCGA LUAD 同协议
ti = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
mm = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
tt = ti.merge(mm[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
tt = tt[tt.project == "LUAD"].reset_index(drop=True)
XT = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"), mmap_mode="r")
Xtp = np.zeros((len(tt), XT.shape[2]), np.float32)
for i, r in enumerate(tt.mem_row.values):
    Xtp[i] = np.asarray(XT[r], np.float32).mean(0)
Yt = tt[GENES].values.astype(int)
aucs_t = [oof_auc(Xtp, Yt[:, j]) for j in range(len(GENES))]
print(f"\n[C] TCGA-LUAD 内部同协议 (mean-pool + LR, n={len(tt)})")
print("   " + "".join(f"{g}={a:.3f}  " for g, a in zip(GENES, aucs_t)) + f" mean={np.nanmean(aucs_t):.3f}")

# ---------- 跨队列迁移: TCGA 训练 -> CPTAC 测试 ----------
print("\n=== (D) 跨队列迁移 (TCGA-LUAD 训练 -> CPTAC 测试, 完全同一特征空间) ===")
evL = ev[ev.disease == "LUAD"].reset_index(drop=True)
XpL = pool(evL.feat.values)
YpL = evL[GENES].values.astype(int)
print(f"{'gene':<8}{'TCGA内':>9}{'CPTAC迁移':>11}{'Δ':>9}{'CPTAC内':>10}{'迁移损失':>10}")
cl = [oof_auc(Xtp, Yt[:, j]) for j in range(len(GENES))]
cl_in = [oof_auc(XpL, YpL[:, j]) for j in range(len(GENES))]
tr_ = []
for j, g in enumerate(GENES):
    sc = StandardScaler().fit(Xtp)
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtp), Yt[:, j])
    p = m.predict_proba(sc.transform(XpL))[:, 1]
    tr_ = roc_auc_score(YpL[:, j], p) if 10 <= YpL[:, j].sum() < len(YpL) - 10 else np.nan
    print(f"{g:<8}{cl[j]:>9.3f}{tr_:>11.3f}{tr_-cl[j]:>+9.3f}{cl_in[j]:>10.3f}{cl_in[j]-tr_:>10.3f}")
print(f"{'mean':<8}{np.nanmean(cl):>9.3f}{np.nanmean(tr_):>11.3f}"
      f"{np.nanmean(tr_)-np.nanmean(cl):>+9.3f}{np.nanmean(cl_in):>10.3f}"
      f"{np.nanmean(cl_in)-np.nanmean(tr_):>10.3f}")

meta.drop(columns=["feat"]).to_csv(os.path.join(C, "cptac_labels_rebuilt.csv"), index=False)
print(f"\n[saved] {C}\\cptac_labels_rebuilt.csv")
print("DONE")