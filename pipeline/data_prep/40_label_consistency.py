# -*- coding: utf-8 -*-
"""40_label_consistency.py — 标签一致性审计 (跨源交叉验证)

依据: MDPI Computers 2025 报告, 剔除跨源不一致标签后 AUC 0.855 -> 0.928 (+0.073),
是本轮文献调研里单项增益最大的数字。我们从没用独立来源校验过 MC3 标签。

做法:
  1. 从 cBioPortal API 拉 luad_tcga 的 6 个基因突变 (独立于 MC3 的来源)
  2. 与 MC3 标签逐 case 逐基因比对, 算一致率
  3. 在「一致子集」和「不一致子集」上分别评估本模型的 OOF AUC
  4. 若不一致标签确实是噪声源, 不一致子集上的 AUC 应显著更低
"""
import os, sys, json, time
from urllib.request import Request, urlopen
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
ENTREZ = {"EGFR": 1956, "KRAS": 3845, "TP53": 7157, "KEAP1": 9817, "NF1": 4763, "STK11": 6794}
API = "https://www.cbioportal.org/api"
SEP = "=" * 96

def get(url, data=None, tries=3):
    for i in range(tries):
        try:
            req = Request(url, data=json.dumps(data).encode() if data else None,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
            with urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                print(f"  [warn] {url[:90]} -> {e}", flush=True)
                return None
            time.sleep(2 * (i + 1))

# ---- 1) 拉 cBioPortal 突变 ----
print("[1] 从 cBioPortal 拉取 luad_tcga 突变 ...", flush=True)
mp = get(f"{API}/studies/luad_tcga/molecular-profiles")
if mp:
    muts = [m["molecularProfileId"] for m in mp if "mutation" in m["molecularProfileId"].lower()]
    print(f"    可用突变 profile: {muts}", flush=True)
    MPROF = "luad_tcga_mutations" if "luad_tcga_mutations" in muts else (muts[0] if muts else None)
else:
    MPROF = None

cbio = {}
if MPROF:
    # 逐基因取: projection 参数对 mutations 端点无效, 带上会 400
    e2g = {v: k for k, v in ENTREZ.items()}
    nrec = 0
    for gname, eid in ENTREZ.items():
        d = get(f"{API}/molecular-profiles/{MPROF}/mutations/fetch",
                {"sampleListId": "luad_tcga_all", "entrezGeneIds": [eid]})
        if not d:
            continue
        nrec += len(d)
        for m in d:
            p = m.get("patientId")
            if p:
                cbio.setdefault(p, set()).add(gname)
    print(f"    原始突变记录 {nrec} 条, 覆盖 case {len(cbio)} 个", flush=True)

# ---- 2) 与 MC3 比对 ----
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = idx[["case_id", "mem_row"]].merge(iu[["case_id"]], on="case_id", how="inner")
df = df.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df)
print(f"\n[2] 与 MC3 逐基因比对 (LUAD n={N}, cBioPortal 有数据的 case {len(set(df.case_id) & set(cbio))})", flush=True)
Y = df[GENES].values.astype(int)
COV = df.case_id.isin(cbio).values          # cBioPortal 真正覆盖的 case
NCOV = int(COV.sum())
print(f"    >>> 比对只在 cBioPortal 覆盖的 {NCOV} 例上进行(未覆盖不等于阴性)", flush=True)
P = 1.0 / (1.0 + np.exp(-np.load(os.path.join(C, "oof_giga_5seed.npy"))))

have = df.case_id.isin(cbio).values
print(f"{'gene':<8}{'MC3+':>7}{'cBio+':>8}{'共有':>7}{'MC3独有':>9}{'cBio独有':>9}{'两阴一致':>10}{'一致率':>9}")
agree_mask = np.ones((NCOV, len(GENES)), bool)
for j, g in enumerate(GENES):
    cbp = np.array([1 if g in cbio.get(c, set()) else 0 for c in df.case_id])[COV]
    m3 = Y[COV, j].astype(bool)
    cbp = cbp.astype(bool)
    a = agree_mask[:, j] = (m3 == cbp)
    both = int((m3 & cbp).sum()); only3 = int((m3 & ~cbp).sum())
    onlyb = int((~m3 & cbp).sum()); neither = int((~m3 & ~cbp).sum())
    print(f"{g:<8}{int(m3.sum()):>7}{int(cbp.sum()):>8}{both:>7}{only3:>9}{onlyb:>9}{neither:>10}{a.mean():>9.3f}")
print(f"\n逐基因平均一致率(仅覆盖 case): {agree_mask.mean():.4f}")
print(f"覆盖 case 中 6 基因全一致: {int(agree_mask.all(1).sum())}/{NCOV} = {agree_mask.all(1).mean():.3f}")

# ---- 3) 分层评估 ----
print("\n" + SEP, flush=True)
print("[3] 在一致子集 / 不一致 case 上分别评估 OOF AUC", flush=True)
full = agree_mask.all(1)          # 已限定在覆盖 case 内
disc = ~full
auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
Yc, Pc = Y[COV], P[COV]                 # 先切到覆盖子集, 再用 full/disc 索引
print(f"(仅在 cBioPortal 覆盖的 {NCOV} 例内分层; 全一致 {int(full.sum())} 例, 有分歧 {int(disc.sum())} 例)", flush=True)
print(f"{'gene':<8}{'覆盖全体':>10}{'全一致':>12}{'有分歧':>12}{'差':>9}")
for j, g in enumerate(GENES):
    a_all = auc(Yc[:, j], Pc[:, j])
    a_ok = auc(Yc[full, j], Pc[full, j]) if full.sum() >= 20 and 0 < Yc[full, j].sum() < full.sum() else np.nan
    a_no = auc(Yc[disc, j], Pc[disc, j]) if disc.sum() >= 20 and 0 < Yc[disc, j].sum() < disc.sum() else np.nan
    d = a_ok - a_no if np.isfinite(a_ok) and np.isfinite(a_no) else np.nan
    print(f"{g:<8}{a_all:>10.3f}{a_ok:>12.3f}{a_no:>12.3f}{d:>+9.3f}")
mk = [auc(Yc[full, j], Pc[full, j]) for j in range(len(GENES))]
mo = [auc(Yc[disc, j], Pc[disc, j]) for j in range(len(GENES))]
ma = [auc(Yc[:, j], Pc[:, j]) for j in range(len(GENES))]
print(f"{'mean':<8}{np.nanmean(ma):>10.3f}{np.nanmean(mk):>12.3f}{np.nanmean(mo):>12.3f}{np.nanmean(mk)-np.nanmean(mo):>+9.3f}")

covpos = np.flatnonzero(COV)          # 覆盖 case 在 0..N-1 中的位置
_rows = []
for i in range(N):
    k = int(np.flatnonzero(covpos == i)[0]) if COV[i] else -1
    _rows.append(dict(case_id=df.case_id.iloc[i], covered=bool(COV[i]),
                      n_disagree=int((~agree_mask[k]).sum()) if COV[i] else -1,
                      **{g: int(Y[i, j]) for j, g in enumerate(GENES)}))
pd.DataFrame(_rows).to_csv(os.path.join(C, "label_consistency.csv"), index=False)
print(f"\n[saved] {C}\\label_consistency.csv")
print("DONE", flush=True)