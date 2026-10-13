# -*- coding: utf-8 -*-
"""32_external_revalidated.py — 用重建标签 + 完整队列重跑外部验证

起因: 09 号脚本丢弃 35 例全野生型(真阴性缺失), 11 号脚本把未知癌种默认成 LUSC,
外验队列只剩 115 例 LUAD(实为 216), 报出 0.533 / 掉 0.205 的悲观结论。

约定: Xtr 是 memmap, rows_tr 是 case -> memmap 行 的映射; 折的划分作用在 case 轴。
协议声明: CPTAC 无 GigaPath 特征, 外验只能用 UNI v1(对应我们 UNI 内部水平),
不能与 GigaPath-512 的 0.737 直接比大小。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
S_INST, HID, EP, BATCH, LR, FOLDS = 512, 256, 25, 32, 3e-4, 5
dev = "cuda"

class Net(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))

def pack_row(v, rng):
    m = v.shape[0]
    return v[rng.choice(m, S_INST, replace=m < S_INST)]

def fit_predict(Xsrc, rows, Y, strata, Xte, dim, seed, n_folds=FOLDS):
    """在 rows 指定的病例上 5-fold 训练, 每折都对 Xte 推理后平均。"""
    K = Y.shape[1]
    out = np.zeros((len(Xte), K), np.float32)
    n = len(Y)
    # n_folds==1 表示"不划分, 直接在给定子集上训练" —— 供外层手工划分折时复用
    splits = ([(np.arange(n), np.arange(n))] if n_folds == 1
              else list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n), strata)))
    for f, (tr, va) in enumerate(splits):
        pw = torch.tensor(np.where(Y[tr].sum(0) > 0,
                                   (len(tr) - Y[tr].sum(0)) / np.maximum(Y[tr].sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(dim, HID, K).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EP, steps_per_epoch=max(1, max(1, len(tr)) // BATCH))
        rng = np.random.RandomState(seed * 1000 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = np.empty((len(rr), S_INST, dim), np.float16)
                for k2, ci in enumerate(rows[rr]):
                    bt[k2] = pack_row(Xsrc[ci], rng)
                bt = torch.from_numpy(bt).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            b = torch.from_numpy(Xte).to(dev, torch.float32)
            out += net(b).float().cpu().numpy() / n_folds
    return out

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan

ti = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
mm = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
tr = ti.merge(mm[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
tr = tr[tr.project == "LUAD"].reset_index(drop=True)
XT = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"), mmap_mode="r")
DIMT = XT.shape[2]
RW = tr.mem_row.values.astype(int)
YT = tr[GENES].values.astype(int)
strataT = np.clip(YT.sum(1), 0, 3).astype(int)
print(f"[train] TCGA-LUAD n={len(tr)}  feat={DIMT}", flush=True)

meta = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
fi = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
fmap = {c: r for c, r in zip(fi.case_id, fi.mem_row)}
meta["row"] = meta.submitter_id.map(lambda s: fmap.get(s, -1))
XE = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"), mmap_mode="r")
DIME = XE.shape[2]
print(f"[ext]   CPTAC feat={DIME}", flush=True)

def pack_ext(sub):
    rng = np.random.RandomState(0)
    X = np.stack([pack_row(np.asarray(XE[r], np.float32), rng) for r in sub.row.values])
    return X.astype(np.float16), sub[GENES].values.astype(int)

SEEDS = [0, 1, 2]
print("\n" + "=" * 88, flush=True)
print("=== 外部验证: TCGA-LUAD(UNI) 训练 -> CPTAC 测试, gated MIL, 3 seed ===", flush=True)
store = {}
for proj in ["LUAD", "LUSC"]:
    sub = meta[(meta.disease == proj) & (meta.row >= 0)].reset_index(drop=True)
    if len(sub) < 40:
        continue
    Xe, Ye = pack_ext(sub)
    accs = []
    for s in SEEDS:
        p = fit_predict(XT, RW, YT, strataT, Xe, DIMT, s)
        accs.append([auc(Ye[:, j], p[:, j]) for j in range(len(GENES))])
        if proj == "LUAD":
            store[s] = (Ye, p)
    a = np.array(accs)
    print(f"\n[CPTAC-{proj}] n={len(sub)}   " + "  ".join(f"{g}+{int(Ye[:,j].sum())}" for j, g in enumerate(GENES)), flush=True)
    for j, g in enumerate(GENES):
        if np.isfinite(a[:, j]).all():
            print(f"   {g:<7} 迁移 AUC = {a[:,j].mean():.3f} ± {a[:,j].std():.3f}")
        else:
            print(f"   {g:<7} 阳性不足, 不可评估")
    print(f"   {'mean':<7} {np.nanmean(a):.3f}")

# ---- CPTAC-LUAD 内部天花板: 同模型在 CPTAC 内部 OOF ----
sub = meta[(meta.disease == "LUAD") & (meta.row >= 0)].reset_index(drop=True)
Ye = sub[GENES].values.astype(int)
strataE = np.clip(Ye.sum(1), 0, 3).astype(int)
print("\n=== CPTAC-LUAD 内部 5-fold OOF (同模型, 作为迁移天花板) ===", flush=True)
rng = np.random.RandomState(0)
Xe_full = np.stack([pack_row(np.asarray(XE[r], np.float32), rng) for r in sub.row.values]).astype(np.float16)
oof_e = np.zeros((len(sub), len(GENES)), np.float32)
for f, (a_, b_) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(np.zeros(len(sub)), strataE)):
    Xtr_p = Xe_full[a_]; Xte_p = Xe_full[b_]
    oof_e[b_] = fit_predict(Xtr_p, np.arange(len(a_)), Ye[a_], strataE[a_], Xte_p, DIME, f, n_folds=1)
a_in = [auc(Ye[:, j], oof_e[:, j]) for j in range(len(GENES))]
print("   " + "  ".join(f"{g}={a:.3f}" for g, a in zip(GENES, a_in)) + f"   mean={np.nanmean(a_in):.3f}", flush=True)

# ---- TCGA-LUAD 内部同协议参照 ----
print("\n=== 参照: TCGA-LUAD 内部 5-fold OOF (同模型同特征) ===", flush=True)
oof_t = np.zeros((len(tr), len(GENES)), np.float32)
rng = np.random.RandomState(0)
Xt_full = np.stack([pack_row(np.asarray(XT[r], np.float32), rng) for r in RW]).astype(np.float16)
for f, (a_, b_) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(np.zeros(len(tr)), strataT)):
    oof_t[b_] = fit_predict(Xt_full[a_], np.arange(len(a_)), YT[a_], strataT[a_], Xt_full[b_], DIMT, f, n_folds=1)
a_it = [auc(YT[:, j], oof_t[:, j]) for j in range(len(GENES))]
print("   " + "  ".join(f"{g}={a:.3f}" for g, a in zip(GENES, a_it)) + f"   mean={np.nanmean(a_it):.3f}", flush=True)

print("\n" + "=" * 88)
print("=== 汇总 ===", flush=True)
tr_auc = []
for s in SEEDS:
    Ye_, p_ = store[s]
    tr_auc.append([auc(Ye_[:, j], p_[:, j]) for j in range(len(GENES))])
tr_auc = np.array(tr_auc).mean(0)
print(f"{'gene':<8}{'TCGA内':>10}{'CPTAC迁移':>11}{'Δ':>9}{'CPTAC内':>10}{'迁移损失':>10}")
for j, g in enumerate(GENES):
    if np.isfinite(tr_auc[j]):
        print(f"{g:<8}{a_it[j]:>10.3f}{tr_auc[j]:>11.3f}{tr_auc[j]-a_it[j]:>+9.3f}{a_in[j]:>10.3f}{a_in[j]-tr_auc[j]:>10.3f}")
print(f"{'mean':<8}{np.nanmean(a_it):>10.3f}{np.nanmean(tr_auc):>11.3f}"
      f"{np.nanmean(tr_auc)-np.nanmean(a_it):>+9.3f}{np.nanmean(a_in):>10.3f}{np.nanmean(a_in)-np.nanmean(tr_auc):>10.3f}")
np.save(os.path.join(C, "ext_reval_oof_cptac.npy"), oof_e)
np.save(os.path.join(C, "ext_reval_oof_tcga.npy"), oof_t)
print(f"\n[saved] {C}\\ext_reval_oof_cptac.npy / ext_reval_oof_tcga.npy")
print("DONE", flush=True)