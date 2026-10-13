# -*- coding: utf-8 -*-
"""53b_cptac_augmentation.py — CPTAC-LUAD 作为训练数据 (修正版)

上一版(53)的结构错误: train_mask = COH==0 意味着 held-out 全是 CPTAC,
TCGA 444 例一行都没被留出, 因此无法得到 TCGA 的 OOF 预测。整张表作废。

本版结构:
  A 基线   : 只在 TCGA-LUAD 444 上做 5-fold CV -> 444 例全部有 OOF
  B 增训   : 在 TCGA 444 + CPTAC 216 = 660 上做 5-fold CV -> 660 例全部有 OOF
             然后【分 cohort 报告】, 这样能直接看出 CPTAC 数据在 TCGA 上是帮忙还是拖累
  B 的 CPTAC 子集是 in-domain(进了训练集), 不是外验, 报告时必须标注
对照: A 应复现 49 号 uni_patch 的 0.7044 / 集成 0.7175 (434 例)
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
EP, BATCH, LR_LR, HID, S_INST = 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan

# ---------- 数据 ----------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
t = iu[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner")
t = t[t.project == "LUAD"].dropna(subset=["TP53"]).reset_index(drop=True)
XT = np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))[t.mem_row.values.astype(int)])
YT = t[GENES].values.astype(int)

fi = pd.read_csv(C + "/CPTAC_uni_sub512_index.csv")
fmap = dict(zip(fi.case_id, fi.mem_row))
meta = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
meta = meta[(meta.disease == "LUAD") & (meta.submitter_id.isin(fmap))].reset_index(drop=True)
XC = np.ascontiguousarray(np.load(os.path.join(C, "CPTAC_uni_sub512.npy"),
                                 mmap_mode="r")[[fmap[s] for s in meta.submitter_id]])
YC = meta[GENES].values.astype(int)
print("[data] TCGA-LUAD %d  CPTAC-LUAD %d" % (len(XT), len(XC)), flush=True)


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


def cv_oof(X, Y, seed):
    """在 X 上做标准 5-fold CV, 返回 (N, K) 的 OOF logits —— 每一例都被留出过一次"""
    N, S, D = X.shape
    oof = np.zeros((N, len(GENES)), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(D, HID, len(GENES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr)//BATCH))
        rng = np.random.RandomState(seed*100+f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        def gb(rr, r):
            bt = np.empty((len(rr), S, D), np.float16)
            for k2, rw in enumerate(rr):
                v = X[rw]; m = v.shape[0]
                bt[k2] = v[r.choice(m, S, replace=m < S)]
            return torch.from_numpy(bt).to(dev, torch.float32)
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm)-BATCH+1, BATCH):
                rr = perm[i:i+BATCH]
                yb = torch.from_numpy(Ytr[rr].astype(np.float32)).to(dev)
                loss = lf(net(gb(tr[rr], rng)), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            oof[te] = net(gb(te, np.random.RandomState(7))).float().cpu().numpy()
    return oof


def rep(tag, P, Y):
    per = [auc(Y[:, j], P[:, j]) for j in range(len(GENES))]
    ok = [v for v in per if v == v]
    print("  %-26s" % tag + "".join("%9.3f" % v for v in per)
          + "%10.4f  集成%9.4f" % (np.mean(ok) if ok else float("nan"),
                                    np.mean([auc(Y[:, j], P[:, j]) for j in range(len(GENES)) if auc(Y[:, j], P[:, j]) == auc(Y[:, j], P[:, j])])),
          flush=True)
    return per, (np.mean(ok) if ok else float("nan"))


print("\n" + "="*100)
print("=== A 基线: 只用 TCGA-LUAD 444 (5-fold OOF) ===")
print("="*100, flush=True)
oofA, ensA = [], []
for seed in SEEDS:
    p = cv_oof(XT, YT, seed); oofA.append(p); ensA.append(p)
    print("  seed %d mean=%.4f" % (seed, np.mean([auc(YT[:, j], p[:, j]) for j in range(6) if auc(YT[:, j], p[:, j]) == auc(YT[:, j], p[:, j])])), flush=True)
meanA = np.mean(oofA, 0)
rep("A 单seed均值", meanA, YT)
rep("A 5-seed集成", np.mean(ensA, 0), YT)

print("\n" + "="*100)
print("=== B 增训: TCGA 444 + CPTAC 216 = 660 (5-fold OOF, 分 cohort 报) ===")
print("="*100, flush=True)
X = np.concatenate([XT, XC], 0); Y = np.concatenate([YT, YC], 0)
NT = len(XT)
COH = np.array([0]*NT + [1]*len(XC))
oofB, ensB = [], []
for seed in SEEDS:
    p = cv_oof(X, Y, seed); oofB.append(p); ensB.append(p)
    print("  seed %d 完成" % seed, flush=True)
meanB = np.mean(oofB, 0); meanBe = np.mean(ensB, 0)
mT = COH == 0
mC = COH == 1
print()
rep("B TCGA 子集 单seed均值", meanB[mT], Y[mT])
rep("B TCGA 子集 5-seed集成", meanBe[mT], Y[mT])
rep("B CPTAC 子集 单seed均值", meanB[mC], Y[mC])
rep("B CPTAC 子集 5-seed集成", meanBe[mC], Y[mC])

print("\n对照 49 号: TCGA-only 内部 0.7044 / 集成 0.7175")
print("对照 32 号: CPTAC 纯迁移 0.6532 (纯外验)")
print("DONE", flush=True)