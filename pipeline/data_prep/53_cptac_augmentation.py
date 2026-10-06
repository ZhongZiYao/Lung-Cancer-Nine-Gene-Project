# -*- coding: utf-8 -*-
"""53_cptac_augmentation.py — 把 CPTAC-LUAD 用作训练数据 (最后一条未试的杠杆)

背景: 22 项实验里, 表征/架构/损失/融合全部无效或有害, 唯一有效的是种子集成(+0.0124)。
      UNI2-h 配对对比也输了(-0.0166)。所有"换更好的表征"的路已排除。
      剩下唯一还有空间的杠杆是【数据量】: 我们有 216 例 CPTAC-LUAD,
      因标签 bug 一直只当外部验证集, 从未用于训练。+50% 数据。

协议(关键: 不能重复 46 号那个标签泄漏):
  把 TCGA-LUAD 444 + CPTAC-LUAD 216 = 660 例合并,
  5-fold CV 时 **同一 cohort 的病例会被分到不同折**,
  评估时【分 cohort 报告】, 这样能看出:
    - TCGA 子集上: 加了 CPTAC 数据是帮助还是伤害(域偏移 vs 正则)
    - CPTAC 子集上: 从纯迁移(0.653)变成域内训练能到多少
  对照基线: TCGA-only 内部 0.7044 / 5-seed 集成 0.7175(49 号 uni_patch)
            CPTAC 纯迁移 0.6532(32 号)

  注意这不是"外验"了 —— CPTAC 进了训练集, 所以 CPTAC 子集的数字是 in-domain。
  真正的外验只能等下一个队列。报告时必须标注这一点。
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

# ---------- TCGA-LUAD (UNI v1) ----------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
t = iu[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner")
t = t[(t.project == "LUAD")].dropna(subset=["TP53"]).reset_index(drop=True)
XT = np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))[t.mem_row.values.astype(int)])
YT = t[GENES].values.astype(int)
NT = len(t)

# ---------- CPTAC-LUAD (UNI v1, 重建标签) ----------
meta = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
fmap = {c: r for c, r in zip(pd.read_csv(C + "/CPTAC_uni_sub512_index.csv").case_id,
                            pd.read_csv(C + "/CPTAC_uni_sub512_index.csv").mem_row)}
meta = meta[(meta.disease == "LUAD") & (meta.submitter_id.isin(fmap))].reset_index(drop=True)
XC = np.ascontiguousarray(np.load(os.path.join(C, "CPTAC_uni_sub512.npy"),
                                 mmap_mode="r")[[fmap[s] for s in meta.submitter_id]])
YC = meta[GENES].values.astype(int)
NC = len(meta)
print("[data] TCGA-LUAD %d  CPTAC-LUAD %d  合计 %d" % (NT, NC, NT + NC), flush=True)
for g, v in zip(GENES, YT.mean(0)):
    print("   TCGA %-7s prev=%.3f n+=%d" % (g, v, int((YT[:, GENES.index(g)] == 1).sum())), flush=True)
for g, v in zip(GENES, YC.mean(0)):
    print("   CPTAC %-7s prev=%.3f n+=%d" % (g, v, int((YC[:, GENES.index(g)] == 1).sum())), flush=True)

X = np.concatenate([XT, XC], 0)
Y = np.concatenate([YT, YC], 0)
N = len(X)
COH = np.array([0] * NT + [1] * NC)      # 0=TCGA 1=CPTAC
print("[data] 合并 %s  cohort: TCGA %d / CPTAC %d" % (X.shape, int((COH == 0).sum()), int((COH == 1).sum())), flush=True)


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


def run(train_mask, seed):
    """train_mask: 哪些行参与训练。返回 (n_test, K) logits 与对应行号"""
    tr_all = np.flatnonzero(train_mask)
    te_all = np.flatnonzero(~train_mask)
    S, D = X.shape[1], X.shape[2]
    # oof 按训练侧行数分配, 用折内位置 b 索引, 最后再映射回全局行号
    oof = np.zeros((len(tr_all), len(GENES)), np.float32)
    strata = np.clip(Y[tr_all].sum(1), 0, 3).astype(int)
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr_all)), strata)):
        tr = tr_all[a]; te = tr_all[b]
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(D, HID, len(GENES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr)//BATCH))
        rng = np.random.RandomState(seed*100+f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        def gb(rows, r):
            bt = np.empty((len(rows), S, D), np.float16)
            for k2, rw in enumerate(rows):
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
            oof[b] = net(gb(te, np.random.RandomState(7))).float().cpu().numpy()
    return tr_all[b], oof[b]


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y)-10 else np.nan

res = {}
for name, tm in [("tcga_only", COH == 0), ("tcga_plus_cptac", np.ones(N, bool))]:
    per_coh = {0: np.zeros((len(SEEDS), len(GENES)), np.float32),
               1: np.zeros((len(SEEDS), len(GENES)), np.float32)}
    store = {0: [], 1: []}
    for seed in SEEDS:
        te, oof = run(tm, seed)
        for c in (0, 1):
            m = COH[te] == c
            if m.sum() == 0:
                continue
            per_coh[c][seed] = [auc(Y[te][m, j], oof[m, j]) for j in range(len(GENES))]
            store[c].append((te[m], oof[m]))
    res[name] = per_coh
    print("  [%s] 完成" % name, flush=True)

print("\n" + "="*100)
print("=== CPTAC-LUAD 作为训练数据 (merged 660 例, 5-fold, 分 cohort 报告) ===")
print("="*100)
for name in ["tcga_only", "tcga_plus_cptac"]:
    print("\n[%s]" % name, flush=True)
    for c, cn in [(0, "TCGA 子集(域内)"), (1, "CPTAC 子集")]:
        r = res[name][c]
        if r.mean() == 0 and r.max() == 0:
            continue
        m = r.mean(0)
        ens = None
        try:
            ens = np.mean([o[1] for o in store[c]], axis=0)
            a_ens = np.mean([auc(np.concatenate([o[0] for o in store[c]])[:, 0] * 0 + Y[np.concatenate([o[0] for o in store[c]]), j], ens[:, j])
                             for j in range(len(GENES))
                             if 10 <= Y[np.concatenate([o[0] for o in store[c]]), j].sum() <= len(ens) - 10])
        except Exception:
            a_ens = float("nan")
        print("  %-16s" % cn + "".join("%9.3f" % v for v in m)
              + "%10.4f  集成%9.4f" % (np.mean(m), a_ens), flush=True)

print("\n对照基线:")
print("  TCGA-only 内部 0.7044 / 集成 0.7175  (49 号 uni_patch, 434 例)")
print("  CPTAC 纯迁移 0.6532                (32 号, 216 例)", flush=True)
print("DONE", flush=True)