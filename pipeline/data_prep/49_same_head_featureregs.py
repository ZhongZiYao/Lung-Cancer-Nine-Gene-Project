# -*- coding: utf-8 -*-
"""49_same_head_featureregs.py — 同一个 MIL 头, 换特征源

之前的实验缺口: 教师(TITAN / GP-slide)只用 PCA128+LR 线性探针评估, 而学生是
526K 参数端到端 MIL。两者读出不同档, "教师弱于学生"有一部分是探针弱造成的。
本脚本用【完全相同】的 gated MIL 头(25 epoch / 5 seed / 同一折划分)评估 4 种
特征源, 顺带回答"袋子里放多少实例才够":

  giga_patch  Prov-GigaPath patch  512 实例 x 1536 维   (主线)
  uni_patch   UNI v1 patch         512 实例 x 1024 维
  gp14        GigaPath slide 14 层   14 实例 x  768 维   (真正的 MIL)
  titan1      TITAN slide            1 实例 x  768 维   (退化为 MLP)

  bag size: 512 -> 512 -> 14 -> 1
若 titan1/gp14 在同头下明显高于线性探针, 说明"教师弱"是读出造成的;
若仍低于 giga_patch, 则确认 patch 级特征本身才是天花板。
"""
import os, sys, re, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
EP, BATCH, LR_LR, HID = 25, 32, 3e-4, 256
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
tn = lambda b: "-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
ig = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = ig[["case_id", "mem_row"]].rename(columns={"mem_row": "rg"})
df = df.merge(iu[["case_id", "mem_row"]].rename(columns={"mem_row": "ru"}), on="case_id", how="inner")
df = df.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)

tk = pickle.load(open(os.path.join(R, "models", "TITAN", "TCGA_TITAN_features.pkl"), "rb"))
T = {}
for f, e in zip(tk["filenames"], tk["embeddings"]):
    c = tn(os.path.basename(f))
    if c and c not in T:
        T[c] = np.asarray(e, np.float32)
E14 = np.load(os.path.join(C, "NSCLC_gigap_slide_L14.npy"))
ids14 = pd.read_csv(os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")).case_id.tolist()
m14 = {c: i for i, c in enumerate(ids14)}
df = df[df.case_id.isin(T) & df.case_id.isin(m14)].reset_index(drop=True)
N, K = len(df), len(GENES)
Y = df[GENES].values.astype(int)

SRC = {
    "giga_patch": np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))[df.rg.values.astype(int)]),
    "uni_patch": np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))[df.ru.values.astype(int)]),
    "gp14": E14[[m14[c] for c in df.case_id]].astype(np.float16),
    "titan1": np.stack([T[c] for c in df.case_id])[:, None, :].astype(np.float16),
}
print("[data] LUAD n=%d  genes=%d" % (N, K), flush=True)
for k, v in SRC.items():
    print("   %-11s %s   bag=%d" % (k, v.shape, v.shape[1]), flush=True)


def lp(M, y):
    X = M.astype(np.float32).mean(1)
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        o[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, o)


print(chr(10) + "[preflight] 线性探针 TP53 AUC (与 44 号对照):", flush=True)
for k, v in SRC.items():
    a = lp(v, Y[:, 1])
    print("   %-11s %.3f" % (k, a), flush=True)
    assert a > 0.55, "%s 映射可疑 (%.3f)" % (k, a)


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


def run(M, seed):
    S, D = M.shape[1], M.shape[2]
    oof = np.zeros((N, K), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)

    def gb(rows, r):
        bt = np.empty((len(rows), S, D), np.float16)
        for i, rw in enumerate(rows):
            v = M[rw]; m = v.shape[0]
            # S==1 时只有一个实例, 直接取第 0 行(v[r] 里 r 是 RandomState, 不是下标)
            bt[i] = v[0] if S == 1 else v[r.choice(m, S, replace=m < S)]
        return torch.from_numpy(bt).to(dev, torch.float32)

    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(D, HID, K).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
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


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
res, store = {}, {}
for name, M in SRC.items():
    res[name] = np.zeros((len(SEEDS), K), np.float32)
    store[name] = []
    for seed in SEEDS:
        op = run(M, seed)
        res[name][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[name].append(op)
    mm = np.mean(res[name].mean(0))
    ens = np.mean(store[name], axis=0)
    ae = np.mean([auc(Y[:, j], ens[:, j]) for j in range(K)])
    print("  %-11s mean=%.4f  ens=%.4f" % (name, mm, ae), flush=True)

print(chr(10) + "=" * 100)
print("=== 同一个 gated MIL 头, 4 种特征源 (LUAD n=%d, 5 seed) ===" % N)
print("=" * 100)
base_s = res["giga_patch"].mean(1)
hdr = "%-12s%6s" % ("source", "bag") + "".join("%9s" % g for g in GENES)
hdr += "%10s%11s%7s%10s" % ("mean", "delta", "same", "ens")
print(hdr, flush=True)
summary = []
for name, M in SRC.items():
    mm = res[name].mean(0); d = res[name].mean(1) - base_s
    ens = np.mean(store[name], axis=0)
    a_ens = np.mean([auc(Y[:, j], ens[:, j]) for j in range(K)])
    row = "%-12s%6d" % (name, M.shape[1]) + "".join("%9.3f" % v for v in mm)
    row += "%10.4f%11.4f%6d/5%10.4f" % (np.mean(mm), d.mean(), int((d > 0).sum()), a_ens)
    print(row, flush=True)
    summary.append(dict(source=name, bag=int(M.shape[1]), mean_auc=float(np.mean(mm)),
                        delta=float(d.mean()), agree=int((d > 0).sum()), ens_auc=float(a_ens)))

print(chr(10) + "=== 同头 MIL vs 线性探针: 读出差异是否解释弱教师 ===")
print("%-12s%11s%10s%10s" % ("source", "linear", "mil", "gain"))
for name, M in SRC.items():
    a_lp = np.mean([lp(M, Y[:, j]) for j in range(K)])
    a_mil = float(np.mean(res[name].mean(0)))
    print("%-12s%11.4f%10.4f%+10.4f" % (name, a_lp, a_mil, a_mil - a_lp), flush=True)
pd.DataFrame(summary).to_csv(os.path.join(C, "same_head_featureregs.csv"), index=False)
print(chr(10) + "[saved] %s" % os.path.join(C, "same_head_featureregs.csv"))
print("DONE", flush=True)