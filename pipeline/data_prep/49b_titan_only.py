# -*- coding: utf-8 -*-
"""49b_titan_only.py — 补跑 49 号的 titan1 分支 (S=1, 退化为 MLP)

49 号主脚本在 S==1 时写成 v[r] (r 是 RandomState 而非下标) 崩在最后一个源。
前三个源已出结果, 这里只补 titan1, 协议与其余三个完全一致。
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
M = np.stack([T[c] for c in df.case_id])[:, None, :].astype(np.float16)
print("[data] titan1 %s  n=%d" % (M.shape, N), flush=True)


def lp(X, y):
    X = X.astype(np.float32)
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        o[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, o)


a_lp = np.mean([lp(M[:, 0, :], Y[:, j]) for j in range(K)])
print("[preflight] 线性探针 mean AUC = %.4f" % a_lp, flush=True)
assert a_lp > 0.55, "映射可疑"


class Net(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)

    def forward(s, x):
        z = s.trunk(x)
        # 袋子里只有 1 个实例时注意力在数学上无定义(softmax 轴长度为 1),
        # 此时等价于一个普通 MLP —— 这正是 TITAN 单向量作为教师时的真实形态。
        if z.dim() == 2:          # 只有一个实例, z 是 (B,H), 注意力轴不存在
            return s.od(z)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def run(seed):
    oof = np.zeros((N, K), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(M.shape[2], HID, K).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = torch.from_numpy(M[tr[rr], 0, :]).to(dev, torch.float32)
                yb = torch.from_numpy(Ytr[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            oof[te] = net(torch.from_numpy(M[te, 0, :]).to(dev, torch.float32)).float().cpu().numpy()
    return oof


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
res, store = [], []
for seed in SEEDS:
    op = run(seed)
    res.append([auc(Y[:, j], op[:, j]) for j in range(K)])
    store.append(op)
    print("  seed %d  mean=%.4f" % (seed, np.mean(res[-1])), flush=True)
res = np.array(res)
ens = np.mean(store, axis=0)
a_ens = [auc(Y[:, j], ens[:, j]) for j in range(K)]
print(chr(10) + "titan1  mean=%.4f  ens=%.4f  (线性探针 %.4f)" % (res.mean(0).mean(), np.mean(a_ens), a_lp), flush=True)
print("逐基因: " + "  ".join("%s=%.3f" % (g, a) for g, a in zip(GENES, res.mean(0))), flush=True)
np.save(os.path.join(C, "oof_titan1_5seed.npy"), ens)
print("DONE", flush=True)