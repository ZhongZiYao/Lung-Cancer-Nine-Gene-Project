# -*- coding: utf-8 -*-
"""41_joint_vs_sep.py — 联合训练 vs 分基因独立训练 + epoch 扫描

用户的疑问: 既然有 6 个基因, 干脆分开训 6 个模型, 推理时分别输入, 效果会不会更好?

这背后是一个真实的结构性问题: 当前模型共享 trunk, 而共突变是已知的混杂源。
共享 trunk 可能让基因 A 的预测借道基因 B 的共突变结构 —— 那就是"隐式共突变泄漏"。
分开训练没有共享层, 天然免疫。代价是失去多任务正则。

同时扫描 epoch: 当前每折仅 250 步(25 epoch x 10 步), 可能欠拟合。

变体(5 seed 配对, 同一折划分):
  e15 / e25 / e50 / e100   联合模型, 不同 epoch
  sep6_e25                6 个完全独立的单基因模型(trunk/attention/head 全不共享)
  sep6_e50                同上但 50 epoch, 排除"只是没训够"

判读:
  若 sep6 > joint  -> 共享 trunk 确实在利用共突变结构(隐式泄漏), 分开训更好
  若 sep6 ≈ joint  -> 多任务既没帮也没害
  若 sep6 < joint  -> 多任务正则有效, 共突变泄漏不足以抵消
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, LR, HID, S_INST = 1536, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = idx[["case_id", "mem_row"]].merge(iu[["case_id"]], on="case_id", how="inner")
df = df.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); RT = df.mem_row.values.astype(int)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
strata = np.clip(Y.sum(1), 0, 3).astype(int)
K = len(GENES)
print(f"[LUAD n={N}] X={X.shape}  联合 vs 分离", flush=True)


class Bag(nn.Module):
    """共享的 trunk + gated attention; k 个输出头"""
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, DIM), np.float16)
    for i, rw in enumerate(rows):
        v = X[RT[rw]]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


def train_one(Xtr, ytr, Xte, epochs, seed_tag):
    """训练一个单输出模型(Xtr/ytr 为已经过滤到单基因的数组)"""
    n = len(ytr)
    ycol = ytr.reshape(-1, 1).astype(np.float32)
    pwv = np.where(ycol.sum(0) > 0, (n - ycol.sum(0)) / np.maximum(ycol.sum(0), 1), 1.0)
    pw = torch.tensor(pwv, dtype=torch.float32, device=dev)
    net = Bag(DIM, HID, 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=epochs, steps_per_epoch=max(1, n // BATCH))
    rng = np.random.RandomState(seed_tag)
    lf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        net.train(); perm = rng.permutation(n)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            bt = torch.from_numpy(gather(Xtr[rr], rng)).to(dev, torch.float32)
            yb = torch.from_numpy(ycol[rr]).to(dev)
            loss = lf(net(bt), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return net(torch.from_numpy(Xte).to(dev, torch.float32)).float().cpu().numpy().ravel()


def run_joint(epochs, seed):
    op = np.zeros((N, K), np.float32)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        ytr = Y[tr]
        pw = torch.tensor(np.where(ytr.sum(0) > 0, (len(tr) - ytr.sum(0)) / np.maximum(ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Bag(DIM, HID, K).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=epochs, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(epochs):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = torch.from_numpy(gather(rr, rng)).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            bt = torch.from_numpy(gather(te, np.random.RandomState(7))).to(dev, torch.float32)
            op[te] = net(bt).float().cpu().numpy()
    return op


def run_sep(epochs, seed):
    """6 个完全独立的单基因模型, 各自一套 trunk/attention/head"""
    op = np.zeros((N, K), np.float32)
    rows_all = np.arange(N)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        for j in range(K):
            pos = tr[Y[tr, j] == 1]; neg = tr[Y[tr, j] == 0]
            Xtr = rows_all[np.concatenate([pos, neg])]
            ytr = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            Xte = gather(te, np.random.RandomState(7)).astype(np.float16)
            op[te, j] = train_one(Xtr, ytr, Xte, epochs, seed * 1000 + f * 10 + j)
    return op


auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
CFGS = [("joint_e15", lambda s: run_joint(15, s)),
        ("joint_e25", lambda s: run_joint(25, s)),
        ("joint_e50", lambda s: run_joint(50, s)),
        ("joint_e100", lambda s: run_joint(100, s)),
        ("sep6_e25", lambda s: run_sep(25, s)),
        ("sep6_e50", lambda s: run_sep(50, s))]

res = {n: np.zeros((len(SEEDS), K), np.float32) for n, _ in CFGS}
store = {n: [] for n, _ in CFGS}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for name, fn in CFGS:
        op = fn(seed)
        res[name][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[name].append(op)
        line += f"{name}={np.nanmean(res[name][seed]):.3f}  "
    print(line, flush=True)

print("\n" + "=" * 100, flush=True)
print("=== 联合 vs 分离 + epoch 扫描 (5 seed 配对) ===", flush=True)
base_s = res["joint_e25"].mean(1)
print(f"{'config':<12}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}{'Δ vs e25':>11}{'同号':>7}{'集成':>9}")
summary = []
for name, _ in CFGS:
    m = res[name].mean(0); d = res[name].mean(1) - base_s
    ens = np.mean(store[name], axis=0)
    a_ens = [auc(Y[:, j], ens[:, j]) for j in range(K)]
    print(f"{name:<12}" + "".join(f"{x:>9.3f}" for x in m) + f"{m.mean():>9.4f}{d.mean():>+11.4f}"
          + f"{int((d>0).sum()):>6}/5{np.mean(a_ens):>9.4f}", flush=True)
    summary.append(dict(config=name, mean_auc=float(m.mean()), delta=float(d.mean()),
                        agree=int((d > 0).sum()), ens_auc=float(np.mean(a_ens)),
                        **{f"auc_{g}": float(a) for g, a in zip(GENES, m)}))

print("\n=== 判读 ===", flush=True)
for s in summary:
    d = s["delta"]; ag = s["agree"]
    v = ("有效提升 ✅" if d > 0.01 and ag >= 4 else "明确有害 ❌" if d < -0.01 else "噪声")
    print(f"  {s['config']:<12} Δ={d:>+8.4f}  同号 {ag}/5  集成 {s['ens_auc']:.4f}  -> {v}")

pd.DataFrame(summary).to_csv(os.path.join(C, "joint_vs_sep.csv"), index=False)
rows = [dict(config=n, seed=si, gene=g, auc=float(res[n][si, j]))
        for n, _ in CFGS for si in range(len(SEEDS)) for j, g in enumerate(GENES)]
pd.DataFrame(rows).to_csv(os.path.join(C, "joint_vs_sep_sweep.csv"), index=False)
for n, _ in CFGS:
    np.save(os.path.join(C, f"oof_{n}_5seed.npy"), np.mean(store[n], axis=0))
print(f"\n[saved] {C}\\joint_vs_sep.csv / joint_vs_sep_sweep.csv / oof_*_5seed.npy")
print("DONE", flush=True)