# -*- coding: utf-8 -*-
"""39_pooling_round.py — 聚合器第三轮: 直接攻击"MIL 增量只有 +0.08"这个核心问题

37 号脚本测出: 切片均值(1536d) + 线性探针 = 0.663, MIL = 0.743, 注意力只贡献 +0.08。
要扩大这个增量, 只能换聚合方式。本轮把经典 MIL 池化补齐——

注意: 22 号脚本的消融只测了 gated/clam_sb/abmilx/proto_loo,
【从未测过最经典的 max-pooling 和 NN 内 mean-pooling】, 缺一个真正的参照系。

变体(全部 5 seed 配对, 同一折划分):
  base    gated 注意力(现有最优)
  mean    NN 内平均池化        <- 真正的强基线(同架构同参数量)
  max     最大池化            <- 经典 MIL 基线, 至今未测
  lse     LogSumExp 池化(可学习温度), 介于 max 和 mean 之间
  topk    top-10% 均值池化
  gctx    gated 加 max 上下文分支拼接
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, EP, BATCH, LR, HID, S_INST = 1536, 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["base", "mean", "max", "lse", "topk", "gctx"]

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
print(f"[LUAD n={N}] X={X.shape}", flush=True)

def gated_att(V, U, w, z):
    return torch.softmax(w(torch.tanh(V(z)) * torch.sigmoid(U(z))).squeeze(-1), 1)

class Net(nn.Module):
    def __init__(s, d, h, k, mode):
        super().__init__()
        s.mode = mode
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False)
        out_h = 2 * h if mode == "gctx" else h
        s.od = nn.Linear(out_h, k)
        if mode == "lse":
            s.tau = nn.Parameter(torch.tensor(3.0))
    def forward(s, x):
        z = s.trunk(x)                      # (B,S,H)
        a = gated_att(s.V, s.U, s.w, z)
        m = s.mode
        if m == "base":
            bag = (z * a.unsqueeze(-1)).sum(1)
        elif m == "mean":
            bag = z.mean(1)
        elif m == "max":
            bag = z.max(1).values
        elif m == "lse":
            t = s.tau.clamp(min=0.5, max=20.0)
            bag = t * torch.logsumexp(z / t, dim=1)
        elif m == "topk":
            kk = max(1, int(0.10 * z.shape[1]))
            bag = z.topk(kk, dim=1).values.mean(1)
        else:                                # gctx
            bag = torch.cat([(z * a.unsqueeze(-1)).sum(1), z.max(1).values], 1)
        return s.od(bag)

def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, DIM), np.float16)
    for i, rw in enumerate(rows):
        v = X[RT[rw]]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt

def run(mode, seed):
    op = np.zeros((N, K), np.float32)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        pw = torch.tensor(np.where(Y[tr].sum(0) > 0,
                                   (len(tr) - Y[tr].sum(0)) / np.maximum(Y[tr].sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(DIM, HID, K, mode).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
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

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
res = {v: np.zeros((len(SEEDS), K), np.float32) for v in VARIANTS}
store = {v: [] for v in VARIANTS}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for v in VARIANTS:
        op = run(v, seed)
        res[v][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[v].append(op)
        line += f"{v}={np.nanmean(res[v][seed]):.3f}  "
    print(line, flush=True)

print("\n" + "=" * 98, flush=True)
print("=== 聚合器第三轮 5-seed 配对 ===", flush=True)
base_m = res["base"].mean(0); base_s = res["base"].mean(1)
summary = []
for v in VARIANTS:
    m = res[v].mean(0); d = res[v].mean(1) - base_s
    ens = np.mean(store[v], axis=0)
    a_ens = [auc(Y[:, j], ens[:, j]) for j in range(K)]
    print(f"{v:<8}" + "".join(f"{x:>9.3f}" for x in m) + f"{m.mean():>9.4f}"
          + f"{d.mean():>+10.4f}{int((d>0).sum()):>6}/5  ens={np.mean(a_ens):.4f}", flush=True)
    summary.append(dict(variant=v, mean_auc=float(m.mean()), delta=float(d.mean()),
                        agree=int((d > 0).sum()), ens_auc=float(np.mean(a_ens))))

print("\n=== 判定 ===", flush=True)
best = max(summary, key=lambda s: s["mean_auc"])
for s in summary:
    if s["variant"] == "base":
        continue
    d = s["delta"]; ag = s["agree"]
    v = ("有效提升 ✅" if d > 0.01 and ag >= 4 else
         "明确有害 ❌" if d < -0.01 else "噪声")
    print(f"  {s['variant']:<8} Δ={d:>+8.4f}  同号 {ag}/5  集成 {s['ens_auc']:.4f}  -> {v}")
print(f"\n最佳单种子: {best['variant']} {best['mean_auc']:.4f}")
best_ens = max(summary, key=lambda s: s["ens_auc"])
print(f"最佳 5-seed 集成: {best_ens['variant']} {best_ens['ens_auc']:.4f}")

rows = [dict(variant=v, seed=si, gene=g, auc=float(res[v][si, j]))
        for v in VARIANTS for si in range(len(SEEDS)) for j, g in enumerate(GENES)]
pd.DataFrame(rows).to_csv(os.path.join(C, "pooling_sweep.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(C, "pooling_summary.csv"), index=False)
for v in VARIANTS:
    np.save(os.path.join(C, f"oof_{v}pool_5seed.npy"), np.mean(store[v], axis=0))
print(f"\n[saved] {C}\\pooling_sweep.csv / pooling_summary.csv / oof_*pool_5seed.npy")
print("DONE", flush=True)