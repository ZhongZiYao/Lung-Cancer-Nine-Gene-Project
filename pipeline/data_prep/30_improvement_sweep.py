# -*- coding: utf-8 -*-
"""30_improvement_sweep.py — 单变量改进扫描 (5 seed 配对)

协议纪律:
  - 每个变体只改一处, 其余与 base 完全一致
  - 同一 seed 下所有变体共用同一套 StratifiedKFold 划分 -> 配对比较
  - 不带联合头 (共突变捷径通道, 只加 +0.004 且污染"纯形态"测量)
  - 单次结果一律不采信; |Δ|<0.01 视为噪声

变体与文献依据:
  base    gated 注意力 + BCE                       参照
  focal   focal loss gamma=2                       类别不平衡常规手段
  rank    pairwise ranking surrogate               直接优化 AUC 下界
  rank_hn ranking + 难负样本加权                    HNM-WSI, MICCAI 2024 (同协议 SOTA)
  std     bag = mean ⊕ std                          VarPool / SoMIL 思路, 二阶统计廉价代理
  temp    softmax 温度可学习                        注意力锐度

外部证据(调研得到, 决定不做的事):
  - THREADS (arXiv 2501.16652) 独立报告 transformer slide encoder 比 ABMIL 低 6.3%
  - Nisar et al. Sci Rep 2024: 6 种 stain normalization 全部不显著 (p=0.11)
  - 融合共突变 LOO 已有负结果, 不再尝试
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, EP, BATCH, LR, HID, FOLDS = 1536, 25, 32, 3e-4, 256, 5
S_INST = 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["base", "focal", "rank", "rank_hn", "std", "temp"]

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(CACHE, "NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); RT = df.mem_row.values.astype(int)
X = np.load(os.path.join(CACHE, "NSCLC_gigap_sub512.npy"))
strata = np.clip(Y.sum(1), 0, 3).astype(int)
K = len(GENES)
print(f"[LUAD n={N}] Prov-GigaPath patch={X.shape[1]} featdim={X.shape[2]} genes={K}", flush=True)

tn = lambda s: (lambda m: f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(s)))
gp = pd.read_parquet(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                  "provgigapath_embeddings_with_metadata.parquet"),
                     columns=["_PATIENT", "ajcc_pathologic_tumor_stage", "age_at_initial_pathologic_diagnosis"])
gp["case"] = gp["_PATIENT"].map(tn)
gg = gp.dropna(subset=["case"]).groupby("case").first()
oh = lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
STAGE = oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
AGE = np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1, 1), nan=0.0)

def geno_oof():
    o = np.zeros((N, K), np.float32)
    for j in range(K):
        Xf = np.column_stack([STAGE, AGE] + [Y[:, k] for k in range(K) if k != j])
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(Xf, strata):
            if len(set(Y[tr, j])) < 2:
                continue
            sc = StandardScaler().fit(Xf[tr])
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xf[tr]), Y[tr, j])
            o[te, j] = m.predict_proba(sc.transform(Xf[te]))[:, 1]
    return o

class Net(nn.Module):
    def __init__(s, d, h, k, variant):
        super().__init__()
        s.variant = variant; s.K = k
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False)
        if variant == "temp":
            s.logT = nn.Parameter(torch.zeros(1))
        s.od = nn.Linear(h * (2 if variant == "std" else 1), k)
    def forward(s, x):
        z = s.trunk(x)
        a = s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1)
        if s.variant == "temp":
            a = a / s.logT.exp().clamp(min=0.2, max=5.0)
        a = torch.softmax(a, 1)
        if s.variant == "std":
            bag = torch.cat([(z * a.unsqueeze(-1)).sum(1), z.std(1)], 1)
        else:
            bag = (z * a.unsqueeze(-1)).sum(1)
        return s.od(bag)

def focal_loss(logits, y, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    return (ce * (1 - pt).pow(gamma)).mean()

def rank_loss(scores, y, hard=False):
    tot = 0.0; n = 0
    for j in range(y.shape[1]):
        p = scores[y[:, j] > 0.5]; q = scores[y[:, j] < 0.5]
        if len(p) == 0 or len(q) == 0:
            continue
        d = p.unsqueeze(1) - q.unsqueeze(0)                 # (P,N)
        if hard:
            w = torch.softmax(q.detach() * 3.0, dim=0)      # 难负样本(高分)权重更高
            l = (F.softplus(-d) * w.unsqueeze(0)).sum() / (w.sum() * len(p) + 1e-8)
        else:
            l = F.softplus(-d).mean()
        tot = tot + l; n += 1
    return tot / max(n, 1)

def run(variant, seed):
    op = np.zeros((N, K), np.float32)
    for f, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        pw = torch.tensor(np.where(Y[tr].sum(0) > 0,
                                   (len(tr) - Y[tr].sum(0)) / np.maximum(Y[tr].sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(DIM, HID, K, variant).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = np.empty((len(rr), S_INST, DIM), np.float16)
                for k2, rw in enumerate(rr):
                    v = X[RT[rw]]; m = v.shape[0]
                    bt[k2] = v[rng.choice(m, S_INST, replace=m < S_INST)]
                bt = torch.from_numpy(bt).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                p = net(bt)
                if variant == "focal":
                    loss = focal_loss(p, yb, 2.0)
                elif variant == "rank":
                    loss = rank_loss(p, yb, hard=False)
                elif variant == "rank_hn":
                    loss = rank_loss(p, yb, hard=True)
                else:
                    loss = bce(p, yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            b = torch.from_numpy(X[np.ix_(RT[te], range(S_INST))]).to(dev, torch.float32)
            op[te] = net(b).float().cpu().numpy()
    return op

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
gA = geno_oof()
gA_auc = [auc(Y[:, j], gA[:, j]) for j in range(K)]
print(f"[基因组基线 LOO] mean AUC = {np.nanmean(gA_auc):.3f}\n", flush=True)

res = {v: np.zeros((len(SEEDS), K), np.float32) for v in VARIANTS}
saved = {}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for v in VARIANTS:
        op = run(v, seed)
        res[v][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        if v == "base":
            saved[seed] = op
        line += f"{v}={np.nanmean(res[v][seed]):.3f}  "
    print(line, flush=True)

print("\n" + "=" * 92, flush=True)
print("=== 5-seed 配对汇总 (纯形态 AUC) ===", flush=True)
print(f"{'variant':<10}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}{'seedstd':>9}{'净信号':>9}{'Δ vs base':>11}{'同号':>7}", flush=True)
base_mean = res["base"].mean(0)
base_seeds = res["base"].mean(1)
summary = []
for v in VARIANTS:
    m = res[v].mean(0); sd = res[v].mean(1).std()
    d = m - base_mean
    per_seed_delta = res[v].mean(1) - base_seeds
    agree = int((per_seed_delta > 0).sum())
    delta_mean = float(per_seed_delta.mean()); delta_std = float(per_seed_delta.std())
    print(f"{v:<10}" + "".join(f"{a:>9.3f}" for a in m) + f"{m.mean():>9.3f}{sd:>9.4f}"
          + f"{m.mean() - np.nanmean(gA_auc):>+9.3f}{delta_mean:>+11.4f}{agree:>6}/5", flush=True)
    summary.append(dict(variant=v, mean_auc=float(m.mean()), seed_std=float(sd),
                        net_signal=float(m.mean() - np.nanmean(gA_auc)),
                        delta_mean=delta_mean, delta_std=delta_std, agree=agree,
                        **{f"auc_{g}": float(a) for g, a in zip(GENES, m)}))

print("\n=== 判定 (|Δ|<0.01 或 seed 不一致 = 噪声) ===", flush=True)
for s in summary:
    d = s["delta_mean"]; ag = s["agree"]; ds = s["delta_std"]
    if d > 0.01 and ag >= 4 and ds < 0.01:
        v = "有效提升 ✅"
    elif d < -0.01 and ag <= 1:
        v = "明确有害 ❌"
    elif abs(d) < 0.01:
        v = "噪声,不可信"
    else:
        v = "方向不一致,需复核"
    print(f"  {s['variant']:<10} Δ={d:>+8.4f} (seed std {ds:.4f})  同号 {ag}/5  -> {v}", flush=True)

rows = [dict(variant=v, seed=si, gene=g, auc=float(res[v][si, j]))
        for v in VARIANTS for si in range(len(SEEDS)) for j, g in enumerate(GENES)]
pd.DataFrame(rows).to_csv(os.path.join(CACHE, "improvement_sweep.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(CACHE, "improvement_summary.csv"), index=False)
for s in SEEDS:
    np.save(os.path.join(CACHE, f"improvement_base_seed{s}.npy"), saved[s])
print(f"\n[saved] {CACHE}\\improvement_sweep.csv / improvement_summary.csv", flush=True)
print("DONE", flush=True)