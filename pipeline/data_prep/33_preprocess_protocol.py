# -*- coding: utf-8 -*-
"""33_preprocess_protocol.py — 第二轮: 特征预处理 + 协议级提升

第一轮(30)结论: 损失函数与注意力变体全部落在噪声内或崩塌。
本轮换轴: 不动损失, 改【喂进模型之前的特征形态】与【评估协议】。

变体:
  base    参照 (同 30 号脚本, 保证可比)
  l2      每个 patch 向量 L2 归一化后再进 trunk
          理由: ViT 系 patch 特征范数差异大, gated attention 的 softmax 对尺度敏感
  zslide  逐切片逐维 z-score (在该 slide 自己的 patch 上算均值方差)
          理由: 抹掉切片级染色强度/批次偏移 —— 期望同时改善内部与外部
  deep    两层 trunk (d->512->256) + LayerNorm
  tta     推理期 3 次不同 patch 采样取平均
  f10     10-fold 取代 5-fold (每折训练样本翻倍)

安全性: zslide 只用该样本自身的 patch, 不跨样本, 无标签泄漏。
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, EP, BATCH, LR, HID, S_INST = 1536, 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["base", "l2", "zslide", "deep", "tta", "f10"]

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); RT = df.mem_row.values.astype(int)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
strata = np.clip(Y.sum(1), 0, 3).astype(int)
K = len(GENES)
print(f"[LUAD n={N}] X={X.shape} genes={K}", flush=True)

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
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, strata):
            if len(set(Y[tr, j])) < 2:
                continue
            sc = StandardScaler().fit(Xf[tr])
            m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xf[tr]), Y[tr, j])
            o[te, j] = m.predict_proba(sc.transform(Xf[te]))[:, 1]
    return o

def pre(x, mode):
    if mode == "l2":
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if mode == "zslide":
        m = x.mean(1, keepdim=True)
        s = x.std(1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    return x

class Net(nn.Module):
    def __init__(s, d, h, k, variant):
        super().__init__()
        s.variant = variant
        if variant == "deep":
            s.trunk = nn.Sequential(nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(),
                                    nn.Dropout(0.3), nn.Linear(512, h), nn.ReLU(), nn.Dropout(0.25))
        else:
            s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        x = pre(x, s.variant)
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))

def gather(rows, rng, tag):
    bt = np.empty((len(rows), S_INST, DIM), np.float16)
    for i, rw in enumerate(rows):
        v = X[RT[rw]]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt

def run(variant, seed, folds):
    op = np.zeros((N, K), np.float32)
    tta_reps = 3 if variant == "tta" else 1
    for f, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        pw = torch.tensor(np.where(Y[tr].sum(0) > 0,
                                   (len(tr) - Y[tr].sum(0)) / np.maximum(Y[tr].sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(DIM, HID, K, variant).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = torch.from_numpy(gather(rr, rng, "tr")).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        acc = np.zeros((len(te), K), np.float32)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for r in range(tta_reps):
                bt = torch.from_numpy(gather(te, np.random.RandomState(7000 + r), "te")).to(dev, torch.float32)
                acc += net(bt).float().cpu().numpy()
        op[te] = acc / tta_reps
    return op

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
gA = geno_oof()
gA_auc = [auc(Y[:, j], gA[:, j]) for j in range(K)]
print(f"[基因组基线 LOO] mean = {np.nanmean(gA_auc):.3f}\n", flush=True)

FOLDS_OF = {"f10": 10}
res = {v: np.zeros((len(SEEDS), K), np.float32) for v in VARIANTS}
oof_store = {v: [] for v in VARIANTS}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for v in VARIANTS:
        op = run(v, seed, FOLDS_OF.get(v, 5))
        res[v][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        oof_store[v].append(op)
        line += f"{v}={np.nanmean(res[v][seed]):.3f}  "
    print(line, flush=True)

print("\n" + "=" * 96, flush=True)
print("=== 5-seed 配对汇总 ===", flush=True)
base_m = res["base"].mean(0)
base_s = res["base"].mean(1)
summary = []
for v in VARIANTS:
    m = res[v].mean(0); sd = res[v].mean(1).std()
    d_seed = res[v].mean(1) - base_s
    agree = int((d_seed > 0).sum())
    print(f"{v:<9}" + "".join(f"{a:>9.3f}" for a in m) + f"{m.mean():>9.3f}{sd:>9.4f}"
          + f"{d_seed.mean():>+10.4f}{d_seed.std():>8.4f}{agree:>6}/5", flush=True)
    summary.append(dict(variant=v, mean_auc=float(m.mean()), seed_std=float(sd),
                        delta_mean=float(d_seed.mean()), delta_std=float(d_seed.std()), agree=agree,
                        net_signal=float(m.mean() - np.nanmean(gA_auc))))

# 多种子集成: 平均 5 个 seed 的 OOF 预测
print("\n=== 多种子集成 (5 seed OOF 预测取平均) ===", flush=True)
for v in VARIANTS:
    ens = np.mean(oof_store[v], axis=0)
    a_ens = [auc(Y[:, j], ens[:, j]) for j in range(K)]
    single = res[v].mean()
    print(f"  {v:<9} 单 seed 均值 {single:.4f}  ->  5-seed 集成 {np.mean(a_ens):.4f}  "
          f"({np.mean(a_ens)-single:+.4f})", flush=True)
    summary.append(dict(variant=v + "_ens5", mean_auc=float(np.mean(a_ens)),
                        seed_std=0.0, delta_mean=float(np.mean(a_ens) - single),
                        delta_std=0.0, agree=5,
                        net_signal=float(np.mean(a_ens) - np.nanmean(gA_auc))))

rows = [dict(variant=v, seed=si, gene=g, auc=float(res[v][si, j]))
        for v in VARIANTS for si in range(len(SEEDS)) for j, g in enumerate(GENES)]
pd.DataFrame(rows).to_csv(os.path.join(C, "preprocess_sweep.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(C, "preprocess_summary.csv"), index=False)
np.save(os.path.join(C, "oof_base5seed.npy"), np.mean(oof_store["base"], axis=0))
print(f"\n[saved] {C}\\preprocess_sweep.csv / preprocess_summary.csv / oof_base5seed.npy")
print("DONE", flush=True)