# -*- coding: utf-8 -*-
"""34_dual_encoder_fusion.py — 双编码器融合 (本项目最可能出真增量的轴)

动机: 本地同时有两套公开 patch 特征, 覆盖同一批 955 张 NSCLC 切片:
  - UNI v1        1024 维 (DINOv1 系自监督)
  - Prov-GigaPath 1536 维 (1.4B tile 预训练)
二者训练目标和数据分布完全不同, 信息可能互补。此前所有实验都只用其中一套。
调研侧证据: FuseCPath (arXiv 2510.27237) 正是融合 CONCH+Virchow+GigaPath 后蒸馏,
声称较最佳基线 +12.7% —— 但它是端到端重提特征, 我们没有那个算力;
本脚本换成"冻结特征 + 轻量融合", 8.5GB 显卡可跑。

变体:
  uni        只用 UNI v1
  giga       只用 Prov-GigaPath
  concat     特征直接拼接 (2560 维) 后进同一个 trunk
  late       两套独立 trunk, logits 取平均 (late fusion)
  score_avg  训练完 uni/giga 后, 离线平均两者 OOF logits (零成本集成)
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
EP, BATCH, LR, HID, S_INST = 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
ig = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
j = iu[["case_id", "mem_row"]].merge(ig[["case_id", "mem_row"]], on="case_id", suffixes=("_u", "_g"))
j = j.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
j = j[j.project == "LUAD"].reset_index(drop=True)
N = len(j); Y = j[GENES].values.astype(int)
RU = j.mem_row_u.values.astype(int); RG = j.mem_row_g.values.astype(int)
XU = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))
XG = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
DU, DG = XU.shape[2], XG.shape[2]
strata = np.clip(Y.sum(1), 0, 3).astype(int)
K = len(GENES)
print(f"[LUAD n={N}]  UNI={XU.shape}  GigaPath={XG.shape}", flush=True)

class Bag(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))

class Net(nn.Module):
    """mode: uni / giga / concat / late"""
    def __init__(s, mode, h, k):
        super().__init__(); s.mode = mode
        if mode == "uni":     s.net = Bag(DU, h, k)
        elif mode == "giga":  s.net = Bag(DG, h, k)
        elif mode == "concat":s.net = Bag(DU + DG, h, k)
        else:                 s.n1 = Bag(DU, h, k); s.n2 = Bag(DG, h, k)
    def forward(s, xu, xg):
        if s.mode == "uni":     return s.net(xu)
        if s.mode == "giga":    return s.net(xg)
        if s.mode == "concat":  return s.net(torch.cat([xu, xg], -1))
        return (s.n1(xu) + s.n2(xg)) / 2

def gather(X, rows, rng):
    bt = np.empty((len(rows), S_INST, X.shape[2]), np.float16)
    for i, rw in enumerate(rows):
        v = X[rw]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt

def run(mode, seed, folds=5):
    op = np.zeros((N, K), np.float32)
    for f, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        pw = torch.tensor(np.where(Y[tr].sum(0) > 0,
                                   (len(tr) - Y[tr].sum(0)) / np.maximum(Y[tr].sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(mode, HID, K).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bu = torch.from_numpy(gather(XU, RU[rr], rng)).to(dev, torch.float32)
                bg = torch.from_numpy(gather(XG, RG[rr], rng)).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                loss = lf(net(bu, bg), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            bu = torch.from_numpy(gather(XU, RU[te], np.random.RandomState(7))).to(dev, torch.float32)
            bg = torch.from_numpy(gather(XG, RG[te], np.random.RandomState(7))).to(dev, torch.float32)
            op[te] = net(bu, bg).float().cpu().numpy()
    return op

auc = lambda y, p: float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
MODES = ["uni", "giga", "concat", "late"]
res = {m: np.zeros((len(SEEDS), K), np.float32) for m in MODES}
store = {m: [] for m in MODES}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for m in MODES:
        op = run(m, seed)
        res[m][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[m].append(op)
        line += f"{m}={np.nanmean(res[m][seed]):.3f}  "
    print(line, flush=True)

# 零成本后验集成: 平均 uni 与 giga 的 logits
sa = [(np.mean(store["uni"], 0) + np.mean(store["giga"], 0)) / 2]
a_sa = [auc(Y[:, j], sa[0][:, j]) for j in range(K)]
res["score_avg"] = np.array([a_sa] * len(SEEDS), np.float32)

print("\n" + "=" * 92, flush=True)
print("=== 双编码器融合 5-seed 配对 ===", flush=True)
base_m = res["giga"].mean(0)
base_s = res["giga"].mean(1)
summary = []
for m in MODES + ["score_avg"]:
    mm = res[m].mean(0); sd = res[m].mean(1).std()
    d = res[m].mean(1) - base_s
    print(f"{m:<10}" + "".join(f"{a:>9.3f}" for a in mm) + f"{mm.mean():>9.3f}{sd:>9.4f}"
          + f"{d.mean():>+10.4f}{int((d>0).sum()):>6}/5", flush=True)
    summary.append(dict(mode=m, mean_auc=float(mm.mean()), seed_std=float(sd),
                        delta_mean=float(d.mean()), agree=int((d > 0).sum()),
                        **{f"auc_{g}": float(a) for g, a in zip(GENES, mm)}))

# 互补性: 两个编码器的 OOF 残差相关
ru = res["uni"].mean(0) - base_m
rg = res["giga"].mean(0) - base_m
print("\n=== 互补性检查: 两编码器相对 GigaPath 基线的逐基因差值相关 ===", flush=True)
print("  uni-giga :", " ".join(f"{g}={v:+.3f}" for g, v in zip(GENES, ru)), flush=True)
print("  如果两者优劣互补(一个高一个低的基因多), 说明有融合空间", flush=True)

rows = [dict(mode=m, seed=si, gene=g, auc=float(res[m][si, j]))
        for m in MODES + ["score_avg"] for si in range(len(SEEDS)) for j, g in enumerate(GENES)]
pd.DataFrame(rows).to_csv(os.path.join(C, "dual_encoder_fusion.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(C, "dual_encoder_summary.csv"), index=False)
for m in MODES:
    np.save(os.path.join(C, f"oof_{m}_5seed.npy"), np.mean(store[m], axis=0))
print(f"\n[saved] {C}\\dual_encoder_fusion.csv / dual_encoder_summary.csv / oof_{m}_5seed.npy")
print("DONE", flush=True)