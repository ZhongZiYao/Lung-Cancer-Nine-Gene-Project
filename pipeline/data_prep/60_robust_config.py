# -*- coding: utf-8 -*-
"""60_robust_config.py — 用跨折划分重复, 而不是 87 例 val, 来定最终超参

§20.5 的发现: 55 号在 val(87 例)上选出的 h=512/ep=40, 在 OOF 上比
朴素的 h=256/ep=25 差 0.019, 超过噪声底 0.014。
也就是说"用小验证集选超参"在这个数据规模下是净负面的。

正确做法 (§20.6 的口径): 固定一组超参, 用多个折划分种子的平均 OOF 来比配置。
判据从"1 个 87 例 val"换成"3 个折划分种子的平均", 评估集从 87 变成 436,
判据本身的噪声从 ±0.05 降到 ±0.007。

设计: 完全配对 —— 所有配置共用同一批 (fold_seed, model_seed) 组合,
差值按对相减, 消除折划分带来的方差。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST, LR, WD = 1536, 32, 512, 3e-4, 1e-3
dev = "cuda"
FOLD_SEEDS = [101, 202, 303]
MODEL_SEEDS = [0, 1, 2]
NFOLD = 5

CFGS = [
    dict(h=256, ep=25, drop=0.25),   # 41 号的历史最优
    dict(h=256, ep=40, drop=0.25),
    dict(h=384, ep=25, drop=0.25),
    dict(h=512, ep=25, drop=0.25),
    dict(h=512, ep=40, drop=0.25),   # 55 号 val 选出来的
    dict(h=256, ep=25, drop=0.50),
]

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
idxf = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
rowmap = dict(zip(idxf.case_id, idxf.mem_row))
common = sorted(set(lab.index) & set(rowmap))
Y = lab.loc[common, GENES].values.astype(int)
rows_all = np.array([rowmap[c] for c in common])
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
print("[data] LUAD %d case" % len(common), flush=True)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
print("[ram] %.0f MB" % (X.nbytes / 1e6), flush=True)


class Net(nn.Module):
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def batches(rws, rng):
    bt = np.empty((len(rws), S_INST, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tri, tei, cfg, seed):
    torch.manual_seed(seed)
    tr_rows, tr_y, te_rows = rows_all[tri], Y[tri], rows_all[tei]
    net = Net(DIM, cfg["h"], len(GENES), cfg["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=cfg["ep"],
                                              steps_per_epoch=max(1, len(tri) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(cfg["ep"]):
        net.train(); perm = rng.permutation(len(tri))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(batches(tr_rows[rr], rng)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return net(batches(te_rows, np.random.RandomState(seed + 999))).float().cpu().numpy()


# folds 预先算好, 所有配置共用 -> 完全配对
FOLDS = []
for fs in FOLD_SEEDS:
    FOLDS.append([(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(common)), tier)])
print("[folds] %d fold-seeds x %d folds (所有配置共用)" % (len(FOLDS), NFOLD), flush=True)

print("\n" + "=" * 92)
print("=== 判据 = 3 个折划分种子的平均 OOF (评估集 436 例, 噪声约 +-0.01) ===")
print("=" * 92, flush=True)
per_cfg_mean, rows_out = {}, []
for ci, cfg in enumerate(CFGS):
    oof_per_seed = []
    for fs_i, folds in enumerate(FOLDS):
        oof = np.zeros((len(common), len(GENES)))
        for tri, tei in folds:
            oof[tei] = np.mean([fit(tri, tei, cfg, s) for s in MODEL_SEEDS], 0)
        per = [float(roc_auc_score(Y[:, j], oof[:, j])) for j in range(len(GENES))]
        oof_per_seed.append(np.mean(per))
    m = float(np.mean(oof_per_seed))
    sd = float(np.std(oof_per_seed, ddof=1)) if len(oof_per_seed) > 1 else float("nan")
    per_cfg_mean[ci] = oof_per_seed
    rows_out.append(dict(h=cfg["h"], ep=cfg["ep"], drop=cfg["drop"], oof_mean=m, oof_std=sd,
                         seeds=" ".join("%.4f" % v for v in oof_per_seed)))
    print("  h=%-4d ep=%-3d drop=%.2f  OOF=%.4f (std %.4f)  seeds=%s"
          % (cfg["h"], cfg["ep"], cfg["drop"], m, sd, " ".join("%.4f" % v for v in oof_per_seed)), flush=True)

base = per_cfg_mean[0]
print("\n  与历史最优 h=256/ep=25 的配对差 (逐 fold-seed 相减):", flush=True)
for ci, cfg in enumerate(CFGS):
    if ci == 0:
        continue
    d = [a - b for a, b in zip(per_cfg_mean[ci], base)]
    print("    h=%-4d ep=%-3d drop=%.2f  mean d=%+.4f  per-seed=%s  同号 %d/3"
          % (cfg["h"], cfg["ep"], cfg["drop"], np.mean(d), " ".join("%+.4f" % x for x in d),
             sum(1 for x in d if x > 0)), flush=True)

best = max(range(len(CFGS)), key=lambda i: float(np.mean(per_cfg_mean[i])))
print("\n  最优配置: h=%d ep=%d drop=%.2f  ->  OOF %.4f"
      % (CFGS[best]["h"], CFGS[best]["ep"], CFGS[best]["drop"], float(np.mean(per_cfg_mean[best]))), flush=True)
pd.DataFrame(rows_out).to_csv(os.path.join(C, "robust_config.csv"), index=False)
print("[saved] robust_config.csv", flush=True)
print("DONE", flush=True)