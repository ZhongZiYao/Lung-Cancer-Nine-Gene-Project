# -*- coding: utf-8 -*-
"""59_oof_noise_floor.py — 量出 5-fold OOF 协议自身的噪声底

问题:
  §18 用 bootstrap 证明了 test(mean) 的 CI 半宽是 0.054, 并在 §18.4 写下
  "任何 |Δ|<0.05 的配置差异不作为结论"。但此前 19 项实验全部是在
  5-fold OOF(434 例)上、用 0.01 作为噪声阈值判定的。

  0.01 是【种子层面】的噪声(已由 5-seed 配对实测)。而 OOF 这个【估计量】
  本身跨折划分重随机会有多大波动, 我们从来没量过。如果它是 0.03,
  那么 sep6 -0.0330、joint_e50 -0.0199 这些"结论"就不成立。

做法:
  固定最优配置(gigap / h=512 / lr=3e-4 / drop=0.25 / ep=40 / 6 基因联合),
  只改变 5 折的划分种子, 每个划分种子内跑 5 个模型种子取平均。
  于是 OOF mean 的标准差 = 我们引用 0.7431 这个数字时应该附上的误差。

输出: oof_noise_floor.json / .csv
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST = 1536, 32, 512
dev = "cuda"
FOLD_SEEDS = [101, 202, 303, 404, 505]     # 折划分的随机性
MODEL_SEEDS = [0, 1, 2, 3, 4]
HP = dict(h=512, lr=3e-4, drop=0.25, ep=40)
NFOLD = 5

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
idxf = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
rowmap = dict(zip(idxf.case_id, idxf.mem_row))
common = sorted(set(lab.index) & set(rowmap))
Y = lab.loc[common, GENES].values.astype(int)
rows_all = np.array([rowmap[c] for c in common])
print("[data] LUAD case %d   pos %s" % (len(common), " ".join(
    "%s=%d" % (g, int(Y[:, j].sum())) for j, g in enumerate(GENES))), flush=True)

# 分层的依据: TP53(阳性最多) 与 NF1(阳性最少) 合成一个粗分层标签
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))


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


_X = None
def X():
    global _X
    if _X is None:
        print("[ram] loading %.0f MB" % (os.path.getsize(os.path.join(C, "NSCLC_gigap_sub512.npy")) / 1e6), flush=True)
        _X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
    return _X


def batches(rws, rng, Xa):
    bt = np.empty((len(rws), S_INST, DIM), np.float16)
    for i, r in enumerate(rws):
        v = Xa[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr_rows, tr_y, te_rows, seed):
    torch.manual_seed(seed)
    Xa = X()
    net = Net(DIM, HP["h"], len(GENES), HP["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), HP["lr"], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, HP["lr"], epochs=HP["ep"],
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(HP["ep"]):
        net.train(); perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(batches(tr_rows[rr], rng, Xa)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return net(batches(te_rows, np.random.RandomState(seed + 999), Xa)).float().cpu().numpy()


print("\n" + "=" * 92)
print("=== 5-fold OOF 均值的噪声底: 换折划分种子, 每划分内 5 个模型种子 ===")
print("=" * 92, flush=True)
rows_out, summary = [], {}
for fs in FOLD_SEEDS:
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=fs)
    oof = np.zeros((len(common), len(GENES)))
    for f, (tri, tei) in enumerate(skf.split(np.zeros(len(common)), tier)):
        ps = np.mean([fit(rows_all[tri], Y[tri], rows_all[tei], s) for s in MODEL_SEEDS], 0)
        oof[tei] = ps
    per = [float(roc_auc_score(Y[:, j], oof[:, j])) if 10 <= Y[:, j].sum() <= len(Y) - 10 else np.nan
           for j in range(len(GENES))]
    m = float(np.nanmean(per))
    summary[fs] = m
    for g, a in zip(GENES, per):
        rows_out.append(dict(fold_seed=fs, gene=g, oof_auc=a, oof_mean=m))
    print("  fold_seed=%-4d  %s  mean=%.4f"
          % (fs, "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a) for g, a in zip(GENES, per)), m), flush=True)

vals = np.array([summary[fs] for fs in FOLD_SEEDS])
print("\n  OOF mean over %d fold-split seeds: %.4f +- %.4f   (min %.4f  max %.4f)"
      % (len(vals), vals.mean(), vals.std(ddof=1), vals.min(), vals.max()), flush=True)
print("  => 本项目引用 0.7431 这个 OOF 数字时, 应附带的误差是 +-%s" % ("%.4f" % (2 * vals.std(ddof=1))), flush=True)
print("\n  重新审视此前用 0.01 阈值判定的结论:", flush=True)
for name, d in [("53b CPTAC 增训", 0.0068), ("34 score_avg 集成", 0.0074), ("41 joint_e50", -0.0199),
                ("41 sep6 分基因训练", -0.0330), ("41 joint_e100", -0.0281), ("52 UNI2-h", -0.0166),
                ("43 LUSC 增训", -0.0498), ("33 zslide", -0.0611), ("39 meanpool", -0.0376),
                ("46 routeA_pure", -0.0052), ("30 rank loss", -0.1953)]:
    thr = 2 * vals.std(ddof=1)
    print("    %-22s d=%+.4f   %s" % (name, d, "仍成立" if abs(d) > thr else "**不成立(落在噪声内)**"), flush=True)

pd.DataFrame(rows_out).to_csv(os.path.join(C, "oof_noise_floor.csv"), index=False)
json.dump({"fold_seeds": FOLD_SEEDS, "means": {str(k): v for k, v in summary.items()},
           "mean": float(vals.mean()), "std": float(vals.std(ddof=1))},
          open(os.path.join(C, "oof_noise_floor.json"), "w"), indent=2)
print("\n[saved] oof_noise_floor.csv / .json", flush=True)
print("DONE", flush=True)