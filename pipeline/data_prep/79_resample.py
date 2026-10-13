# -*- coding: utf-8 -*-
"""79_resample.py - the patch-sampling question, asked properly

What we actually do today (measured, not assumed):
  cache build : from n_patch_orig (median 9943, max 46378) randomly take 512, ONCE, fixed
  train step  : v[rng.choice(512, 512)]  -> samples 512 out of a pool of 512
  test        : same, with RandomState(seed+999)
So the model sees the SAME 512 patches in every epoch. We use 512/9943 = 5.1% of the
available tissue and the model never sees a different subset.

Experiment 56 already showed that USING MORE INSTANCES does not help
(S_INST 512 -> 2048 from a 4096 pool: val +0.0039, test 0.7834 vs 0.7838, i.e. nothing).
What has never been tested is the other half: resampling a FRESH subset every step.
Slot-Mixup (arXiv:2311.17466) and the WACV 2026 spatial-dependency work
(arXiv:2602.19487) both argue that fixed subsets are what over-concentrates attention,
not that the instance count is too low.

Arms (all served from ONE 2048-patch pool so the fixed-vs-fresh contrast is exact):
  R0_fixed512    512 patches, fixed subset              <- current behaviour
  R1_resample512 512 patches drawn fresh from the 2048 pool at every step
  R2_fixed2048   all 2048, fixed                        <- isolates "more" from "fresh"

Protocol: same as 78 (gigap, h=256/ep=25, 4 fold-split seeds x 5 folds x 3 model seeds),
verdict by PAIRED bootstrap on the same 436 cases.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, LR, WD = 1536, 32, 3e-4, 1e-3
dev = "cuda"
FOLD_SEEDS = [101, 202, 303, 404]
MODEL_SEEDS, NFOLD, EPOCH, MIN_POS = [0, 1, 2], 5, 25, 8
POOL, B_BOOT = 2048, 2000

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
proj_all = man.set_index("case_id")["project"].to_dict()
i4 = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub4096_index.csv"))
rm = dict(zip(i4.case_id, i4.mem_row))
cases = sorted(c for c in rm if proj_all.get(c) == "LUAD" and c in lab.index)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
rows = np.array([rm[c] for c in cases])
K = len(GENES)
ok_genes = [j for j in range(K) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]
print("[data] LUAD %d ; 源 patch 中位 %d (min %d max %d)"
      % (len(cases), i4.set_index("case_id").loc[cases].n_patch_orig.median(),
         i4.n_patch_orig.min(), i4.n_patch_orig.max()), flush=True)
print("[data] 池 = 每例前 %d 个(来自 sub4096 缓存); 512/%d = %.1f%% 组织被用到"
      % (POOL, int(i4.set_index("case_id").loc[cases].n_patch_orig.median()),
         100.0 * 512 / i4.set_index("case_id").loc[cases].n_patch_orig.median()), flush=True)

X = np.load(os.path.join(C, "NSCLC_gigap_sub4096.npy"), mmap_mode="r")
print("[ram] mmap %.0f MB" % (os.path.getsize(os.path.join(C, "NSCLC_gigap_sub4096.npy")) / 1e6), flush=True)
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
         for fs in FOLD_SEEDS]


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


FIXED = {r: np.random.RandomState(1234 + r).choice(POOL, 512, replace=False) for r in rows}
print("[check] 固定子集在两例间余弦(CLS均值)应明显<1 :", flush=True)


def make_batch(rws, S, rng, fresh):
    bt = np.empty((len(rws), S, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]
        n = min(POOL, v.shape[0])
        if fresh:
            idx = rng.choice(n, S, replace=n < S)
        else:
            # fixed arm: 512 uses the pinned subset, 2048 uses the whole pool
            idx = FIXED[r][:S] if S <= len(FIXED[r]) else np.arange(n)
        bt[i] = v[idx]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr, ytr, te, S, fresh, seed):
    torch.manual_seed(seed)
    net = Net(DIM, 256, K, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCH, steps_per_epoch=max(1, len(tr) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(ytr.sum(0) > 0, (len(ytr) - ytr.sum(0)) / np.maximum(ytr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        net.train(); perm = rng.permutation(len(tr))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(ytr[rr].astype(np.float32)).to(dev)
            loss = lf(net(make_batch(tr[rr], S, rng, fresh)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        return net(make_batch(te, S, np.random.RandomState(seed + 999), fresh)).float().cpu().numpy()


ARMS = [("R0_fixed512", 512, False), ("R1_resample512", 512, True), ("R2_fixed2048", 2048, False)]
mean_auc = lambda P: float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in ok_genes]))


def paired_boot(PA, PB, B=B_BOOT):
    rng = np.random.RandomState(0); n = len(Y); d = np.empty(B)
    for b in range(B):
        s = rng.randint(0, n, n)
        d[b] = np.mean([roc_auc_score(Y[s, j], PA[s, j]) for j in ok_genes]) - \
               np.mean([roc_auc_score(Y[s, j], PB[s, j]) for j in ok_genes])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


print("\n" + "=" * 96, flush=True)
print("=== 采样策略: 固定子集 vs 每 step 重采样 ===", flush=True)
print("=" * 96, flush=True)
oof = {a[0]: np.zeros((len(FOLDS), len(cases), K), np.float32) for a in ARMS}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    for tri, tei in folds:
        for name, S, fresh in ARMS:
            oof[name][fi][tei] = np.mean([fit(rows[tri], Y[tri], rows[tei], S, fresh, s) for s in MODEL_SEEDS], 0)
    print("  foldseed %d: %s  [%.0fs]" % (fi, "  ".join("%s=%.4f" % (n, mean_auc(oof[n][fi])) for n, _, _ in ARMS),
                                           time.time() - T0), flush=True)

print("\n=== 汇总 ===", flush=True)
ps = {}
for name, _, _ in ARMS:
    m = [mean_auc(oof[name][i]) for i in range(len(FOLDS))]
    ps[name] = m
    print("  %-15s %.4f +- %.4f" % (name, np.mean(m), np.std(m, ddof=1)), flush=True)
print("\n=== 配对 bootstrap vs R0_fixed512 ===", flush=True)
P = {n: oof[n].mean(0) for n, _, _ in ARMS}
rows_out = []
for name, _, _ in ARMS[1:]:
    d, lo, hi, pp = paired_boot(P[name], P["R0_fixed512"])
    print("  %-15s d=%+.4f  95%%CI [%+.4f, %+.4f]  P(d>0)=%.3f  %s"
          % (name, d, lo, hi, pp, "**显著**" if (lo > 0 or hi < 0) else "不显著"), flush=True)
    rows_out.append(dict(arm=name, auc=np.mean(ps[name]), d=d, ci_lo=lo, ci_hi=hi, p_pos=pp))
pd.DataFrame(rows_out).to_csv(os.path.join(C, "resample_sweep.csv"), index=False)
print("\n[saved] resample_sweep.csv", flush=True)
print("DONE", flush=True)