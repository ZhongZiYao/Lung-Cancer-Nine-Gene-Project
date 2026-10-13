# -*- coding: utf-8 -*-
"""78_decisive.py - the decisive architecture test, with the statistics fixed first

Three changes relative to 77, all driven by the literature review and by our own
measured noise floor:

1. NO slide-level mixup. MixUp-MIL (MICCAI 2023) reports that plain cross-slide MixUp
   DECREASES accuracy for MIL; only intra-slide interpolation helps. 77 had it in three
   arms, which both confounds the arms and wastes a run on a known-negative.

2. 8 fold-split seeds instead of 3. Our OOF mean has std 0.007 over 3 fold-split seeds
   (section 20), so the decision threshold 2 sigma = 0.014. Going to 8 seeds scales the
   std by sqrt(3/8) -> ~0.0043, threshold ~0.009. This is the only lever that raises
   statistical power, and it is cheaper than any architecture change.

3. PAIRED bootstrap on the OOF difference instead of a threshold rule. Resampling the
   same 436 cases for both arms cancels the shared evaluation-set noise, so the CI on
   delta-AUC is far tighter than the CI on either AUC alone. This is what actually lets
   us detect sub-0.01 effects, and it reports an interval rather than a verdict.

Arms (all paired: identical folds, identical model seeds, one variable changed):
  A0_base       h=256, shared attention                      (reference)
  A1_geneattn   h=256, per-gene gate + per-gene head
  A2_h128       h=128, shared attention   (lr=3e-4; never tested before)
  A3_featreg    h=256, shared attention + feature-space reg (WSI-uniform noise + patch
                dropout; the only valid regulariser left once geometric augmentation is
                ruled out by working on pre-extracted embeddings)
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST, LR, WD = 1536, 32, 512, 3e-4, 1e-3
dev = "cuda"
N_FOLDSEED = 8
FOLD_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
MODEL_SEEDS = [0, 1, 2]
NFOLD, EPOCH, MIN_POS = 5, 25, 8
B_BOOT = 2000

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
proj_all = man.set_index("case_id")["project"].to_dict()
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
rm = dict(zip(idx.case_id, idx.mem_row))
cases = sorted(c for c in rm if proj_all.get(c) == "LUAD" and c in lab.index)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
rows = np.array([rm[c] for c in cases])
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
K = len(GENES)
ok_genes = [j for j in range(K) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]
print("[data] LUAD %d ; 阳性 %s" % (len(cases), {g: int(Y[:, j].sum()) for j, g in enumerate(GENES)}), flush=True)
print("[data] 可报基因 %s" % [GENES[j] for j in ok_genes], flush=True)
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
         for fs in FOLD_SEEDS[:N_FOLDSEED]]
print("[folds] %d 个折划分种子 x %d 折, 全部 arm 共用" % (len(FOLDS), NFOLD), flush=True)


class NetShared(nn.Module):
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
        s.drop = nn.Dropout(drop)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


class NetGeneAttn(nn.Module):
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Parameter(torch.randn(k, h, h) * (h ** -0.5))
        s.U = nn.Parameter(torch.randn(k, h, h) * (h ** -0.5))
        s.w = nn.Parameter(torch.randn(k, h) * (h ** -0.5))
        s.oh = nn.Parameter(torch.randn(k, h) * (h ** -0.5))
        s.b = nn.Parameter(torch.zeros(k))
    def forward(s, x):
        z = s.trunk(x)
        g = torch.tanh(torch.einsum("bsh,khg->kbsg", z, s.V)) * \
            torch.sigmoid(torch.einsum("bsh,khg->kbsg", z, s.U))
        a = torch.softmax(torch.einsum("kbsg,kg->kbs", g, s.w), dim=2)
        p = torch.einsum("kbs,bsh->kbh", a, z)
        return (p * s.oh[:, None, :]).sum(-1).t() + s.b


ARMS = [
    dict(name="A0_base",     net=NetShared,   h=256, drop=0.25, reg=False),
    dict(name="A1_geneattn", net=NetGeneAttn, h=256, drop=0.25, reg=False),
    dict(name="A2_h128",     net=NetShared,   h=128, drop=0.25, reg=False),
    dict(name="A3_featreg",  net=NetShared,   h=256, drop=0.25, reg=True),
]
print("\n%-14s %10s %10s" % ("arm", "params", "per-case"), flush=True)
for a in ARMS:
    n = sum(p.numel() for p in a["net"](DIM, a["h"], K, a["drop"]).parameters())
    a["params"] = n
    print("%-14s %10d %10.1f" % (a["name"], n, n / len(cases)), flush=True)


def batch(rws, rng):
    bt = np.empty((len(rws), S_INST, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    # NO pixel normalisation here. These caches are already standardised embeddings
    # (measured: mean 0.008 / std 0.79), not 0-255 pixels. Dividing by 255 again
    # crushed the std to 0.006 and every arm collapsed to AUC ~0.46.
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr, ytr, te, cfg, seed):
    torch.manual_seed(seed)
    net = cfg["net"](DIM, cfg["h"], K, cfg["drop"]).to(dev)
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
            xb = batch(tr[rr], rng)
            if cfg["reg"]:
                # WSI-uniform multiplicative noise: one sigma per SLIDE, not per patch
                # (HistAug's "uniform WSI-wise augmentation" insight), plus patch dropout.
                sig = torch.randn(xb.shape[0], 1, 1, device=dev) * 0.05
                xb = xb * torch.exp(sig)
                keep = (torch.rand(xb.shape[0], xb.shape[1], 1, device=dev) > 0.15).to(xb.dtype)
                xb = xb * keep
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        return net(batch(te, np.random.RandomState(seed + 999))).float().cpu().numpy()


def mean_auc(P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in ok_genes]))


def paired_boot(PA, PB, B=B_BOOT):
    """CI on mean_AUC(A) - mean_AUC(B) resampling the SAME 436 cases for both arms."""
    rng = np.random.RandomState(0)
    n = len(Y)
    d = np.empty(B)
    for b in range(B):
        s = rng.randint(0, n, n)
        ya, yb_ = Y[s], Y[s]
        d[b] = np.mean([roc_auc_score(ya[:, j], PA[s, j]) for j in ok_genes]) - \
               np.mean([roc_auc_score(yb_[:, j], PB[s, j]) for j in ok_genes])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
           float((d > 0).mean())


print("\n" + "=" * 100, flush=True)
print("=== 5-fold OOF x %d 个折划分种子 x 3 个模型种子, 完全配对 ===" % len(FOLDS), flush=True)
print("=" * 100, flush=True)
oof = {a["name"]: np.zeros((len(FOLDS), len(cases), K), np.float32) for a in ARMS}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    for tri, tei in folds:
        for a in ARMS:
            oof[a["name"]][fi][tei] = np.mean([fit(rows[tri], Y[tri], rows[tei], a, s) for s in MODEL_SEEDS], 0)
    line = "  foldseed %d: " % fi
    for a in ARMS:
        line += "%s=%.4f  " % (a["name"], mean_auc(oof[a["name"]][fi]))
    print(line + "[%.0fs]" % (time.time() - T0), flush=True)
    np.savez_compressed(os.path.join(C, "decisive_oof.npz"),
                        **{k: v for k, v in oof.items()}, Y=Y)

print("\n=== 每个 arm 的 OOF mean (跨 %d 个折划分种子) ===" % len(FOLDS), flush=True)
per_seed = {}
for a in ARMS:
    m = [mean_auc(oof[a["name"]][i]) for i in range(len(FOLDS))]
    per_seed[a["name"]] = m
    print("  %-14s %.4f +- %.4f  (min %.4f max %.4f)   params=%d"
          % (a["name"], np.mean(m), np.std(m, ddof=1), min(m), max(m), a["params"]), flush=True)
print("\n  新的判定阈值 2sigma = %.4f (8 个折划分种子, 噪声底从 0.0070 降到 %.4f)"
      % (2 * np.std(per_seed["A0_base"], ddof=1), np.std(per_seed["A0_base"], ddof=1)), flush=True)

print("\n=== 配对 bootstrap: delta-AUC 的 95%% CI (重采样同一批 436 例) ===", flush=True)
PAvg = {a["name"]: oof[a["name"]].mean(0) for a in ARMS}
rows_out = []
for a in ARMS[1:]:
    md, lo, hi, ppos = paired_boot(PAvg[a["name"]], PAvg["A0_base"])
    sig = "**显著**" if lo > 0 or hi < 0 else "不显著"
    print("  %-14s - A0_base  d=%+.4f  95%%CI [%+.4f, %+.4f]  P(d>0)=%.3f  %s"
          % (a["name"], md, lo, hi, ppos, sig), flush=True)
    rows_out.append(dict(arm=a["name"], params=a["params"], auc=np.mean(per_seed[a["name"]]),
                         auc_std=np.std(per_seed[a["name"]], ddof=1), d=md, ci_lo=lo, ci_hi=hi, p_pos=ppos))
# A0 自己对 A0 的自举, 作为"这个方法能测出多小的效应"的标尺
md, lo, hi, ppos = paired_boot(PAvg["A0_base"], PAvg["A0_base"])
print("  %-14s - A0_base  (自举对照, 应恒为 0)  d=%+.4f  CI [%+.4f, %+.4f]" % ("A0_base", md, lo, hi), flush=True)
pd.DataFrame(rows_out).to_csv(os.path.join(C, "decisive_arch.csv"), index=False)
print("\n[saved] decisive_oof.npz / decisive_arch.csv", flush=True)
print("DONE", flush=True)