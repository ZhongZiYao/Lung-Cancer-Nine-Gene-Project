# -*- coding: utf-8 -*-
"""77_arch_size.py - model side: architecture, regularisation, and capacity matched to n=436

Context that drives every design choice here:
  * noise floor is measured (section 20): OOF mean std = 0.007 over fold-split seeds,
    so the decision threshold is 2 sigma = 0.014. Anything smaller is NOT a result.
  * 25 prior experiments already failed (pooling, loss, preprocessing, fusion, instances,
    epochs, capacity at h>=256, data augmentation with other cohorts). Do not repeat.
  * h=128 has NEVER been tested at lr=3e-4 -- every earlier h=128 arm was bound to
    lr=1e-4, which was the worst cell in the whole grid. Capacity shrink is untested.

Arms (all fully paired: same fold split, same model seeds):
  A0  baseline            h=256 drop=.25 ep=25, single shared attention
  A1  + feature-space regularisation  mixup + patch dropout
      (NOTE: geometric augmentation is impossible on pre-extracted embeddings)
  A2  per-gene attention  each gene gets its OWN V/U/w gate and its own scalar head
  A3  capacity shrink     h=128 @ lr=3e-4   (the never-tested cell)
  A4  A1 + A2
  A5  A1 + A2 + h=128 + drop=0.50   (aggressive shrink + regularisation)

Reports AUC plus ACC@Youden, SEN, SPE, and the parameter count for every arm,
because "match the parameter count to the data size" is only answerable if we
actually print the parameter counts.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST, LR, WD = 1536, 32, 512, 3e-4, 1e-3
dev = "cuda"
FOLD_SEEDS, MODEL_SEEDS, NFOLD = [101, 202, 303], [0, 1, 2], 5
EPOCH = 25
MIN_POS = 8

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
print("[data] LUAD %d 例 ; 阳性 %s" % (len(cases), {g: int(Y[:, j].sum()) for j, g in enumerate(GENES)}), flush=True)
print("[ram] %.0f MB" % (X.nbytes / 1e6), flush=True)
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
         for fs in FOLD_SEEDS]
K = len(GENES)


class Base(nn.Module):
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))


class NetShared(Base):
    """baseline: ONE shared attention head for all 6 genes"""
    def __init__(s, d, h, k, drop):
        super().__init__(d, h, k, drop)
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


class NetGeneAttn(Base):
    """per-gene attention with a per-gene scalar head (clean version)"""
    def __init__(s, d, h, k, drop):
        super().__init__(d, h, k, drop)
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
        pooled = torch.einsum("kbs,bsh->kbh", a, z)          # (k,B,h)
        return (pooled * s.oh[:, None, :]).sum(-1).t() + s.b  # (B,k)


def featreg(x, y, rng):
    """Feature-space regularisation.

    Geometric augmentation (flip / rot90 / colour jitter) is IMPOSSIBLE here: these are
    pre-extracted patch EMBEDDINGS, not images. That is a direct cost of the
    "pre-extracted features only" route and it removes the single most commonly
    reported regulariser in the WSI-MIL literature. What remains is feature-space:
      mixup     - convex mix of two slides' patch bags, labels mixed the same way
      patchdrop - drop 20% of the instances (a bag-level analogue of cutout)
      noise     - small Gaussian perturbation, acts like weight-space noise
    """
    mode = CFG_REG["mode"]
    if mode == "none":
        return x, y
    if mode == "mixup":
        lam = float(np.random.RandomState(int(rng.randint(1 << 30))).beta(0.4, 0.4))
        perm = torch.randperm(x.shape[0], device=x.device)
        return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
    if mode == "mixup+drop":
        lam = float(np.random.RandomState(int(rng.randint(1 << 30))).beta(0.4, 0.4))
        perm = torch.randperm(x.shape[0], device=x.device)
        x = lam * x + (1 - lam) * x[perm]
        y = lam * y + (1 - lam) * y[perm]
    if mode in ("patchdrop", "mixup+drop"):
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.2).to(x.dtype)
        x = x * keep
    if mode in ("noise",):
        x = x + torch.randn_like(x) * 0.05
    return x, y


CFG_REG = {"mode": "none"}


def batch(rws, rng):
    bt = np.empty((len(rws), S_INST, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    x = torch.from_numpy(bt).to(dev, torch.float32)
    return x.div_(255.0).sub_(0.5).div_(0.5)   # 0.5/0.5 normalisation of the upstream encoder


def fit(tr, ytr, te, cfg, seed):
    torch.manual_seed(seed)
    Net = cfg["net"]
    net = Net(DIM, cfg["h"], K, cfg["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), cfg["lr"], weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], epochs=EPOCH,
                                              steps_per_epoch=max(1, len(tr) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(ytr.sum(0) > 0, (len(ytr) - ytr.sum(0)) / np.maximum(ytr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    CFG_REG["mode"] = cfg["reg"]
    for ep in range(EPOCH):
        net.train(); perm = rng.permutation(len(tr))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(ytr[rr].astype(np.float32)).to(dev)
            xb, yb = featreg(batch(tr[rr], rng), yb, rng)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        return net(batch(te, np.random.RandomState(seed + 999))).float().cpu().numpy()


def metrics(Yv, P):
    """AUC + ACC@Youden + SEN + SPE per gene, then mean over reportable genes."""
    ok = [j for j in range(K) if MIN_POS <= Yv[:, j].sum() <= len(Yv) - MIN_POS]
    out = {}
    auc, acc, sen, spe = [], [], [], []
    for j in ok:
        y, p = Yv[:, j], P[:, j]
        auc.append(roc_auc_score(y, p))
        fpr, tpr, th = roc_curve(y, p)
        t = th[np.argmax(tpr - fpr)]
        yh = (p >= t).astype(int)
        acc.append(float((yh == y).mean()))
        sen.append(float(yh[y == 1].mean())); spe.append(float(yh[y == 0].mean()))
    out["auc"] = float(np.mean(auc)); out["acc"] = float(np.mean(acc))
    out["sen"] = float(np.mean(sen)); out["spe"] = float(np.mean(spe))
    out["per_auc"] = [auc[j] for j in range(len(auc))]
    return out


ARMS = [
    dict(name="A0_base",        net=NetShared,   h=256, drop=0.25, lr=3e-4, reg="none"),
    dict(name="A1_featreg",         net=NetShared,   h=256, drop=0.25, lr=3e-4, reg="mixup+drop"),
    dict(name="A2_geneattn",    net=NetGeneAttn, h=256, drop=0.25, lr=3e-4, reg="none"),
    dict(name="A3_h128",        net=NetShared,   h=128, drop=0.25, lr=3e-4, reg="none"),
    dict(name="A4_reg_gene",    net=NetGeneAttn, h=256, drop=0.25, lr=3e-4, reg="mixup+drop"),
    dict(name="A5_shrink_reg",  net=NetGeneAttn, h=128, drop=0.50, lr=3e-4, reg="mixup+drop"),
]
print("\n" + "=" * 104, flush=True)
print("=== 参数量 (这是'参数量匹配数据量'问题的核心数字) ===", flush=True)
print("%-16s %10s %10s %10s" % ("arm", "params", "per-case", "hidden"), flush=True)
for a in ARMS:
    n = sum(p.numel() for p in a["net"](DIM, a["h"], K, a["drop"]).parameters())
    a["params"] = n
    print("%-16s %10d %10.1f %10d" % (a["name"], n, n / len(cases), a["h"]), flush=True)

print("\n" + "=" * 104, flush=True)
print("=== 5-fold OOF x 3 fold-split seeds x 3 model seeds, 完全配对 ===")
print("=" * 104, flush=True)
res = {a["name"]: {"auc": [], "acc": [], "sen": [], "spe": []} for a in ARMS}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof = {a["name"]: np.zeros((len(cases), K)) for a in ARMS}
    for tri, tei in folds:
        for a in ARMS:
            oof[a["name"]][tei] = np.mean([fit(rows[tri], Y[tri], rows[tei], a, s) for s in MODEL_SEEDS], 0)
    line = "  foldseed %d: " % fi
    for a in ARMS:
        m = metrics(Y, oof[a["name"]])
        for k in ("auc", "acc", "sen", "spe"):
            res[a["name"]][k].append(m[k])
        line += "%s auc=%.4f acc=%.4f  " % (a["name"], m["auc"], m["acc"])
    print(line + "[%.0fs]" % (time.time() - T0), flush=True)

print("\n=== 汇总 (mean +- std over 3 fold-split seeds) ===", flush=True)
base = {k: np.array(res["A0_base"][k]) for k in ("auc", "acc", "sen", "spe")}
print("%-16s %-22s %-22s %-10s %s" % ("arm", "AUC", "ACC@Youden", "params", "ΔAUC vs A0 (同号)"), flush=True)
rows_out = []
for a in ARMS:
    n = a["name"]
    auc = np.array(res[n]["auc"]); acc = np.array(res[n]["acc"])
    d = auc - base["auc"]
    same = sum(1 for x, y in zip(d, base["auc"]) if (x > 0) == (np.mean(d) > 0) and x != 0)
    flag = "**可检出**" if abs(np.mean(d)) > 0.014 else "噪声内"
    print("%-16s %.4f +- %.4f    %.4f +- %.4f    %-10d %+.4f (%d/3) %s"
          % (n, auc.mean(), auc.std(ddof=1), acc.mean(), acc.std(ddof=1),
             a["params"], np.mean(d), same, flag), flush=True)
    rows_out.append(dict(arm=n, params=a["params"], h=a["h"], drop=a["drop"], reg=a["reg"],
                         auc=auc.mean(), auc_std=auc.std(ddof=1),
                         acc=acc.mean(), acc_std=acc.std(ddof=1),
                         sen=np.mean(res[n]["sen"]), spe=np.mean(res[n]["spe"]),
                         delta_auc=np.mean(d), same_sign=same, detectable=abs(np.mean(d)) > 0.014))
pd.DataFrame(rows_out).to_csv(os.path.join(C, "arch_size_sweep.csv"), index=False)
print("\n[saved] arch_size_sweep.csv", flush=True)
print("判定阈值 |dAUC| > 0.014 (2 sigma, 3 个折划分种子的配对差)", flush=True)
print("DONE", flush=True)