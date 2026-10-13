# -*- coding: utf-8 -*-
"""84_mammoth_trunk.py - swap the trunk linear layer for MAMMOTH (ICLR 2026)

MAMMOTH (OpenReview S5Io33pc78, mahmoodlab/MAMMOTH) targets exactly the layer we
have never questioned: the FIRST linear layer mapping a 1536-d patch feature to the
256-d task-specific space. Its claim is that "task-specific transformation matters
more than the choice of aggregation method" -- the exact opposite of what our
§4.5 / §10.4 / §28 measurements say.

Why it is still worth a shot:
  * The routing is a SOFTMAX OVER INSTANCES for every expert-slot:
        dispatch_weights = softmax(einsum(x, slot_embeds), dim=1)
    so each of the e*s learned slots is its own dense attention distribution.
    That is precisely the family §28 did NOT rule out -- dense softmax, not top-k.
    (Our §81 query aggregator used a hard MHA bottleneck and lost 0.050.)
  * The per-slot LoRA experts are a PARAMETER REALLOCATION, not a capacity increase.
    slot_dim=128 variants cut the trunk by 42% while adding e*s learned transformations.
    §26 showed a naive shrink to h=128 hurts (-0.0092) -- but that was a UNIFORM shrink.
  * Their ABMIL number on molecular subtyping is 72.8 -> 74.6 AUROC (+0.018),
    which is above our 0.014 detection threshold.

Arms (fully paired: same folds, same model seeds):
  M0_linear    trunk = Linear(1536,256)             393,472  (our current best)
  M1_moe_6x4   e=6 s=4 rank=16 slot_dim=128         229,896  (-42%)
  M2_moe_8x6   e=8 s=6 rank=8  slot_dim=128         224,648  (-43%)
  M3_moe_4x8   e=4 s=8 rank=8  slot_dim=256         414,352  (+5%)
"""
import os, sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from mammoth.mammoth import Mammoth

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST, LR, WD, H = 1536, 32, 512, 3e-4, 1e-3, 256
dev = "cuda"
FOLD_SEEDS = [101, 202, 303]
MODEL_SEEDS = [0, 1]
NFOLD, EPOCH, MIN_POS = 5, 25, 8
B_BOOT = 2000

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
proj_all = man.set_index("case_id")["project"].to_dict()
i4 = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub4096_index.csv"))
rm = dict(zip(i4.case_id, i4.mem_row))
cases = sorted(c for c in rm if proj_all.get(c) == "LUAD" and c in lab.index)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
rows = np.array([rm[c] for c in cases])
N, K = len(cases), len(GENES)
ok_genes = [j for j in range(K) if MIN_POS <= Y[:, j].sum() <= N - MIN_POS]
print("[data] %d LUAD case" % N, flush=True)
X = np.load(os.path.join(C, "NSCLC_gigap_sub4096.npy"), mmap_mode="r")
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(N), tier)]
         for fs in FOLD_SEEDS]

MOE = {
    "M0_linear":  None,
    "M1_moe_6x4": dict(num_experts=6, num_slots=4, lora_rank=16, slot_dim=128),
    "M2_moe_8x6": dict(num_experts=8, num_slots=6, lora_rank=8,  slot_dim=128),
    "M3_moe_4x8": dict(num_experts=4, num_slots=8, lora_rank=8,  slot_dim=256),
}


class Net(nn.Module):
    def __init__(s, d, h, k, drop, moe):
        super().__init__()
        if moe is None:
            s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        else:
            # Mammoth already contains ReLU + dropout internally, so we must NOT
            # add another activation here (the baseline arm's Sequential has one).
            s.trunk = Mammoth(input_dim=d, dim=h, num_heads=16, keep_slots=False,
                              dropout=drop, auto_rank=False, **moe)
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)

    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def make_batch(rws, rng):
    bt = np.empty((len(rws), S_INST, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; n = v.shape[0]
        bt[i] = v[rng.choice(n, S_INST, replace=n < S_INST)]
    return torch.from_numpy(bt).to(dev, torch.float32)


print("\n%-14s %10s %10s" % ("arm", "trunk", "total"), flush=True)
PARAM = {}
for name, moe in MOE.items():
    m = Net(DIM, H, K, 0.25, moe)
    tp = sum(p.numel() for p in m.trunk.parameters())
    tt = sum(q.numel() for q in m.parameters())
    PARAM[name] = tt
    print("%-14s %10d %10d" % (name, tp, tt), flush=True)


def fit(tr, ytr, te, moe, seed):
    torch.manual_seed(seed)
    net = Net(DIM, H, K, 0.25, moe).to(dev)
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
            loss = lf(net(make_batch(tr[rr], rng)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        return net(make_batch(te, np.random.RandomState(seed + 999))).float().cpu().numpy()


mean_auc = lambda P: float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in ok_genes]))


def paired_boot(PA, PB, B=B_BOOT):
    rng = np.random.RandomState(0); d = np.empty(B)
    for b in range(B):
        s = rng.randint(0, N, N)
        d[b] = np.mean([roc_auc_score(Y[s, j], PA[s, j]) for j in ok_genes]) - \
               np.mean([roc_auc_score(Y[s, j], PB[s, j]) for j in ok_genes])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


print("\n" + "=" * 96, flush=True)
print("=== MAMMOTH trunk vs Linear trunk, gated MIL 其余全部冻结 ===", flush=True)
print("=" * 96, flush=True)
oof = {n: np.zeros((len(FOLDS), N, K), np.float32) for n in MOE}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    for tri, tei in folds:
        for name, moe in MOE.items():
            ps = [fit(rows[tri], Y[tri], rows[tei], moe, s) for s in MODEL_SEEDS]
            oof[name][fi][tei] = np.mean(ps, 0)
    print("  foldseed %d: %s  [%.0fs]"
          % (fi, "  ".join("%s=%.4f" % (n, mean_auc(oof[n][fi])) for n in MOE), time.time() - T0), flush=True)

print("\n=== 汇总 ===", flush=True)
ps = {}
for n in MOE:
    m = [mean_auc(oof[n][i]) for i in range(len(FOLDS))]
    ps[n] = m
    print("  %-14s %.4f +- %.4f   params=%d" % (n, np.mean(m), np.std(m, ddof=1), PARAM[n]), flush=True)
print("\n=== 配对 bootstrap vs M0_linear ===", flush=True)
P0 = oof["M0_linear"].mean(0)
rows_out = []
for n in list(MOE)[1:]:
    d, lo, hi, pp = paired_boot(oof[n].mean(0), P0)
    print("  %-14s d=%+.4f  95%%CI [%+.4f, %+.4f]  P(d>0)=%.3f  %s"
          % (n, d, lo, hi, pp, "**显著**" if (lo > 0 or hi < 0) else "不显著"), flush=True)
    rows_out.append(dict(arm=n, params=PARAM[n], auc=np.mean(ps[n]), auc_std=np.std(ps[n], ddof=1),
                         d=d, ci_lo=lo, ci_hi=hi, p_pos=pp))
pd.DataFrame(rows_out).to_csv(os.path.join(C, "mammoth_sweep.csv"), index=False)
np.savez_compressed(os.path.join(C, "mammoth_oof.npz"), **{k: v for k, v in oof.items()}, Y=Y)
print("\n[saved] mammoth_sweep.csv / mammoth_oof.npz", flush=True)
print("DONE", flush=True)