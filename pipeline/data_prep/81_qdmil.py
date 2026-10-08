# -*- coding: utf-8 -*-
"""81_qdmil.py — QD-MIL: Query aggregator + label disambiguation, on ALL cached patches

What we borrow from DeepGEM (TencentAILabHealthcare/DeepGEM, read from source):

  1) WSIAggregator — K learnable query embeddings that CROSS-ATTEND to all patch tokens
     (DETR-style), instead of a single softmax-weighted average.  The WSI classifier
     reads the concatenation of all K query outputs.
     WHY this matters here: §28 showed that HARD selection is what kills us
     (max 0.512 < top5% 0.519 < top10% 0.525 < mean 0.539). A query is not a top-k
     selector: each of the K queries learns its own soft distribution over ALL patches
     and the classifier sees all K summaries at once. It is "learned soft multi-view
     summarisation", which is the one family §28 did not rule out.
     (DeepGEM's own numbers: FEATURE_LEN=500 patches, hidden 64, depth 3, 5 queries.)

  2) PartialLoss label disambiguation — a per-(sample, class) confidence C[i,c] that
     multiplies the loss; C is EMA-updated from whether the model agrees with the label,
     with the EMA coefficient annealed 0.95 -> 0.80.

What we change (the innovation):

  a) MULTI-LABEL adaptation. DeepGEM trains one binary model per gene
     (`--gene EGFR`), so its C is a single scalar per sample. We train all 6 genes
     jointly, so C must be (N, 6) and the loss becomes
         L = mean_{i,g}  C[i,g] * BCE(logit[i,g], y[i,g]) * pos_weight[g]
     which lets a slide be "ambiguous for TP53" while still fully trusted for KRAS.

  b) FLOOR on the confidence. DeepGEM's pseudo_label is one-hot(argmax(p * onehot(y))),
     i.e. exactly 0 whenever the model disagrees -> that sample/class pair's loss is
     annihilated. We use  C <- m*C + (1-m)*(FLOOR + (1-FLOOR)*agree)  with FLOOR=0.3,
     so a label is only ever down-weighted, never deleted. With 436 cases, deleting
     training signal is far more expensive than tolerating a bad label.

  c) TWO-VIEW DISAGREEMENT (the actual new idea). DeepGEM's confidence is
     self-confirmation: the model is most confident on exactly what it fits, so the
     signal is circular. We have two architecturally different views of the same bag —
     the gated-attention pooling and the query aggregator. We down-weight a
     (sample, gene) pair by how much the two views DISAGREE, not by how confident
     either one is:
         d[i,g] = |sigmoid(logit_gated[i,g]) - sigmoid(logit_query[i,g])|
         target = 1 - lam * min(1, d[i,g] / tau)
     Rationale: two genuinely different read-outs of the same tissue reaching opposite
     conclusions is much better evidence of an ambiguous/erroneous label than a single
     model's low confidence, which is dominated by class imbalance.

Arms (all fully paired: same folds, same seeds):
  N0_base       gated only, S=512                    -- our current best
  N1_q          gated + query, S=1024
  N2_qd         + DeepGEM-style per-(sample,gene) disambiguation, S=1024
  N3_qd2        + two-view disagreement, S=1024
  N4_qd2        same as N3 but S=4096 (all cached patches)
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, LR, WD = 1536, 16, 3e-4, 1e-3
dev = "cuda"
FOLD_SEEDS = [101, 202, 303]
MODEL_SEEDS = [0, 1]
NFOLD, EPOCH, MIN_POS = 5, 25, 8
KQ = 5                 # queries, DeepGEM default
FLOOR = 0.3
TAU, LAM = 0.5, 0.5
M_START, M_END = 0.95, 0.80
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
print("[data] %d LUAD ; 源 patch 中位 %d" % (N, i4.set_index("case_id").loc[cases].n_patch_orig.median()), flush=True)
X = np.load(os.path.join(C, "NSCLC_gigap_sub4096.npy"), mmap_mode="r")
print("[ram] mmap %.0f MB (可取最多 %d patch/例)" % (
    os.path.getsize(os.path.join(C, "NSCLC_gigap_sub4096.npy")) / 1e6, X.shape[1]), flush=True)
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(N), tier)]
         for fs in FOLD_SEEDS]


class QD(nn.Module):
    def __init__(s, d=1536, h=256, k=K, nq=KQ, drop=0.25, use_query=True):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False)
        s.od = nn.Linear(h, k)
        s.use_query = use_query
        if use_query:
            s.query = nn.Parameter(torch.randn(nq, h) * 0.02)
            s.mha = nn.MultiheadAttention(h, 4, dropout=drop, batch_first=True)
            s.ln = nn.LayerNorm(h)
            s.odq = nn.Linear(h * nq, k)

    def forward(s, x):
        z = s.trunk(x)                                              # (B,S,h)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        lg = s.od((z * a.unsqueeze(-1)).sum(1))                      # (B,k) gated view
        if not s.use_query:
            return lg, lg, None
        q = s.query.unsqueeze(0).expand(z.shape[0], -1, -1)         # (B,nq,h)
        o, _ = s.mha(q, z, z, need_weights=False)                  # (B,nq,h)
        o = s.ln(o + q)
        lq = s.odq(o.flatten(1))                                    # (B,k) query view
        return lg, lq, a


def make_batch(rws, S, rng):
    bt = np.empty((len(rws), S, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; n = v.shape[0]
        bt[i] = v[rng.choice(n, S, replace=n < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


ARMS = [
    dict(name="N0_base",     S=512,  query=False, dis="none"),
    dict(name="N1_q",        S=1024, query=True,  dis="none"),
    dict(name="N2_qd",       S=1024, query=True,  dis="deepgem"),
    dict(name="N3_qd2",      S=1024, query=True,  dis="twoview"),
    dict(name="N4_qd2_4096", S=4096, query=True,  dis="twoview"),
]
print("\n%-14s %6s %6s %10s" % ("arm", "S_INST", "queries", "params"), flush=True)
for a in ARMS:
    n = sum(p.numel() for p in QD(use_query=a["query"]).parameters())
    a["params"] = n
    print("%-14s %6d %6s %10d" % (a["name"], a["S"], KQ if a["query"] else "-", n), flush=True)


def fit(tr_ci, tr_fr, ytr, te_fr, cfg, seed):
    torch.manual_seed(seed)
    net = QD(use_query=cfg["query"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCH, steps_per_epoch=max(1, len(tr_ci) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(ytr.sum(0) > 0, (len(ytr) - ytr.sum(0)) / np.maximum(ytr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    conf = torch.ones(N, K, device=dev)          # per-(case, gene) confidence
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        m = M_START + (M_END - M_START) * ep / max(1, EPOCH - 1)     # anneal 0.95 -> 0.80
        net.train(); perm = rng.permutation(len(tr_ci))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(ytr[rr].astype(np.float32)).to(dev)
            xb = make_batch(tr_fr[rr], cfg["S"], rng)
            # tr_ci are CASE indices (0..N-1); tr_fr are FEATURE row indices.
            # Indexing conf with feature rows blew past its N=436 bound.
            ci = torch.from_numpy(tr_ci[rr].astype(np.int64)).to(dev)
            lg, lq, _ = net(xb)
            logits = lg + lq
            if cfg["dis"] != "none":
                with torch.no_grad():
                    pl = torch.sigmoid(logits)
                    agree = ((pl > 0.5).float() == yb).float()        # (B,K)
                    if cfg["dis"] == "twoview":
                        dis = (torch.sigmoid(lg) - torch.sigmoid(lq)).abs()
                        agree = agree * (1.0 - LAM * torch.clamp(dis / TAU, max=1.0))
                    tgt = FLOOR + (1 - FLOOR) * agree
                    conf[ci] = m * conf[ci] + (1 - m) * tgt
            w = conf[ci] * pw
            loss = (lf(logits, yb) * w).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        lg, lq, _ = net(make_batch(te_fr, cfg["S"], np.random.RandomState(seed + 999)))
        return (lg + lq).float().cpu().numpy(), lg.float().cpu().numpy(), lq.float().cpu().numpy()


mean_auc = lambda P: float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in ok_genes]))


def paired_boot(PA, PB, B=B_BOOT):
    rng = np.random.RandomState(0); d = np.empty(B)
    for b in range(B):
        s = rng.randint(0, N, N)
        d[b] = np.mean([roc_auc_score(Y[s, j], PA[s, j]) for j in ok_genes]) - \
               np.mean([roc_auc_score(Y[s, j], PB[s, j]) for j in ok_genes])
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


print("\n" + "=" * 100, flush=True)
print("=== QD-MIL: query aggregator + label disambiguation, 全部可用 patch ===", flush=True)
print("=" * 100, flush=True)
oof = {a["name"]: np.zeros((len(FOLDS), N, K), np.float32) for a in ARMS}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    for tri, tei in folds:
        for a in ARMS:
            ps = [fit(tri, rows[tri], Y[tri], rows[tei], a, s) for s in MODEL_SEEDS]
            oof[a["name"]][fi][tei] = np.mean([p[0] for p in ps], 0)
            if a["name"] in ("N3_qd2", "N4_qd2_4096"):
                globals().setdefault("_views", {}).setdefault(a["name"], []).append(
                    (np.mean([p[1] for p in ps], 0), np.mean([p[2] for p in ps], 0)))
    print("  foldseed %d: %s  [%.0fs]" % (fi, "  ".join("%s=%.4f" % (a["name"], mean_auc(oof[a["name"]][fi])) for a in ARMS),
                                          time.time() - T0), flush=True)

print("\n=== 汇总 (跨 %d 套折划分) ===" % len(FOLDS), flush=True)
ps = {}
for a in ARMS:
    m = [mean_auc(oof[a["name"]][i]) for i in range(len(FOLDS))]
    ps[a["name"]] = m
    print("  %-14s %.4f +- %.4f   params=%-9d S_INST=%d" % (a["name"], np.mean(m), np.std(m, ddof=1), a["params"], a["S"]), flush=True)
print("\n=== 配对 bootstrap vs N0_base ===", flush=True)
P0 = oof["N0_base"].mean(0)
rows_out = []
for a in ARMS[1:]:
    P = oof[a["name"]].mean(0)
    d, lo, hi, pp = paired_boot(P, P0)
    print("  %-14s d=%+.4f  95%%CI [%+.4f, %+.4f]  P(d>0)=%.3f  %s"
          % (a["name"], d, lo, hi, pp, "**显著**" if (lo > 0 or hi < 0) else "不显著"), flush=True)
    rows_out.append(dict(arm=a["name"], S=a["S"], params=a["params"], dis=a["dis"],
                         auc=np.mean(ps[a["name"]]), auc_std=np.std(ps[a["name"]], ddof=1),
                         d=d, ci_lo=lo, ci_hi=hi, p_pos=pp))
pd.DataFrame(rows_out).to_csv(os.path.join(C, "qdmil_sweep.csv"), index=False)
np.savez_compressed(os.path.join(C, "qdmil_oof.npz"), **{k: v for k, v in oof.items()}, Y=Y)
print("\n[saved] qdmil_sweep.csv / qdmil_oof.npz", flush=True)
print("DONE", flush=True)