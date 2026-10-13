# -*- coding: utf-8 -*-
"""76_same_encoder_aug.py - the decisive test: does more SAME-CANCER data help?

Why: 53b reported CPTAC augmentation +0.0068; section 20 retracted it (below the 0.014
noise floor). But 53b only added 216 cases on top of 436 -- too small to resolve.
Here the same question is asked on the one clean configuration we actually have:
two LUAD cohorts, SAME encoder (UNI v1), same protocol, fully paired folds.

Arms (TCGA folds are identical across A and B, so the OOF comparison is paired):
  A  train = TCGA-LUAD only (444)
  B  train = TCGA-LUAD + CPTAC-LUAD (571)   -> evaluated on the SAME held-out TCGA fold
  external: A and B each trained on their full set, evaluated once on CPTAC-LUAD (127)

Implementation note (a bug that invalidated the first run of this script):
the first version picked the feature array from a flag describing the TEST rows, so
the 127 CPTAC training rows were used to index the TCGA array -- silently reading other
slides' features, with no out-of-bounds error because both arrays are large enough.
Fix: concatenate the two caches into ONE array and remap every case to a single global
row index, so a row index can never be ambiguous.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, S_INST, LR, WD = 1024, 32, 512, 3e-4, 1e-3
dev = "cuda"
FOLD_SEEDS, MODEL_SEEDS, NFOLD = [101, 202, 303], [0, 1, 2], 5
CFG = dict(h=256, ep=25, drop=0.25)
MIN_POS = 8

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
proj_all = man.set_index("case_id")["project"].to_dict()

ui = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
rm_t = dict(zip(ui.case_id, ui.mem_row))
t_cases = sorted(c for c in rm_t if proj_all.get(c) == "LUAD" and c in lab.index)
Y_t = np.array([[int(lab.loc[c, g]) for g in GENES] for c in t_cases])

ci = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
rm_c = dict(zip(ci.case_id, ci.mem_row))
cp = pd.read_csv(os.path.join(C, "cptac_luad_usable.csv"))
c_cases = list(cp.case_id)
assert all(c in rm_c for c in c_cases), "CPTAC case without features"
Y_c = cp[GENES].values.astype(int)

Xt = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))
Xc = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"))
OFF = len(Xt)
X = np.concatenate([Xt, Xc], axis=0)        # one array, one global row space
del Xt, Xc
g_t = np.array([rm_t[c] for c in t_cases])          # global rows for TCGA cases
g_c = OFF + np.array([rm_c[c] for c in c_cases])    # global rows for CPTAC cases

print("[data] TCGA-LUAD %d  CPTAC-LUAD %d  合计 %d" % (len(t_cases), len(c_cases), len(t_cases) + len(c_cases)), flush=True)
print("[data] TCGA  阳性 %s" % {g: int(Y_t[:, j].sum()) for j, g in enumerate(GENES)}, flush=True)
print("[data] CPTAC 阳性 %s" % {g: int(Y_c[:, j].sum()) for j, g in enumerate(GENES)}, flush=True)
print("[data] 队列重叠 case = %d" % len(set(t_cases) & set(c_cases)), flush=True)
print("[ram] 拼接后 X = %s  %.0f MB" % (X.shape, X.nbytes / 1e6), flush=True)
# sanity: the two cohorts must NOT be trivially identical rows
chk = X[g_t[:50]].reshape(50, -1).astype(np.float32)
chk2 = X[g_c[:50]].reshape(50, -1).astype(np.float32)
print("[check] 前后各取 50 例, 均值绝对差 = %.4f (接近 0 才说明行号映射错了)"
      % float(np.abs(chk.mean(0) - chk2.mean(0)).mean()), flush=True)

tier = (Y_t[:, GENES.index("TP53")].astype(str) + "_" + Y_t[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(t_cases)), tier)]
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


def batch(rows, rng):
    bt = np.empty((len(rows), S_INST, DIM), np.float16)
    for i, r in enumerate(rows):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr_rows, tr_y, te_rows, seed):
    torch.manual_seed(seed)
    net = Net(DIM, CFG["h"], len(GENES), CFG["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=CFG["ep"],
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(CFG["ep"]):
        net.train(); perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(batch(tr_rows[rr], rng)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return net(batch(te_rows, np.random.RandomState(seed + 999))).float().cpu().numpy()


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


print("\n" + "=" * 96, flush=True)
print("=== 同编码器(UNI v1) 同癌种(LUAD) 增训: A=TCGA444  B=+CPTAC127 ===")
print("=" * 96, flush=True)
res = {"A": [], "B": []}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof = {"A": np.zeros((len(t_cases), len(GENES))), "B": np.zeros((len(t_cases), len(GENES)))}
    for tri, tei in folds:
        for name, extra in [("A", False), ("B", True)]:
            rows, y = g_t[tri], Y_t[tri]
            if extra:
                rows = np.concatenate([rows, g_c]); y = np.concatenate([y, Y_c])
            oof[name][tei] = np.mean([fit(rows, y, g_t[tei], s) for s in MODEL_SEEDS], 0)
    for name in ["A", "B"]:
        res[name].append(mean_auc(Y_t, oof[name]))
    print("  foldseed %d: A=%.4f  B=%.4f  (B-A=%+.4f)  [%.0fs]"
          % (fi, res["A"][-1], res["B"][-1], res["B"][-1] - res["A"][-1], time.time() - T0), flush=True)

print("\n=== CPTAC-LUAD 127 例上的评估 (注意: B 的训练集就含这 127 例, 那一行是 in-sample) ===", flush=True)
ext = {}
for name, extra in [("A", False), ("B", True)]:
    rows, y = g_t, Y_t
    if extra:
        rows = np.concatenate([rows, g_c]); y = np.concatenate([y, Y_c])
    P = np.mean([fit(rows, y, g_c, s) for s in MODEL_SEEDS], 0)
    ext[name] = mean_auc(Y_c, P)
    per = [roc_auc_score(Y_c[:, j], P[:, j]) if MIN_POS <= Y_c[:, j].sum() <= len(Y_c) - MIN_POS else np.nan
           for j in range(len(GENES))]
    tag = "IN-SAMPLE(不是外验)" if extra else "外验(合法)"
    print("  %s train=%3d  CPTAC mean=%.4f  [%s]  %s" % (name, len(rows), ext[name], tag,
          "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a) for g, a in zip(GENES, per))), flush=True)

d = [b - a for a, b in zip(res["A"], res["B"])]
print("\n  TCGA OOF  A(仅 TCGA 444)  = %.4f +- %.4f" % (np.mean(res["A"]), np.std(res["A"], ddof=1)), flush=True)
print("  TCGA OOF  B(+CPTAC 127)   = %.4f +- %.4f" % (np.mean(res["B"]), np.std(res["B"], ddof=1)), flush=True)
print("  配对差 B-A = %+.4f  per-fold-seed=%s  同号 %d/3"
      % (np.mean(d), " ".join("%+.4f" % x for x in d), sum(1 for x in d if x > 0)), flush=True)
print("  噪声底 2sigma = 0.0140  ->  %s" % ("成立: 同癌种增训有效" if np.mean(d) > 0.014
                                          else "不成立: 落在噪声内"), flush=True)
print("\n  CPTAC 外验(只有 A 合法): A=%.4f ; B=%.4f 是 in-sample, 不可比" % (ext["A"], ext["B"]), flush=True)
pd.DataFrame(dict(fold_seed=FOLD_SEEDS, A=res["A"], B=res["B"], delta=d,
                  ext_A=ext["A"], ext_B=ext["B"])).to_csv(os.path.join(C, "same_encoder_aug.csv"), index=False)
print("[saved] same_encoder_aug.csv", flush=True)
print("DONE", flush=True)