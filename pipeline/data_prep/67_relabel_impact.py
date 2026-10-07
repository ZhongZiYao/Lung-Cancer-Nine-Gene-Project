# -*- coding: utf-8 -*-
"""67_relabel_impact.py - rebuild labels from cBioPortal PanCanAtlas and measure the impact

Finding that motivated this (66_label_reconcile.py):
  our MC3-derived labels agree with cBioPortal PanCanAtlas on 99.12% of gene-case pairs,
  and the disagreement is UNIDIRECTIONAL: 46 we call wild-type but cBioPortal calls mutated,
  0 the other way. 23 of those are LUAD cases that are in our training set (5.3%).

Why adopt cBioPortal:
  1. It is the de facto standard for computational pathology papers on TCGA WSI mutation
     prediction, so our numbers become comparable to the published baselines.
  2. It is strictly more inclusive, i.e. calling a case positive when any source sees a
     driver mutation is the safer error for a screening-style task.

Protocol: same as 60_robust_config.py (gigap, h=256/ep=25/drop=0.25, S_INST=512,
3 fold-seeds x 3 model-seeds x 5 folds, fully paired), run on OLD vs NEW labels.
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
FOLD_SEEDS, MODEL_SEEDS, NFOLD = [101, 202, 303], [0, 1, 2], 5
CFG = dict(h=256, ep=25, drop=0.25)

# ---------------- build new labels ----------------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
cb = pd.read_csv(os.path.join(C, "cba_luad_lusc_6gene_labels.csv"))
cb["case_id"] = cb.case_id.astype(str)
cbd = cb.set_index("case_id")
idxf = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
feat = set(idxf.case_id)

proj_all = man.set_index("case_id")["project"].to_dict()
luad_all = sorted(c for c in feat if proj_all.get(c) == "LUAD")
luad = [c for c in luad_all if c in lab.index]          # paired set: both arms same cases
orphan = [c for c in luad_all if c not in lab.index]    # cBioPortal knows, we had no label
print("[data] LUAD with gigap features: %d (paired set %d, orphan-no-label %d)"
      % (len(luad_all), len(luad), len(orphan)), flush=True)

Yold = np.array([[int(lab.loc[c, g]) for g in GENES] for c in luad])
new = Yold.copy()
n_up, in_cb = 0, 0
for i, c in enumerate(luad):
    if c in cbd.index:
        in_cb += 1
        row = cbd.loc[c]
        for j, g in enumerate(GENES):
            v = int(row[g])
            if v == 1 and new[i, j] == 0:
                new[i, j] = 1; n_up += 1
print("[labels] cBioPortal 能查到 %d / %d ; 标签位被改 0->1 共 %d 个"
      % (in_cb, len(luad), n_up), flush=True)
for j, g in enumerate(GENES):
    print("   %-6s pos %d -> %d  (+%d)" % (g, int(Yold[:, j].sum()), int(new[:, j].sum()),
                                           int(new[:, j].sum() - Yold[:, j].sum())), flush=True)

# also add the cases that only cBioPortal knows about (all wild-type)
print("[labels] cBioPortal 独有、本地无标签的 LUAD (%d): %s" % (len(orphan), orphan), flush=True)
print("   它们在 cBioPortal 的 6 基因状态: %s"
      % ({c: [int(cbd.loc[c, g]) for g in GENES] for c in orphan if c in cbd.index}), flush=True)

np.save(os.path.join(C, "luad_Y_old.npy"), Yold)
np.save(os.path.join(C, "luad_Y_new.npy"), new)
pd.Series(luad).to_csv(os.path.join(C, "luad_case_list.csv"), index=False, header=["case_id"])
print("[saved] luad_Y_old.npy / luad_Y_new.npy / luad_case_list.csv", flush=True)

tier = (new[:, GENES.index("TP53")].astype(str) + "_" + new[:, GENES.index("NF1")].astype(str))
tier_old = (Yold[:, GENES.index("TP53")].astype(str) + "_" + Yold[:, GENES.index("NF1")].astype(str))
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
rowmap = dict(zip(idxf.case_id, idxf.mem_row))
rows_all = np.array([rowmap[c] for c in luad])
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


def fit(tri, tei, Y, seed):
    torch.manual_seed(seed)
    tr_rows, tr_y, te_rows = rows_all[tri], Y[tri], rows_all[tei]
    net = Net(DIM, CFG["h"], len(GENES), CFG["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=CFG["ep"],
                                              steps_per_epoch=max(1, len(tri) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(CFG["ep"]):
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


FOLDS = {}
for tag, tv in [("old", tier_old), ("new", tier)]:
    FOLDS[tag] = [[(a, b) for a, b in
                   StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(luad)), tv)]
                  for fs in FOLD_SEEDS]

res = {}
print("\n" + "=" * 92, flush=True)
for tag, Y in [("old", Yold), ("new", new)]:
    means, per_gene = [], []
    for fi, folds in enumerate(FOLDS[tag]):
        oof = np.zeros((len(luad), len(GENES)))
        for k, (tri, tei) in enumerate(folds):
            oof[tei] = np.mean([fit(tri, tei, Y, s) for s in MODEL_SEEDS], 0)
            print("      [%s] foldseed %d fold %d done" % (tag, fi, k), flush=True)
        per = [float(roc_auc_score(Y[:, j], oof[:, j])) for j in range(len(GENES))]
        means.append(float(np.mean(per))); per_gene.append(per)
    res[tag] = means
    print("  labels=%-4s OOF per fold-seed: %s   mean=%.4f  std=%.4f"
          % (tag, " ".join("%.4f" % v for v in means), np.mean(means), np.std(means, ddof=1)), flush=True)
    print("     per-gene mean: %s" % " ".join("%s=%.3f" % (g, np.mean([p[j] for p in per_gene]))
                                               for j, g in enumerate(GENES)), flush=True)

d = [a - b for a, b in zip(res["new"], res["old"])]
print("\n  paired delta (new - old): mean=%+.4f  per-seed=%s  same-sign %d/3"
      % (np.mean(d), " ".join("%+.4f" % x for x in d), sum(1 for x in d if x > 0)), flush=True)
print("  noise floor 2sigma = 0.0140 ->", "STILL成立" if abs(np.mean(d)) > 0.014 else "落在噪声内, 不构成结论", flush=True)
pd.DataFrame(dict(fold_seed=FOLD_SEEDS, old=res["old"], new=res["new"], delta=d)).to_csv(
    os.path.join(C, "relabel_impact.csv"), index=False)
print("[saved] relabel_impact.csv", flush=True)
print("DONE", flush=True)