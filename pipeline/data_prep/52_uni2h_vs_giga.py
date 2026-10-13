# -*- coding: utf-8 -*-
"""52_uni2h_vs_giga.py — UNI2-h vs Prov-GigaPath 配对对比 (同 case, 同头, 同协议)

为什么现在能做: UNI2-h 特征还在下载, 但已覆盖 320/486 个 LUAD case(66%),
              足够做一次有统计意义的配对比较。
做法: 取【两者都有的 case 交集】, 在同一批 case 上用完全相同的
      gated MIL 头(25 epoch / 5 seed / 同一折划分)跑两遍。
      注意 GigaPath 那侧必须限制到同一批 case, 否则不是配对比较。
"""
import os, sys, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
EP, BATCH, LR_LR, HID, S_INST = 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
CID = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"])
lab = man.set_index("case_id")

# ---- 找 UNI2-h 已下载的 LUAD 文件 ----
u2h = {}
for proj in ["TCGA-LUAD"]:
    for p in glob.glob(os.path.join(r"D:\features\uni2h_nsclc", proj, "features", "*.h5")):
        m = CID.search(os.path.basename(p))
        if m and m.group(1) in lab.index:
            u2h.setdefault(m.group(1), p)
print("UNI2-h 已下载且有标签的 LUAD case: %d" % len(u2h), flush=True)

# ---- GigaPath 索引 ----
ig = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
gmap = dict(zip(ig.case_id, ig.mem_row))
common = sorted([c for c in u2h if c in gmap])
print("两者都有的 case: %d" % len(common), flush=True)
if len(common) < 100:
    print("太少, 中止", flush=True); sys.exit(0)

# ---- 载入 UNI2-h 特征, 建 512 定长袋 ----
print("载入 UNI2-h 特征 ...", flush=True)
U = np.zeros((len(common), S_INST, 1536), np.float16)
Y = np.zeros((len(common), len(GENES)), np.int64)
keep = []
for i, c in enumerate(common):
    try:
        with h5py.File(u2h[c], "r") as h:
            f = np.asarray(h["features"][:], dtype=np.float32)
        f = f.reshape(-1, f.shape[-1])
        m = f.shape[0]
        if m < 256:
            continue
        sel = np.random.RandomState(abs(hash(c)) % (2 ** 31)).choice(m, S_INST, replace=m < S_INST)
        U[keep.__len__()] = f[sel].astype(np.float16)
        keep.append(c)
    except Exception as e:
        print("  [skip] %s %s" % (c, str(e)[:70]), flush=True)
U = U[:len(keep)]
common = keep
Y = lab.loc[common, GENES].values.astype(int)
print("可用 case: %d   U=%s" % (len(common), U.shape), flush=True)
for g, v in zip(GENES, Y.mean(0)):
    print("   %-7s prev=%.3f  n+=%d" % (g, v, int((lab.loc[common, GENES].values[:, GENES.index(g)] == 1).sum())), flush=True)

# ---- GigaPath 特征, 限制到同一批 case ----
XG = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"), mmap_mode="r")
rows_g = np.array([gmap[c] for c in common])
G = np.ascontiguousarray(XG[rows_g])
del XG
print("G=%s" % (G.shape,), flush=True)

# preflight: 线性探针
from sklearn.linear_model import LogisticRegression
def lp(M, y):
    X = M.astype(np.float32).mean(1); o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        o[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, o)
print("\n[preflight] LUAD 线性探针 TP53:", flush=True)
for nm, M in [("uni2h", U), ("gigap", G)]:
    a = lp(M, Y[:, 1]); print("   %-6s %.3f" % (nm, a), flush=True)
    assert a > 0.55, "%s 映射可疑" % nm

class Net(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))

def run(M, seed):
    S, D = M.shape[1], M.shape[2]
    N = M.shape[0]
    oof = np.zeros((N, len(GENES)), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(D, HID, len(GENES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr)//BATCH))
        rng = np.random.RandomState(seed*100+f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        def gb(rr, r):
            bt = np.empty((len(rr), S, D), np.float16)
            for k2, rw in enumerate(rr):
                v = M[rw]; m = v.shape[0]
                bt[k2] = v[r.choice(m, S, replace=m < S)]
            return torch.from_numpy(bt).to(dev, torch.float32)
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm)-BATCH+1, BATCH):
                rr = perm[i:i+BATCH]
                yb = torch.from_numpy(Ytr[rr].astype(np.float32)).to(dev)
                loss = lf(net(gb(tr[rr], rng)), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            oof[te] = net(gb(te, np.random.RandomState(7))).float().cpu().numpy()
    return oof

auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y)-10 else np.nan
res, store = {}, {}
for nm, M in [("gigap", G), ("uni2h", U)]:
    res[nm] = np.zeros((len(SEEDS), len(GENES)), np.float32); store[nm] = []
    for seed in SEEDS:
        op = run(M, seed)
        res[nm][seed] = [auc(Y[:, j], op[:, j]) for j in range(len(GENES))]
        store[nm].append(op)
        print("  [%s] seed %d mean=%.4f" % (nm, seed, np.mean(res[nm][seed])), flush=True)

print("\n" + "="*94)
print("=== UNI2-h vs Prov-GigaPath (同一批 %d 例 LUAD, 同头 gated MIL, 5 seed) ===" % len(common))
print("="*94)
base_s = res["gigap"].mean(1)
print("%-8s" % "encoder" + "".join("%9s" % g for g in GENES) + "%10s%11s%7s%10s" % ("mean", "delta", "same", "ens"))
summary = []
for nm in ["gigap", "uni2h"]:
    m = res[nm].mean(0); d = res[nm].mean(1) - base_s
    ens = np.mean(store[nm], axis=0)
    a_ens = [auc(Y[:, j], ens[:, j]) for j in range(len(GENES))]
    print("%-8s" % nm + "".join("%9.3f" % v for v in m)
          + "%10.4f%11.4f%6d/5%10.4f" % (np.mean(m), d.mean(), int((d>0).sum()), np.mean(a_ens)), flush=True)
    summary.append(dict(encoder=nm, mean_auc=float(np.mean(m)), delta=float(d.mean()),
                        agree=int((d>0).sum()), ens_auc=float(np.mean(a_ens)),
                        **{"auc_"+g: float(v) for g, v in zip(GENES, m)}))
print("\n判定: %s" % ("UNI2-h 有效提升" if summary[1]["delta"] > 0.01 and summary[1]["agree"] >= 4
                     else ("有害" if summary[1]["delta"] < -0.01 else "噪声")), flush=True)
pd.DataFrame(summary).to_csv(os.path.join(C, "uni2h_vs_giga.csv"), index=False)
print("[saved] %s" % os.path.join(C, "uni2h_vs_giga.csv"), flush=True)
print("DONE", flush=True)