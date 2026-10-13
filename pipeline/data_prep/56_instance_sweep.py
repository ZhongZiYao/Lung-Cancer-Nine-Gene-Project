# -*- coding: utf-8 -*-
"""56_instance_sweep.py — 每张切片喂给 MIL 的 patch 实例数 S_INST 扫描

为什么做这个:
  此前 19 项实验全部固定 S_INST=512, 从没扫过这个维度。而缓存里每例的中位
  patch 数是 5000~11000, 也就是说我们一直在丢掉 90% 以上的组织再让 gated
  attention 去挑。MIL 的聚合质量通常随实例数上升, 这是一个被漏掉的大维度。

做法:
  1) 先从已有的 NSCLC_gigap_sub4096.npy 建一个 LUAD-only 的 sub2048 缓存
     (434 例 x 2048 x 1536 fp16 = 2.7 GB, 可常驻 RAM)
  2) 复用 55 号脚本落盘的 split_luad.csv, 同一套 train/val/test
  3) S_INST in {512, 1024, 2048}, 每个 arm 5 seed 配对, 只在 val 上比较
     胜出的那个再单独跑一次 test
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH = 1536, 32
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
MIN_POS_VAL, MIN_POS_TEST = 8, 10
HP = dict(h=512, lr=3e-4, drop=0.25, ep=40)   # 55 号在 gigap 上选出来的

SRC = os.path.join(C, "NSCLC_gigap_sub4096.npy")
SRC_IDX = os.path.join(C, "NSCLC_gigap_sub4096_index.csv")
DST = os.path.join(C, "NSCLC_gigap_luad_sub2048.npy")
DST_IDX = os.path.join(C, "NSCLC_gigap_luad_sub2048_index.csv")
S_BUILD = 2048

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")

sp = pd.read_csv(os.path.join(C, "split_luad.csv"))
SPLIT_CASES = list(sp.case_id)
print("[split] split_luad.csv has %d case" % len(SPLIT_CASES), flush=True)

if os.path.exists(DST) and os.path.exists(DST_IDX):
    di = pd.read_csv(DST_IDX)
    rowmap = dict(zip(di.case_id, di.mem_row))
    print("[cache] reuse %s (%d case)" % (os.path.basename(DST), len(di)), flush=True)
else:
    si = pd.read_csv(SRC_IDX)
    srow = dict(zip(si.case_id, si.mem_row))
    common = [c for c in SPLIT_CASES if c in srow]
    print("[cache] build sub%d from sub4096, LUAD %d case" % (S_BUILD, len(common)), flush=True)
    Xs = np.load(SRC, mmap_mode="r")
    rng = np.random.RandomState(12345)
    out = np.empty((len(common), S_BUILD, DIM), np.float16)
    cnt = []
    t0 = time.time()
    for i, c in enumerate(common):
        v = Xs[srow[c]]
        m = v.shape[0]
        cnt.append(m)
        out[i] = v[rng.choice(m, S_BUILD, replace=m < S_BUILD)]
        if (i + 1) % 100 == 0:
            print("   %d/%d  %.0fs" % (i + 1, len(common), time.time() - t0), flush=True)
    del Xs
    np.save(DST, out)
    pd.DataFrame(dict(case_id=common, mem_row=range(len(common)), src_patches=cnt)).to_csv(DST_IDX, index=False)
    print("[cache] wrote %s  median src patches=%d" % (DST, int(np.median(cnt))), flush=True)
    rowmap = dict(zip(common, range(len(common))))
    del out

common = list(SPLIT_CASES)
assert all(c in rowmap for c in common), "sub2048 cache missing some split case"
Y = lab.loc[common, GENES].values.astype(int)
split = sp.set_index("case_id").loc[common, "split"].values
rows = np.array([rowmap[c] for c in common])
tr, va, te = split == "train", split == "val", split == "test"
rows_tr, rows_va, rows_te = rows[tr], rows[va], rows[te]
y_tr, y_va, y_te = Y[tr], Y[va], Y[te]
print("\n[split] train %d / val %d / test %d" % (len(rows_tr), len(rows_va), len(rows_te)), flush=True)


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


X = np.load(DST)
print("[ram] %s %.0f MB" % (os.path.basename(DST), X.nbytes / 1e6), flush=True)


def batches(rws, S, rng):
    bt = np.empty((len(rws), S, DIM), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S, replace=m < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def run(S, seed, want_test=False):
    torch.manual_seed(seed)
    net = Net(DIM, HP["h"], len(GENES), HP["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), HP["lr"], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, HP["lr"], epochs=HP["ep"],
                                              steps_per_epoch=max(1, len(rows_tr) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(y_tr.sum(0) > 0, (len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    best, best_state, bad = -1, None, 0
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for ep in range(HP["ep"]):
        net.train(); perm = rng.permutation(len(rows_tr))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(y_tr[rr].astype(np.float32)).to(dev)
            loss = lf(net(batches(rows_tr[rr], S, rng)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pv = net(batches(rows_va, S, np.random.RandomState(1))).float().cpu().numpy()
        use = [j for j in range(len(GENES)) if y_va[:, j].sum() >= MIN_POS_VAL]
        v = float(np.mean([roc_auc_score(y_va[:, j], pv[:, j]) for j in use]))
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
        if ep % 10 == 0:
            print("      S=%-5d seed=%d ep=%2d val=%.4f  (%.0fs)" % (S, seed, ep, v, time.time() - t0), flush=True)
    if not want_test:
        return best, None
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        pt = net(batches(rows_te, S, np.random.RandomState(2))).float().cpu().numpy()
    return best, pt


print("\n" + "=" * 96)
print("=== S_INST sweep: gigap / h=512 lr=3e-4 drop=0.25 ep=40 / 5 seed paired / val only ===")
print("=" * 96, flush=True)
res, seeds_v = {}, {}
for S in [512, 1024, 2048]:
    vs = [run(S, s)[0] for s in SEEDS]
    res[S] = float(np.mean(vs)); seeds_v[S] = vs
    print("   S_INST=%-5d val=%.4f  seeds=%s" % (S, res[S], " ".join("%.4f" % v for v in vs)), flush=True)

b0 = 512
print("\n  paired diff vs S=512 (same seed):")
for S in [1024, 2048]:
    d = [a - b for a, b in zip(seeds_v[S], seeds_v[b0])]
    same = sum(1 for x in d if (x > 0) == (res[S] > res[b0]))
    print("   S=%-5d mean d=%+.4f  per-seed=%s  same-sign %d/5" % (S, np.mean(d), " ".join("%+.4f" % x for x in d), same), flush=True)

bestS = max(res, key=res.get)
print("\n  winner S_INST=%d (val=%.4f), test run once on it only" % (bestS, res[bestS]), flush=True)
pt = np.mean([run(bestS, s, want_test=True)[1] for s in SEEDS], 0)
t_auc = lambda y, p: float(roc_auc_score(y, p)) if MIN_POS_TEST <= y.sum() <= len(y) - MIN_POS_TEST else np.nan
per = [t_auc(y_te[:, j], pt[:, j]) for j in range(len(GENES))]
print("  TEST S=%d  " % bestS + "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a)
                                          for g, a in zip(GENES, per)) + "  mean=%.4f" % np.nanmean(per), flush=True)
pd.DataFrame([dict(S_inst=S, val_auc=res[S]) for S in res]).to_csv(os.path.join(C, "inst_sweep_val.csv"), index=False)
np.save(os.path.join(C, "inst_sweep_test_pred.npy"), pt)
print("DONE", flush=True)