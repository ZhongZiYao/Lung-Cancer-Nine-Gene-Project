# -*- coding: utf-8 -*-
"""57_repeated_split.py — 重复抽样 + val 加权集成

为什么要做:
  §15 的 0.7838 只来自一套划分, test 86 例, STK11 只有 13 个阳性
  (AUC 95%CI 约 0.76~0.97)。单点数字不能当结论。本脚本把 60/20/20
  拆成 5 套独立划分, 报告 test mean 的均值 ± 标准差 —— 这个 std
  直接告诉我们"换个划分结论会动多少"。

第二件事: §15.4 发现四编码器【等权】集成 0.7740 < gigap 单模型 0.7838,
  因为 chief 只有 0.697 被等权平均拖后腿。这里改成在 val 上学一组
  6 个基因【共享】的权重(3 维单纯形粗网格), 再应用到 test。
  共享而不是逐基因, 是因为 val 每基因只有 8~46 个阳性, 逐基因拟合
  4 个权重必然过拟合 val。

协议(每套划分):
  - train 60% 训练
  - val  20% 只用于早停 + 超参选择 + 集成权重拟合
  - test 20% 只在选定后跑一次, 5 seed 集成
外层循环是编码器(特征只 np.load 一次), 内层循环是 5 套划分。
"""
import os, sys, time, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
BATCH, S_INST = 32, 512
dev = "cuda"
SPLIT_SEEDS = [20261006, 20261007, 20261008, 20261009, 20261010]
PROBE_SEEDS = [0, 1, 2]          # 探超参用 3 个种子
FINAL_SEEDS = [0, 1, 2, 3, 4]    # 最终报告用 5 个种子
MIN_POS_VAL, MIN_POS_TEST = 8, 10

ENC = {}
for enc, dim in [("gigap", 1536), ("uni", 1024), ("chief", 768), ("uni2h_luad", 1536)]:
    idxf = os.path.join(C, "NSCLC_%s_sub%d_index.csv" % (enc, 512))
    npyf = os.path.join(C, "NSCLC_%s_sub%d.npy" % (enc, 512))
    if os.path.exists(idxf) and os.path.exists(npyf):
        d = pd.read_csv(idxf)
        ENC[enc] = (dict(zip(d.case_id, d.mem_row)), npyf, dim)
        print("[enc] %-10s dim=%-5d case=%d" % (enc, dim, len(d)), flush=True)

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
common = sorted(set(lab.index))
for enc in ENC:
    common = sorted(set(common) & set(ENC[enc][0].keys()))
Y = lab.loc[common, GENES].values.astype(int)
print("\n[align] LUAD common case: %d" % len(common), flush=True)

HP_GRID = [
    dict(h=256, lr=3e-4, drop=0.25, ep=40),
    dict(h=256, lr=3e-4, drop=0.50, ep=40),
    dict(h=128, lr=1e-4, drop=0.25, ep=60),
    dict(h=512, lr=3e-4, drop=0.25, ep=40),
]
t_auc = lambda y, p: float(roc_auc_score(y, p)) if MIN_POS_TEST <= y.sum() <= len(y) - MIN_POS_TEST else np.nan


def make_split(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(common))
    assign = np.empty(len(common), object)
    for tier in ["TP53", "NF1"]:
        col = GENES.index(tier)
        for v in (0, 1):
            pool = [i for i in idx if Y[i, col] == v and assign[i] is None]
            rng.shuffle(pool)
            n = len(pool); n_tr, n_va = int(round(n * 0.6)), int(round(n * 0.2))
            for k, i in enumerate(pool):
                assign[i] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    for i in idx:
        if assign[i] is None:
            assign[i] = "train"
    return assign


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


_RAM = {}
def _ram(p):
    if p not in _RAM:
        _RAM.clear(); _RAM[p] = np.load(p)
        print("   [ram] %s %.0f MB" % (os.path.basename(p), _RAM[p].nbytes / 1e6), flush=True)
    return _RAM[p]


def batches(rws, X, dim, rng, S=S_INST):
    bt = np.empty((len(rws), S, dim), np.float16)
    for i, r in enumerate(rws):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S, replace=m < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def run(rows, y_tr, y_va, rows_va, rows_te, X, dim, hp, seed, want_test):
    torch.manual_seed(seed)
    net = Net(dim, hp["h"], len(GENES), hp["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), hp["lr"], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, hp["lr"], epochs=hp["ep"],
                                              steps_per_epoch=max(1, len(rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(y_tr.sum(0) > 0, (len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    best, best_state, bad, best_pv = -1, None, 0, None
    rng = np.random.RandomState(seed)
    for ep in range(hp["ep"]):
        net.train(); perm = rng.permutation(len(rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(y_tr[rr].astype(np.float32)).to(dev)
            loss = lf(net(batches(rows[rr], X, dim, rng)), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pv = net(batches(rows_va, X, dim, np.random.RandomState(1))).float().cpu().numpy()
        use = [j for j in range(len(GENES)) if y_va[:, j].sum() >= MIN_POS_VAL]
        v = float(np.mean([roc_auc_score(y_va[:, j], pv[:, j]) for j in use]))
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            best_pv = pv
        else:
            bad += 1
            if bad >= 8:
                break
    if not want_test:
        return best, None
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        pt = net(batches(rows_te, X, dim, np.random.RandomState(2))).float().cpu().numpy()
    return best, pt, best_pv


# step 0.1 粗网格的 6 维单纯形点(4 个权重和为 1)
GRID = [w for w in itertools.product([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], repeat=4)
        if abs(sum(w) - 1.0) < 1e-9]
print("[grid] simplex points: %d" % len(GRID), flush=True)

store = {}   # (enc, split_seed) -> dict(val=..., test=...)
for enc in ENC:
    rowmap, npyf, dim = ENC[enc]
    X = _ram(npyf)
    rows_all = np.array([rowmap[c] for c in common])
    for ss in SPLIT_SEEDS:
        t0 = time.time()
        sp = make_split(ss)
        tr, va, te = sp == "train", sp == "val", sp == "test"
        rows, y_tr, y_va, y_te = rows_all[tr], Y[tr], Y[va], Y[te]
        r_va, r_te = rows_all[va], rows_all[te]
        bv, bh = -1, None
        for hp in HP_GRID:
            v = float(np.mean([run(rows, y_tr, y_va, r_va, r_te, X, dim, hp, s, False)[0] for s in PROBE_SEEDS]))
            if v > bv:
                bv, bh = v, hp
        finals = [run(rows, y_tr, y_va, r_va, r_te, X, dim, bh, s, True) for s in FINAL_SEEDS]
        pv = np.mean([f[2] for f in finals], 0)
        pt = np.mean([f[1] for f in finals], 0)
        store[(enc, ss)] = dict(val=pv, test=pt, hp=bh, val_auc=bv,
                                pos_test=[int(y_te[:, j].sum()) for j in range(len(GENES))])
        per = [t_auc(y_te[:, j], pt[:, j]) for j in range(len(GENES))]
        print("  %-10s split=%d val=%.4f TEST mean=%.4f  (%.0fs)"
              % (enc, ss % 100, bv, np.nanmean(per), time.time() - t0), flush=True)

EL = sorted(ENC)
print("\n" + "=" * 100)
print("=== 重复抽样结果: 5 套独立 60/20/20 划分 ===")
print("=" * 100, flush=True)
rows_out = []
for enc in EL + ["ENS_EQ", "ENS_VALW"]:
    means = []
    for ss in SPLIT_SEEDS:
        if enc.startswith("ENS_"):
            V = np.stack([store[(e, ss)]["val"] for e in EL], 0)     # (4, n_va, 6)
            T = np.stack([store[(e, ss)]["test"] for e in EL], 0)
            if enc == "ENS_EQ":
                w = np.ones(4) / 4
            else:
                bestw, bestv = None, -1
                sp = make_split(ss)
                y_va = Y[sp == "val"]
                use = [j for j in range(len(GENES)) if y_va[:, j].sum() >= MIN_POS_VAL]
                for wv in GRID:
                    w = np.array(wv)
                    P = (w[:, None, None] * V).sum(0)
                    v = float(np.mean([roc_auc_score(y_va[:, j], P[:, j]) for j in use]))
                    if v > bestv:
                        bestv, bestw = v, w
                w = bestw
            P = (w[:, None, None] * T).sum(0)
            y_te = Y[make_split(ss) == "test"]
            per = [t_auc(y_te[:, j], P[:, j]) for j in range(len(GENES))]
            wtxt = " ".join("%.1f" % x for x in w)
        else:
            y_te = Y[make_split(ss) == "test"]
            per = [t_auc(y_te[:, j], store[(enc, ss)]["test"][:, j]) for j in range(len(GENES))]
            wtxt = ""
        m = float(np.nanmean(per))
        means.append(m)
        for g, a in zip(GENES, per):
            rows_out.append(dict(encoder=enc, split_seed=ss, gene=g, test_auc=a, test_mean=m))
        print("  %-10s split=%d  %s  mean=%.4f   w[%s]"
              % (enc, ss % 100, "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a)
                                          for g, a in zip(GENES, per)), m, wtxt), flush=True)
    print("  --> %-10s  mean over 5 splits = %.4f   std = %.4f\n"
          % (enc, float(np.mean(means)), float(np.std(means, ddof=1))), flush=True)

df = pd.DataFrame(rows_out)
df.to_csv(os.path.join(C, "repeated_split_results.csv"), index=False)

print("=" * 100)
print("=== 汇总: test mean over 5 splits (mean +- std) ===", flush=True)
for enc in EL + ["ENS_EQ", "ENS_VALW"]:
    s = df[df.encoder == enc].groupby("split_seed").test_mean.first()
    print("  %-10s %.4f +- %.4f   (min %.4f  max %.4f)"
          % (enc, s.mean(), s.std(ddof=1), s.min(), s.max()), flush=True)
print("\n=== 逐基因: 5 套划分的 test AUC ===", flush=True)
piv = df.pivot_table(index="gene", columns="encoder", values="test_auc", aggfunc=["mean", "std"])
print(piv.round(3).to_string(), flush=True)
print("DONE", flush=True)