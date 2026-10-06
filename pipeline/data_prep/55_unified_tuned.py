# -*- coding: utf-8 -*-
"""55_unified_tuned.py v2 — 统一多编码器数据集 + 真 train/val/test 划分 + val 调参

v2 相对 v1 的三处修正:
  1. [性能] 每个编码器的特征只 np.load 一次常驻 RAM (v1 用 mmap, 每 step 重读
     gigap 1407MB, 估算要读几百 GB)。加载后清空上一编码器, 常驻只保留一个。
  2. [协议] test 集只在【选中超参】上跑 —— v1 对每组超参都算了 test 预测,
     虽然只报告选中的那组, 但 test 已经被看过 20 次, 不算"只跑一次"。
     v2 改成: 选超参阶段只算 val, 选定后才算 test。
  3. [选择] val 指标门槛从 10 个阳性放宽到 8 个, 否则本划分里 EGFR(val 阳性 9)
     会被当成 0.5 罚分, 等于 EGFR 完全不参与选超参。

仍然刻意不做: 特征级多编码器拼接 —— 不同编码器的 tiling/patch 顺序不一致,
同一位置不是同一块组织。只能 score 级融合。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
BATCH, S_INST = 32, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 20261006
SPLIT_FILE = os.path.join(C, "split_luad.csv")
MIN_POS_VAL, MIN_POS_TEST = 8, 10

# ---------------- 1. 编码器清单 ----------------
ENC = {}
for enc, dim in [("gigap", 1536), ("uni", 1024), ("chief", 768), ("uni2h_luad", 1536)]:
    idxf = os.path.join(C, "NSCLC_%s_sub%d_index.csv" % (enc, 512))
    npyf = os.path.join(C, "NSCLC_%s_sub%d.npy" % (enc, 512))
    if os.path.exists(idxf) and os.path.exists(npyf):
        d = pd.read_csv(idxf)
        ENC[enc] = (dict(zip(d.case_id, d.mem_row)), npyf, dim)
        print("[enc] %-12s dim=%-5d case=%d" % (enc, dim, len(d)), flush=True)

# ---------------- 2. 标签 ----------------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
common = sorted(set(lab.index))
for enc in ENC:
    common = sorted(set(common) & set(ENC[enc][0].keys()))
print("\n[align] 四编码器共有的 LUAD case: %d" % len(common), flush=True)
Y = lab.loc[common, GENES].values.astype(int)
print("   阳性: " + "  ".join("%s=%d" % (g, int(Y[:, i].sum())) for i, g in enumerate(GENES)), flush=True)
assert len(common) >= 150, "可用 case 太少"

# ---------------- 3. 固定划分 60/20/20 (TP53 + NF1 两级贪心分层) ----------------
if os.path.exists(SPLIT_FILE):
    sp = pd.read_csv(SPLIT_FILE)
    assert list(sp.case_id) == common, "划分文件与当前 case 集不一致, 请删除 split_luad.csv 后重跑"
    print("\n[split] 复用 %s" % os.path.basename(SPLIT_FILE), flush=True)
else:
    rng = np.random.RandomState(SPLIT_SEED)
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
    pd.DataFrame(dict(case_id=common, split=assign,
                      **{g: Y[:, i] for i, g in enumerate(GENES)})).to_csv(SPLIT_FILE, index=False)
    print("\n[split] 新建 %s" % os.path.basename(SPLIT_FILE), flush=True)

split = sp.set_index("case_id").loc[common, "split"].values
print("   train %d / val %d / test %d" % (int((split == "train").sum()),
      int((split == "val").sum()), int((split == "test").sum())), flush=True)
for g, i in zip(GENES, range(len(GENES))):
    s = "  ".join("%s+%d/%d" % (t, int(Y[split == t, i].sum()), int((split == t).sum()))
                  for t in ["train", "val", "test"])
    print("      %-6s %s" % (g, s), flush=True)

# ---------------- 4. 模型 ----------------
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
        _RAM.clear()
        _RAM[p] = np.load(p)
        print("   [ram] %s %.0f MB" % (os.path.basename(p), _RAM[p].nbytes / 1e6), flush=True)
    return _RAM[p]


def batches(rows, X, dim, rng, S=S_INST):
    bt = np.empty((len(rows), S, dim), np.float16)
    for i, r in enumerate(rows):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S, replace=m < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def train_eval(enc, hp, seed, want_test=False):
    m, npyf, dim = ENC[enc]
    X = _ram(npyf)
    rows = np.array([m[c] for c in common])
    tr, va, te = split == "train", split == "val", split == "test"
    rows_tr, rows_va, rows_te = rows[tr], rows[va], rows[te]
    y_tr, y_va, y_te = Y[tr], Y[va], Y[te]
    torch.manual_seed(seed)
    net = Net(dim, hp["h"], len(GENES), hp["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), hp["lr"], weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, hp["lr"], epochs=hp["ep"],
                                              steps_per_epoch=max(1, len(rows_tr) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(y_tr.sum(0) > 0, (len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    best, best_state, bad, best_pv = -1, None, 0, None
    rng = np.random.RandomState(seed)
    for ep in range(hp["ep"]):
        net.train(); perm = rng.permutation(len(rows_tr))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            yb = torch.from_numpy(y_tr[rr].astype(np.float32)).to(dev)
            loss = lf(net(batches(rows_tr[rr], X, dim, rng)), yb).mul(pw).mean()
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
            best, best_state, bad = v, {k: t.detach().clone() for k, t in net.state_dict().items()}, 0
            best_pv = pv
        else:
            bad += 1
            if bad >= 8:
                break
    if not want_test:
        return best, best_pv
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        pt = net(batches(rows_te, X, dim, np.random.RandomState(2))).float().cpu().numpy()
    return best, pt


HP_GRID = [
    dict(h=256, lr=3e-4, drop=0.25, ep=40),
    dict(h=256, lr=3e-4, drop=0.50, ep=40),
    dict(h=128, lr=1e-4, drop=0.25, ep=60),
    dict(h=512, lr=3e-4, drop=0.25, ep=40),
]
t_auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan

print("\n" + "=" * 104)
print("=== 阶段1 选超参: 只看 val, 不碰 test ===")
print("=" * 104, flush=True)
best_pt, best_v, best_hp, rows_out = {}, {}, {}, []
for enc in ENC:
    bv, bh = -1, None
    for hp in HP_GRID:
        v = float(np.mean([train_eval(enc, hp, s)[0] for s in SEEDS]))
        print("   probe %-10s val=%.4f  h=%-4d lr=%.0e drop=%.2f ep=%d"
              % (enc, v, hp["h"], hp["lr"], hp["drop"], hp["ep"]), flush=True)
        if v > bv:
            bv, bh = v, hp
    best_v[enc], best_hp[enc] = bv, bh

print("\n" + "=" * 104)
print("=== 阶段2 独立测试集: 只对选中的一组超参跑一次 test, 5 seed 集成 ===")
print("=" * 104, flush=True)
for enc in ENC:
    bh = best_hp[enc]
    runs = [train_eval(enc, bh, s, want_test=True) for s in SEEDS]
    pt = np.mean([x[1] for x in runs], 0)
    best_pt[enc] = pt
    y_te = Y[split == "test"]
    per = [t_auc(y_te[:, j], pt[:, j]) for j in range(len(GENES))]
    print("\n  [%s] val=%.4f  h=%d lr=%.0e drop=%.2f ep=%d"
          % (enc, best_v[enc], bh["h"], bh["lr"], bh["drop"], bh["ep"]), flush=True)
    print("     TEST " + "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a)
                                  for g, a in zip(GENES, per))
          + "   mean=%.4f" % np.nanmean(per), flush=True)
    for g, a in zip(GENES, per):
        rows_out.append(dict(encoder=enc, gene=g, test_auc=a))

print("\n" + "=" * 104)
print("=== 阶段3 四编码器 score-level 集成 (test 概率平均) ===")
print("=" * 104, flush=True)
E = np.mean([best_pt[e] for e in ENC], 0)
y_te = Y[split == "test"]
per = [t_auc(y_te[:, j], E[:, j]) for j in range(len(GENES))]
print("  ENSEMBLE4  " + "  ".join("%s=%s" % (g, "n/a" if a != a else "%.3f" % a)
                                for g, a in zip(GENES, per))
      + "   mean=%.4f" % np.nanmean(per), flush=True)
for g, a in zip(GENES, per):
    rows_out.append(dict(encoder="ENSEMBLE4", gene=g, test_auc=a))

# 单基因基线: test 集阳性率的随机猜测参照
print("\n=== 参照: 单基因 test AUC 汇总 (按编码器) ===")
df = pd.DataFrame(rows_out)
piv = df.pivot(index="gene", columns="encoder", values="test_auc")
piv["n_pos_test"] = [int(y_te[:, j].sum()) for j in range(len(GENES))]
print(piv.round(3).to_string(), flush=True)
df.to_csv(os.path.join(C, "unified_test_results.csv"), index=False)
np.save(os.path.join(C, "unified_test_pred.npy"), E)
print("\n[saved] unified_test_results.csv / unified_test_pred.npy", flush=True)
print("DONE", flush=True)