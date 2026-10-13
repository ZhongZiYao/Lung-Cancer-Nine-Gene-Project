# -*- coding: utf-8 -*-
"""46_routeA_twostage.py — 路线 A: 两阶段(预训练 925 -> 微调 LUAD 436)

严格按用户给的规格实现:
  Stage 1  预训练: loss = BCE(任务) + λ*ADND(TITAN_proj, bag) + μ*ADND(GPslide_proj, bag)
                 用全部 925 例(LUAD+LUSC), 不做留出
  Stage 2  在 LUAD 上正常微调, λ=μ=0, 从 Stage 1 的权重初始化
对照: stage2 从随机初始化(即 45 号的 base)

与 45 号的区别(这是关键):
  45  单阶段, 只 434 例 LUAD, 蒸馏当正则
  46  两阶段, Stage1 见 925 例(2 倍数据), Stage2 无蒸馏项

配置:
  base      Stage2 从随机初始化(对照)
  A_full    Stage1 = BCE + 0.3*(ADND_T + ADND_G)
  A_pure    Stage1 = 只蒸馏, 不用任何标签
  A_titan   Stage1 = BCE + 0.3*ADND_T  (单教师)
"""
import os, sys, re, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
EP, BATCH, LR_LR, HID, S_INST = 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
tn = lambda b: "-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None

# ---------------- 对齐全部数据源(所有 mem_row 显式映射 + 起飞前断言) ----------------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
base = idx[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner")
base = base.dropna(subset=["TP53"]).reset_index(drop=True)

tk = pickle.load(open(os.path.join(R, "models", "TITAN", "TCGA_TITAN_features.pkl"), "rb"))
T = {}
for f, e in zip(tk["filenames"], tk["embeddings"]):
    c = tn(os.path.basename(f))
    if c and c not in T:
        T[c] = np.asarray(e, np.float32)
E14 = np.load(os.path.join(C, "NSCLC_gigap_slide_L14.npy"))
ids14 = pd.read_csv(os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")).case_id.tolist()
m14 = {c: i for i, c in enumerate(ids14)}

base = base[base.case_id.isin(T) & base.case_id.isin(m14)].reset_index(drop=True)
PATCH = np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))[base.mem_row.values.astype(int)])
TT = np.stack([T[c] for c in base.case_id])
GS = E14[[m14[c] for c in base.case_id]].mean(1)
Y = base[GENES].values.astype(int)
isLU = (base.project == "LUAD").values
N, K = len(base), len(GENES)
print(f"[data] pool n={N}  LUAD {int(isLU.sum())} / LUSC {int((~isLU).sum())}", flush=True)
print(f"[data] patch {PATCH.shape}  TITAN {TT.shape}  GPslide {GS.shape}", flush=True)

# 起飞前断言: 三个源各自的 LUAD mean-pool+LR 必须 > 0.60
def mp_probe(M, y):
    X = M.astype(np.float32).mean(1) if M.ndim == 3 else M.astype(np.float32)
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        o[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, o)

for nm, M in [("patch", PATCH), ("TITAN", TT), ("GPslide", GS)]:
    a = mp_probe(M[isLU], Y[isLU, 1])
    print(f"[preflight] LUAD mean-pool+LR  {nm:<8} TP53 = {a:.3f}", flush=True)
    assert a > 0.60, f"{nm} 特征映射错误({a:.3f}), 中止"
print("[preflight] 全部通过", flush=True)

# ---------------- 模型 ----------------
class Net(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
        s.pt = nn.Linear(768, h, bias=False)
        s.pg = nn.Linear(768, h, bias=False)

    def forward(s, x, tt=None, gs=None):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        bag = (z * a.unsqueeze(-1)).sum(1)
        aux = []
        if tt is not None: aux.append(F.normalize(s.pt(tt), dim=-1))
        if gs is not None: aux.append(F.normalize(s.pg(gs), dim=-1))
        return s.od(bag), bag, aux


def adnd(vs, bag):
    b = F.normalize(bag, dim=-1)
    return sum((1.0 - (v * b).sum(-1)).mean() for v in vs) / max(len(vs), 1)


def batch_of(rows, rng, src):
    M = {"patch": PATCH, "TITAN": TT, "GPslide": GS}[src]
    return M[rows]


def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, PATCH.shape[2]), np.float16)
    for i, rw in enumerate(rows):
        v = PATCH[rw]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


def train_stage(rows, Yrows, epochs, lam, mu, seed_tag, use_labels=True, lr=LR_LR, init=None):
    """在 rows 上训练。init 为 state_dict 时从它初始化。返回 net。"""
    torch.manual_seed(seed_tag)
    n = len(rows)
    net = Net(PATCH.shape[2], HID, K).to(dev)
    if init is not None:
        net.load_state_dict(init, strict=False)
    pw = np.ones(K, np.float32)
    if use_labels and Yrows is not None and Yrows.size:
        pos = Yrows.sum(0)
        pw = np.where(pos > 0, (n - pos) / np.maximum(pos, 1), 1.0)
    pw = torch.tensor(pw, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=max(1, n // BATCH))
    rng = np.random.RandomState(seed_tag)
    lf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        net.train(); perm = rng.permutation(n)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            bt = torch.from_numpy(gather(rows[rr], rng)).to(dev, torch.float32)
            tt = torch.from_numpy(TT[rows[rr]]).to(dev, torch.float32) if lam > 0 else None
            gs = torch.from_numpy(GS[rows[rr]]).to(dev, torch.float32) if mu > 0 else None
            p, bag, aux = net(bt, tt, gs)
            loss = torch.zeros((), device=dev)
            if use_labels and Yrows is not None:
                yb = torch.from_numpy(Yrows[rr].astype(np.float32)).to(dev)
                loss = loss + lf(p, yb).mul(pw).mean()
            if aux:
                ws = ([lam] if lam > 0 else []) + ([mu] if mu > 0 else [])
                loss = loss + sum(w * adnd([v], bag) for w, v in zip(ws, aux))
            if loss.requires_grad:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def stage2_oof(init, seed, rows, Yrows):
    """在 LUAD 上 5-fold 微调, λ=μ=0。返回 OOF logits。"""
    oof = np.zeros((len(rows), K), np.float32)
    strata = np.clip(Yrows.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(rows)), strata)):
        net = train_stage(rows[tr], Yrows[tr], EP, 0.0, 0.0, seed * 1000 + f, True, init=init)
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            bt = torch.from_numpy(gather(rows[te], np.random.RandomState(7))).to(dev, torch.float32)
            p, _, _ = net(bt)
            oof[te] = p.float().cpu().numpy()
    return oof


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
lu_rows = np.flatnonzero(isLU)
all_rows = np.arange(N)
Ylu = Y[lu_rows]
print(f"\nStage1 池: {N} 例 (LUAD {len(lu_rows)} + LUSC {N-len(lu_rows)})", flush=True)
print(f"Stage2 集: {len(lu_rows)} 例 LUAD", flush=True)

CFG = [("base", 0.0, 0.0, True), ("A_full", 0.3, 0.3, True),
       ("A_pure", 0.3, 0.3, False), ("A_titan", 0.3, 0.0, True)]

res = {n: np.zeros((len(SEEDS), K), np.float32) for n, _, _, _ in CFG}
store = {n: [] for n, _, _, _ in CFG}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for name, lam, mu, use_lab in CFG:
        if name == "base":
            init = None
        else:
            # Stage 1: 在全部 925 例上预训练(无留出), 只跑一次
            s1 = train_stage(all_rows, Y, EP, lam, mu, seed * 777, use_labels=use_lab)
            init = {k_: v.detach().clone() for k_, v in s1.state_dict().items()}
        oof = stage2_oof(init, seed, lu_rows, Ylu)
        res[name][seed] = [auc(Ylu[:, j], oof[:, j]) for j in range(K)]
        store[name].append(oof)
        line += f"{name}={np.nanmean(res[name][seed]):.4f}  "
    print(line, flush=True)

print("\n" + "=" * 104)
print("=== 路线 A 两阶段 (LUAD n=%d, 5 seed 配对) ===" % len(lu_rows))
print("=" * 104)
base_s = res["base"].mean(1)
summary = []
print(f"{'config':<12}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>10}{'Δ':>10}{'同号':>7}{'集成':>10}")
for name, _, _, _ in CFG:
    m = res[name].mean(0); d = res[name].mean(1) - base_s
    ens = np.mean(store[name], axis=0)
    a_ens = np.mean([auc(Ylu[:, j], ens[:, j]) for j in range(K)])
    print(f"{name:<12}" + "".join(f"{v:>9.3f}" for v in m) + f"{m.mean():>10.4f}{d.mean():>+10.4f}"
          + f"{int((d>0).sum()):>6}/5{a_ens:>10.4f}", flush=True)
    summary.append(dict(config=name, mean_auc=float(m.mean()), delta=float(d.mean()),
                        agree=int((d > 0).sum()), ens_auc=float(a_ens)))

print("\n=== 判定 ===")
for s in summary:
    if s["config"] == "base":
        continue
    d, ag = s["delta"], s["agree"]
    v = ("有效提升 OK" if d > 0.01 and ag >= 4 else "明确有害 BAD" if d < -0.01 else "噪声")
    print(f"  {s['config']:<10} Δ={d:>+8.4f}  同号 {ag}/5  集成 {s['ens_auc']:.4f}  -> {v}")
pd.DataFrame(summary).to_csv(os.path.join(C, "routeA_twostage.csv"), index=False)
print(f"\n[saved] {C}\\routeA_twostage.csv")
print("DONE", flush=True)