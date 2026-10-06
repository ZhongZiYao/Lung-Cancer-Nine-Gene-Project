# -*- coding: utf-8 -*-
"""47_routeB_condattn.py — 路线 B: slide-conditioned attention (结构创新, 零额外算力)

用户原式:  a = softmax( w(tanh(V(z))*sigmoid(U(z))) + P(slide_768) )
数学澄清: 若 P 投影为【标量】再加到每个实例的注意力 logit 上, 沿 S 维做 softmax 是
         平移不变的 => 整个式子恒等于原式, 无任何效果。因此有效版本必须让 slide
         信息进入 hidden state 或进入 bag:
  B_cond_t    z' = z + Proj_T(TITAN)          -> 注意力分布被调制
  B_cond_g    z' = z + Proj_G(GPslide)        -> 同上, 另一教师
  B_cond_both z' = z + Proj_T + Proj_G         -> 双教师
  B_bag       bag = attn_sum(z) ⊕ Proj_T ⊕ Proj_G -> 晚融合进分类头
  base        原始 gated

与 concat 的本质区别: concat 把 slide 塞进分类器输入(只影响最后的线性变换),
B_cond 把 slide 塞进 hidden state(影响【看哪些 patch】), 这才是"条件化选择"。
"""
import os, sys, re, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
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

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner")
df = df.dropna(subset=["TP53"]).reset_index(drop=True)
df = df[df.project == "LUAD"].reset_index(drop=True)

tk = pickle.load(open(os.path.join(R, "models", "TITAN", "TCGA_TITAN_features.pkl"), "rb"))
T = {}
for f, e in zip(tk["filenames"], tk["embeddings"]):
    c = tn(os.path.basename(f))
    if c and c not in T:
        T[c] = np.asarray(e, np.float32)
E14 = np.load(os.path.join(C, "NSCLC_gigap_slide_L14.npy"))
ids14 = pd.read_csv(os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")).case_id.tolist()
m14 = {c: i for i, c in enumerate(ids14)}
df = df[df.case_id.isin(T) & df.case_id.isin(m14)].reset_index(drop=True)

PATCH = np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))[df.mem_row.values.astype(int)])
TT = np.stack([T[c] for c in df.case_id])
GS = E14[[m14[c] for c in df.case_id]].mean(1)
Y = df[GENES].values.astype(int)
N, K = len(df), len(GENES)
print(f"[data] LUAD n={N}  patch {PATCH.shape}  TITAN {TT.shape}  GPslide {GS.shape}", flush=True)


def mp_probe(M, y):
    X = M.astype(np.float32).mean(1) if M.ndim == 3 else M.astype(np.float32)
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        o[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, o)


for nm, M in [("patch", PATCH), ("TITAN", TT), ("GPslide", GS)]:
    a = mp_probe(M, Y[:, 1])
    print(f"[preflight] LUAD mean-pool+LR  {nm:<8} TP53 = {a:.3f}", flush=True)
    assert a > 0.60, f"{nm} 特征映射错误({a:.3f}), 中止"
print("[preflight] 全部通过", flush=True)


class Net(nn.Module):
    """mode 决定 slide 特征如何进入网络"""
    def __init__(s, d, h, k, mode):
        super().__init__()
        s.mode = mode
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False)
        if "t" in mode or mode == "bag":
            s.pt = nn.Linear(768, h)
        if "g" in mode or mode == "bag":
            s.pg = nn.Linear(768, h)
        out_h = h * (1 + 2) if mode == "bag" else h
        s.od = nn.Linear(out_h, k)

    def forward(s, x, tt=None, gs=None):
        z = s.trunk(x)                                   # (B,S,H)
        svec = []
        if s.mode != "base":
            if hasattr(s, "pt") and tt is not None: svec.append(s.pt(tt))
            if hasattr(s, "pg") and gs is not None: svec.append(s.pg(gs))
        if s.mode == "bag":
            a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
            bag = torch.cat([(z * a.unsqueeze(-1)).sum(1)] + svec, 1)   # 晚融合
        else:
            # 关键: slide 向量广播加到每个 patch 的 hidden 上, 改变注意力分布
            zc = z + torch.stack(svec, 0).sum(0).unsqueeze(1) if svec else z
            a = torch.softmax(s.w(torch.tanh(s.V(zc)) * torch.sigmoid(s.U(zc))).squeeze(-1), 1)
            bag = (zc * a.unsqueeze(-1)).sum(1)
        return s.od(bag)


def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, PATCH.shape[2]), np.float16)
    for i, rw in enumerate(rows):
        v = PATCH[rw]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


def run(mode, seed):
    oof = np.zeros((N, K), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(PATCH.shape[2], HID, K, mode).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(tr)
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                # perm = rng.permutation(tr) 返回的已经是 case 下标, 不能再套一层 tr[]
                bt = torch.from_numpy(gather(rr, rng)).to(dev, torch.float32)
                tt = torch.from_numpy(TT[rr]).to(dev, torch.float32)
                gs = torch.from_numpy(GS[rr]).to(dev, torch.float32)
                yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt, tt, gs), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            bt = torch.from_numpy(gather(te, np.random.RandomState(7))).to(dev, torch.float32)
            oof[te] = net(bt, torch.from_numpy(TT[te]).to(dev, torch.float32),
                          torch.from_numpy(GS[te]).to(dev, torch.float32)).float().cpu().numpy()
    return oof


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
MODES = ["base", "cond_t", "cond_g", "cond_tg", "bag"]
res = {m: np.zeros((len(SEEDS), K), np.float32) for m in MODES}
store = {m: [] for m in MODES}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for m in MODES:
        op = run(m, seed)
        res[m][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[m].append(op)
        line += f"{m}={np.nanmean(res[m][seed]):.4f}  "
    print(line, flush=True)

print("\n" + "=" * 104)
print(f"=== 路线 B: slide-conditioned attention (LUAD n={N}, 5 seed 配对) ===")
print("=" * 104)
base_s = res["base"].mean(1)
summary = []
print(f"{'mode':<10}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>10}{'Δ':>10}{'同号':>7}{'集成':>10}")
for m in MODES:
    mm = res[m].mean(0); d = res[m].mean(1) - base_s
    ens = np.mean(store[m], axis=0)
    a_ens = np.mean([auc(Y[:, j], ens[:, j]) for j in range(K)])
    print(f"{m:<10}" + "".join(f"{v:>9.3f}" for v in mm) + f"{mm.mean():>10.4f}{d.mean():>+10.4f}"
          + f"{int((d>0).sum()):>6}/5{a_ens:>10.4f}", flush=True)
    summary.append(dict(mode=m, mean_auc=float(mm.mean()), delta=float(d.mean()),
                        agree=int((d > 0).sum()), ens_auc=float(a_ens)))

print("\n=== 判定 ===")
for s in summary:
    if s["mode"] == "base":
        continue
    d, ag = s["delta"], s["agree"]
    v = ("有效提升 OK" if d > 0.01 and ag >= 4 else "明确有害 BAD" if d < -0.01 else "噪声")
    print(f"  {s['mode']:<10} Δ={d:>+8.4f}  同号 {ag}/5  集成 {s['ens_auc']:.4f}  -> {v}")
pd.DataFrame(summary).to_csv(os.path.join(C, "routeB_condattn.csv"), index=False)
for m in MODES:
    np.save(os.path.join(C, f"oof_routeB_{m}_5seed.npy"), np.mean(store[m], axis=0))
print(f"\n[saved] {C}\\routeB_condattn.csv")
print("DONE", flush=True)