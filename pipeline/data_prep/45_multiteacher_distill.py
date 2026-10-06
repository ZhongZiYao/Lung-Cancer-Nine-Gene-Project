# -*- coding: utf-8 -*-
"""45_multiteacher_distill.py — 双教师蒸馏 (arXiv 2607.14703 路线)

动机(全部来自本项目实测):
  1. 526K 参数 vs 444 例, train AUC 0.95 -> 需要更好的初始化来压过拟合
  2. 特征拼接 TITAN+GPslide = +0.0004 (无效)
  3. 但 TITAN 与 GP-slide 逐基因互补: TITAN 赢 TP53/NF1, GP-slide 赢 STK11/EGFR
     score-level 集成拿到 +0.022
  => 正确用法不是"再加一路预测", 而是让教师约束学生 trunk 的表征

方法(对齐 arXiv 2607.14703 的 angular dispersion normalized distillation):
  bag  = gated_MIL(patches)                     # (B, 256)
  s_t  = Proj_T(TITAN_768)                      # (B, 256)
  s_g  = Proj_G(GPslide_768)                    # (B, 256)
  loss = BCE(task) + lam*ADND(s_t, bag) + mu*ADND(s_g, bag)
  ADND(a,b) = 1 - cos(normalize(a), normalize(b))

配置(5 seed 配对, 同一折划分):
  base    lam=0, mu=0            参照
  d_titan lam=0.3, mu=0          只用 TITAN
  d_gp    lam=0, mu=0.3          只用 GigaPath-slide
  d_both  lam=0.3, mu=0.3        双教师
  d_both1 lam=1.0, mu=1.0        双教师(强)

严谨性: 每个特征源都走 mem_row 映射, 且起飞前用 mean-pool+LR 断言绝对水平。
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
DIM, EP, BATCH, LR_LR, HID, S_INST = 1536, 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]
tn = lambda b: "-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None

# ---------------- 数据接入(每个源都走 mem_row 映射) ----------------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner")
df = df.dropna(subset=["TP53"]).reset_index(drop=True)
df = df[df.project == "LUAD"].reset_index(drop=True)
PATCH = np.ascontiguousarray(np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))[df.mem_row.values.astype(int)])
Y = df[GENES].values.astype(int)
N, K = len(df), len(GENES)
print(f"[data] LUAD n={N}  patch {PATCH.shape}", flush=True)

# TITAN
tk = pickle.load(open(os.path.join(R, "models", "TITAN", "TCGA_TITAN_features.pkl"), "rb"))
T = {}
for f, e in zip(tk["filenames"], tk["embeddings"]):
    c = tn(os.path.basename(f))
    if c and c not in T:
        T[c] = np.asarray(e, np.float32)
# GigaPath slide
E14 = np.load(os.path.join(C, "NSCLC_gigap_slide_L14.npy"))
ids14 = pd.read_csv(os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")).case_id.tolist()
m14 = {c: i for i, c in enumerate(ids14)}
# 三个源取交集: TITAN pkl(9547 case) 与 parquet(453 肺癌 case) 覆盖范围并不一致,
# 少数 LUAD 只有其中一个教师, 直接 assert 会中止 —— 应取交集而非退出。
keep = np.array([i for i, c in enumerate(df.case_id) if c in T and c in m14], dtype=int)
print(f"[data] 缺教师特征的 case 已丢弃: {len(df) - len(keep)}", flush=True)
df = df.iloc[keep].reset_index(drop=True)
PATCH, Y = PATCH[keep], Y[keep]
TT = np.stack([T[c] for c in df.case_id])
GS = E14[[m14[c] for c in df.case_id]].mean(1)          # (n, 768) 14 层均值
N = len(df)
print(f"[data] 对齐后 n={N}  TITAN {TT.shape}  GPslide {GS.shape}", flush=True)

# ---------------- 起飞前自检 ----------------
def mp_probe(M, y, j):
    # patch 特征是 (n, S, D) -> 沿 patch 轴池化; slide 特征是 (n, D) -> 不能池化
    X = M.astype(np.float32).mean(1) if M.ndim == 3 else M.astype(np.float32)
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit((X[tr]-mu)/sd, y[tr])
        o[te] = m.predict_proba((X[te]-mu)/sd)[:, 1]
    return roc_auc_score(y, o)

for nm, M, j in [("patch", PATCH, 1), ("TITAN", TT, 1), ("GPslide", GS, 1)]:
    a = mp_probe(M, Y[:, j], j)
    print(f"[preflight] LUAD mean-pool+LR  {nm:<8} TP53 = {a:.3f}", flush=True)
    assert a > 0.60, f"{nm} 特征映射错误(AUC={a:.3f}), 中止"
print("[preflight] 全部通过", flush=True)

# ---------------- 模型 ----------------
class Net(nn.Module):
    def __init__(s, d, h, k, dT=768, dG=768, use_t=True, use_g=True):
        super().__init__()
        s.use_t, s.use_g = use_t, use_g
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
        if use_t: s.pt = nn.Linear(dT, h, bias=False)
        if use_g: s.pg = nn.Linear(dG, h, bias=False)

    def forward(s, x, tt=None, gs=None):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        bag = (z * a.unsqueeze(-1)).sum(1)
        logits = s.od(bag)
        aux = []
        if s.use_t and tt is not None: aux.append(F.normalize(s.pt(tt), dim=-1))
        if s.use_g and gs is not None: aux.append(F.normalize(s.pg(gs), dim=-1))
        return logits, bag, aux


def adnd(student_vec_list, bag):
    b = F.normalize(bag, dim=-1)
    return sum((1.0 - (v * b).sum(-1)).mean() for v in student_vec_list) / max(len(student_vec_list), 1)


def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, PATCH.shape[2]), np.float16)
    for i, rw in enumerate(rows):
        v = PATCH[rw]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


CFG = [("base", 0.0, 0.0), ("d_titan", 0.3, 0.0), ("d_gp", 0.0, 0.3),
       ("d_both", 0.3, 0.3), ("d_both1", 1.0, 1.0)]


def run(lam, mu, seed):
    oof = np.zeros((N, K), np.float32)
    strata = np.clip(Y.sum(1), 0, 3).astype(int)
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(N), strata)):
        torch.manual_seed(seed * 1000 + f)
        Ytr = Y[tr]
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(PATCH.shape[2], HID, K, use_t=(lam > 0), use_g=(mu > 0)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = torch.from_numpy(gather(tr[rr], rng)).to(dev, torch.float32)
                tt = torch.from_numpy(TT[tr[rr]]).to(dev, torch.float32) if lam > 0 else None
                gs = torch.from_numpy(GS[tr[rr]]).to(dev, torch.float32) if mu > 0 else None
                yb = torch.from_numpy(Ytr[rr].astype(np.float32)).to(dev)
                p, bag, aux = net(bt, tt, gs)
                loss = lf(p, yb).mul(pw).mean()
                if aux:
                    # aux 的顺序随启用的教师而变, 不能用下标 0/1 反推 lam/mu —— 
                    # 否则单教师配置(只开 GP)会拿到 lam=0, 蒸馏项恒为 0, 静默退化成 base。
                    ws = ([lam] if net.use_t else []) + ([mu] if net.use_g else [])
                    loss = loss + sum(w * adnd([v], bag) for w, v in zip(ws, aux))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            bt = torch.from_numpy(gather(te, np.random.RandomState(7))).to(dev, torch.float32)
            p, _, _ = net(bt)
            oof[te] = p.float().cpu().numpy()
    return oof


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
res = {n: np.zeros((len(SEEDS), K), np.float32) for n, _, _ in CFG}
store = {n: [] for n, _, _ in CFG}
for seed in SEEDS:
    line = f"[seed {seed}] "
    for name, lam, mu in CFG:
        op = run(lam, mu, seed)
        res[name][seed] = [auc(Y[:, j], op[:, j]) for j in range(K)]
        store[name].append(op)
        line += f"{name}={np.nanmean(res[name][seed]):.4f}  "
    print(line, flush=True)

print("\n" + "=" * 104)
print("=== 双教师蒸馏 5-seed 配对 (LUAD, Prov-GigaPath 512) ===")
print("=" * 104)
base_m = res["base"].mean(0)
base_s = res["base"].mean(1)
summary = []
for name, lam, mu in CFG:
    m = res[name].mean(0)
    d = res[name].mean(1) - base_s
    ens = np.mean(store[name], axis=0)
    a_ens = [auc(Y[:, j], ens[:, j]) for j in range(K)]
    print(f"{name:<10}(lam={lam},mu={mu})" + "".join(f"{v:>9.3f}" for v in m)
          + f"{m.mean():>10.4f}{d.mean():>+10.4f}{int((d>0).sum()):>5}/5{np.mean(a_ens):>11.4f}", flush=True)
    summary.append(dict(config=name, lam=lam, mu=mu, mean_auc=float(m.mean()),
                        delta=float(d.mean()), agree=int((d > 0).sum()),
                        ens_auc=float(np.mean(a_ens))))

print("\n=== 判定 ===")
for s in summary:
    if s["config"] == "base":
        continue
    d, ag = s["delta"], s["agree"]
    v = ("有效提升 ✅" if d > 0.01 and ag >= 4 else "明确有害 ❌" if d < -0.01 else "噪声")
    print(f"  {s['config']:<10} Δ={d:>+8.4f}  同号 {ag}/5  集成 {s['ens_auc']:.4f}  -> {v}")

pd.DataFrame(summary).to_csv(os.path.join(C, "multiteacher_distill.csv"), index=False)
for name, _, _ in CFG:
    np.save(os.path.join(C, f"oof_{name}_5seed.npy"), np.mean(store[name], axis=0))
print(f"\n[saved] {C}\\multiteacher_distill.csv")
print("DONE", flush=True)