# -*- coding: utf-8 -*-
"""37_stain_shortcut.py — 模型是否在利用切片级染色/批次强度做捷径?

触发线索: 33 号扫描里 patch L2 归一化让 AUC 崩到 0.518, 逐切片 z-score 跌到 0.669。
两者都说明"幅值"与"切片级分布"携带信息。是真形态信号, 还是伪相关?

三问:
  Q1 切片级均值特征(1536维)单独做线性探针能到多少?
     -> 若接近 MIL 的 0.743, 说明注意力没加东西, 信号本质是切片级签名
  Q2 只用幅值统计量(每张切片 ~8 个标量)能预测多少?
     -> 若显著高于 0.5, 说明幅值本身携带基因信息
  Q3 模型预测分与切片级幅值的相关性有多强?
     -> 若强相关, 提示模型可能在跟着强度跑
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
SEP = "=" * 96

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); RT = df.mem_row.values.astype(int)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
D = X.shape[2]
L = np.load(os.path.join(C, "oof_giga_5seed.npy"))
PRED = 1.0 / (1.0 + np.exp(-L))
print(f"[LUAD n={N}] patches={X.shape[1]} dim={D}", flush=True)

# ---- 逐切片统计量 ----
slide_mean = np.zeros((N, D), np.float32)      # 切片级均值(方向)
slide_stat = np.zeros((N, 8), np.float32)      # 幅值类标量
slide_pca = np.zeros((N, 24), np.float32)      # 切片内主成分(结构)
for i, r in enumerate(RT):
    v = X[r].astype(np.float32)                 # (S, D)
    nrm = np.linalg.norm(v, axis=1)             # 每个 patch 的模长
    slide_mean[i] = v.mean(0)
    slide_stat[i] = [nrm.mean(), nrm.std(), np.percentile(nrm, 10), np.percentile(nrm, 50),
                     np.percentile(nrm, 90), np.abs(nrm).max(), v.std(0).mean(), v.mean()]
    pv = v - v.mean(0, keepdims=True)
    p = PCA(n_components=24, random_state=0).fit(pv[:200])
    slide_pca[i] = p.transform(pv)[:24].mean(0) if pv.shape[0] >= 24 else np.zeros(24)
print("[stat] 逐切片统计量提取完成", flush=True)

def probe(Xd, y, seed=0):
    if not (0 < int(y.sum()) < len(y)):
        return np.nan
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, y):
        sc = StandardScaler().fit(Xd[tr])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xd[tr]), y[tr])
        o[te] = m.predict_proba(sc.transform(Xd[te]))[:, 1]
    return roc_auc_score(y, o)

print("\n" + SEP, flush=True)
print("=== Q1/Q2: 各类切片级表示单独预测基因的能力 (5-fold OOF LR) ===", flush=True)
views = [("幅值标量 (8d)", slide_stat), ("切片均值方向 (1536d)", slide_mean), ("切片内 PCA-24", slide_pca)]
print(f"{'表示':<24}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}")
for name, M in views:
    a = [probe(M, Y[:, j]) for j in range(len(GENES))]
    print(f"{name:<24}" + "".join(f"{x:>9.3f}" for x in a) + f"{np.mean(a):>9.3f}", flush=True)
mil = [roc_auc_score(Y[:, j], PRED[:, j]) for j in range(len(GENES))]
print(f"{'MIL 5-seed 集成(对照)':<24}" + "".join(f"{x:>9.3f}" for x in mil) + f"{np.mean(mil):>9.3f}", flush=True)
ratio = np.mean([probe(slide_mean, Y[:, j]) for j in range(len(GENES))]) / np.mean(mil)
print(f"\n  切片均值 / MIL = {ratio:.1%}  <- 若 >80%, 注意力带来的增量很小", flush=True)

print("\n" + SEP, flush=True)
print("=== Q3: 模型预测分与切片级幅值的耦合程度 ===", flush=True)
nrm_mean = slide_stat[:, 0]
q = lambda v: np.percentile(v, [25, 50, 75, 100])   # 4 个边界 -> 3 个四分位
print(f"{'gene':<8}{'Spearman(预测分, 幅值均值)':>28}{'预测分四分位内幅值极差':>24}{'P(秩相关显著)':>14}")
# 手算 Spearman (秩相关), 避免依赖 scipy
def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0

def perm_p(a, b, n=5000, seed=0):
    r = rank_corr(a, b); rng = np.random.RandomState(seed); cnt = 0
    for _ in range(n):
        if abs(rank_corr(a, rng.permutation(b))) >= abs(r):
            cnt += 1
    return r, (cnt + 1) / (n + 1)

for j, g in enumerate(GENES):
    rho, pv = perm_p(PRED[:, j], nrm_mean)
    qs = q(PRED[:, j])
    bins = [nrm_mean[(PRED[:, j] >= qs[k]) & (PRED[:, j] <= qs[k + 1])].mean() for k in range(3)]
    print(f"{g:<8}{rho:>28.3f}{(max(bins)-min(bins))/abs(np.mean(bins)):>24.3f}{pv:>14.4f}", flush=True)

print("\n=== 结论判据 ===", flush=True)
mag_auc = np.mean([probe(slide_stat, Y[:, j]) for j in range(len(GENES))])
print(f"  幅值标量 mean AUC = {mag_auc:.3f}  (0.5 = 无信息)")
print(f"  切片均值 mean AUC = {np.mean([probe(slide_mean, Y[:, j]) for j in range(len(GENES))]):.3f}")
print(f"  MIL       mean AUC = {np.mean(mil):.3f}")
if mag_auc > 0.58:
    print("  -> 幅值本身携带基因信息: 染色/批次强度可能是真实代理, 也可能是捷径, 需要外部数据区分")
elif mag_auc <= 0.58:
    print("  -> 幅值几乎无基因信息; l2 崩塌说明模型需要的是『相对幅度结构』而非绝对强度")
pd.DataFrame([dict(sample=i, **{f"norm_{k}": slide_stat[i, k] for k in range(8)},
                   **{f"pred_{g}": PRED[i, j] for j, g in enumerate(GENES)}) for i in range(N)]
             ).to_csv(os.path.join(C, "slide_stats.csv"), index=False)
print(f"\n[saved] {C}\\slide_stats.csv")
print("DONE", flush=True)