# -*- coding: utf-8 -*-
"""43_luad_vs_luad_lusc.py — LUAD-only vs LUAD+LUSC 联合训练

问题: 481 例 LUSC 的 patch 特征一直就在本地缓存里, 从未用于训练。
     把它们加进联合训练能不能帮到 LUAD?

两个 cohort:
  A) LUAD only        (GigaPath 覆盖 436 例)
  B) LUAD + LUSC      (GigaPath 覆盖 907 例, 数据量翻倍)

评估口径(保证可比):
  - 内部: 5-fold OOF, 但【只在 LUAD 子集上算 AUC】, 两个 cohort 完全同口径
  - 外部: 都在 CPTAC-LUAD 216 例上评估

已知风险(跑之前先说清):
  - TP53 是唯一跨癌种都高发的基因(LUAD 0.535 / LUSC 0.795) -> 预期受益最大
  - KRAS(LUAD .312 / LUSC .015) / STK11(.134/.019) / EGFR(.138/.023)
    在 LUSC 几乎没有信号, 加进去可能让模型学到"总是预测负"从而伤害 LUAD
  所以预期是【逐基因分化】, 不是整体涨或整体跌, 所以必须逐基因报
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, EP, BATCH, LR_LR, HID, S_INST = 1536, 25, 32, 3e-4, 256, 512
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]


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


man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
ig = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = ig[["case_id", "mem_row"]].merge(iu[["case_id"]], on="case_id", how="inner")
df = df.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project.isin(["LUAD", "LUSC"])].reset_index(drop=True)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"))
DIM = X.shape[2]
isLUAD = (df.project == "LUAD").values
nL, nS = int(isLUAD.sum()), int((~isLUAD).sum())
print(f"[cohort] LUAD {nL} + LUSC {nS} = {len(df)}  (Prov-GigaPath {X.shape})", flush=True)
for g in GENES:
    print(f"   {g:<7} LUAD+ {int(df[isLUAD][g].sum()):>4} ({df[isLUAD][g].mean():.3f})"
          f"   LUSC+ {int(df[~isLUAD][g].sum()):>4} ({df[~isLUAD][g].mean():.3f})", flush=True)

# CPTAC-LUAD 外部验证(用 UNI v1 特征, 与 TCGA 侧同空间)
meta = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
fie = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
fmap = {c: r for c, r in zip(fie.case_id, fie.mem_row)}
meta["row"] = meta.submitter_id.map(lambda s: fmap.get(s, -1))
metaL = meta[(meta.disease == "LUAD") & (meta.row >= 0)].reset_index(drop=True)
XE = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"))
YE = metaL[GENES].values.astype(int)
print(f"[ext] CPTAC-LUAD n={len(metaL)}", flush=True)

# 注意: CPTAC 只有 UNI v1, 而上面 X 是 GigaPath -> 维度不同。
# 因此外部验证需要用 GigaPath 训练的模型去跑 1536 维的 GigaPath patch 特征,
# 而 CPTAC 没有 GigaPath -> 外部评估在本脚本里用【UNI 侧训练】的模型。
# 为保持与既有结果可比(脚本 32/42 都是 UNI 训练 -> CPTAC), 这里改用 UNI 特征。
XU = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))
# mem_row 必须取自 UNI 索引 —— 下面索引的是 XU(UNI 特征), 不是 X(GigaPath)。
# 与 ig 求交只为保持 n 与主线可比, 不能用 ig 的 mem_row(两个缓存行布局不同)。
dfu = ig[["case_id"]].merge(iu[["case_id", "mem_row"]], on="case_id", how="inner")
dfu = dfu.merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
dfu = dfu[dfu.project.isin(["LUAD", "LUSC"])].reset_index(drop=True)
# 保证 dfu 与 XU 行号一致: 用 mem_row 映射
isLUAD_U = (dfu.project == "LUAD").values
print(f"[ext-mode] 改用 UNI v1 训练以匹配 CPTAC 特征空间: LUAD {int(isLUAD_U.sum())} + LUSC {int((~isLUAD_U).sum())}", flush=True)


# 关键: dfu 是 merge 出来的, 行序与缓存 CSV 无关。必须先用 mem_row 重排缓存,
# 否则 gather(rows) 里的行号会索引到别人的 patch(本脚本已犯过两次同类错误)。
XU_SUB = np.ascontiguousarray(XU[dfu.mem_row.values.astype(int)])

# 起飞前自检: 若行序映射错了, mean-pool + LR 的 AUC 会掉到 0.55 以下, 直接中止。
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold as _SKF
_pf = []
for _j in [1, 3]:                      # TP53, NF1
    _m = isLUAD_U
    _X = XU_SUB[_m].astype(np.float32).mean(1)
    _y = dfu[GENES].values.astype(int)[_m, _j]
    _o = np.zeros(len(_y))
    for _a, _b in _SKF(5, shuffle=True, random_state=0).split(_X, _y):
        _s = _X[_a].mean(0); _sd = _X[_a].std(0) + 1e-9
        _mm = LogisticRegression(max_iter=2000, class_weight="balanced").fit((_X[_a]-_s)/_sd, _y[_a])
        _o[_b] = _mm.predict_proba((_X[_b]-_s)/_sd)[:, 1]
    _pf.append(roc_auc_score(_y, _o))
print(f"[preflight] LUAD mean-pool+LR  AUC TP53={_pf[0]:.3f}  NF1={_pf[1]:.3f}", flush=True)
# 阈值按基因强度分别设定: TP53 是强基因(应 >0.65), NF1 是本项目最弱基因(UNI 上约 0.55-0.60),
# 但两者都必须明显高于 0.50, 否则说明行序映射仍然错误。
assert _pf[0] > 0.65 and _pf[1] > 0.45, f"行序映射仍然错误(preflight AUC={_pf}), 中止"
print("[preflight] 通过", flush=True)

def gather(rows, rng):
    bt = np.empty((len(rows), S_INST, XU_SUB.shape[2]), np.float16)
    for i, rw in enumerate(rows):
        v = XU_SUB[rw]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


def gather_ext(rows_memrow, rng):
    """CPTAC 外部集专用: rows_memrow 是 CPTAC 缓存的行号, 必须直接索引 XE,
    绝不能走 gather(那会索引 TCGA 的 XU_SUB)。"""
    bt = np.empty((len(rows_memrow), S_INST, XE.shape[2]), np.float16)
    for i, r in enumerate(rows_memrow):
        v = XE[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


# 外部路径 preflight: CPTAC-LUAD mean-pool + LR 应有约 0.66(脚本 31 实测 0.680)
_pfe = []
for _j in [1, 2]:
    _X = XE[metaL.row.values].astype(np.float32).mean(1)
    _y = YE[:, _j]
    _o = np.zeros(len(_y))
    for _a, _b in _SKF(5, shuffle=True, random_state=0).split(_X, _y):
        _s = _X[_a].mean(0); _sd = _X[_a].std(0) + 1e-9
        _mm = LogisticRegression(max_iter=2000, class_weight="balanced").fit((_X[_a]-_s)/_sd, _y[_a])
        _o[_b] = _mm.predict_proba((_X[_b]-_s)/_sd)[:, 1]
    _pfe.append(roc_auc_score(_y, _o))
print(f"[preflight-ext] CPTAC-LUAD mean-pool+LR  TP53={_pfe[0]:.3f}  KEAP1={_pfe[1]:.3f}", flush=True)
assert max(_pfe) > 0.60, f"CPTAC 索引错误(preflight={_pfe}), 中止"
print("[preflight-ext] 通过", flush=True)

EXT_ROWS = np.array([fmap[s] for s in metaL.submitter_id])

def run(cohort_mask, seed):
    """cohort_mask: True 的行参与训练。返回 (LUAD OOF logits, CPTAC 预测)"""
    tr_all = np.flatnonzero(cohort_mask)
    luad_idx = np.flatnonzero(isLUAD_U)
    oof = np.zeros((len(luad_idx), len(GENES)), np.float32)
    ext_acc = np.zeros((len(metaL), len(GENES)), np.float32)
    st = np.clip(dfu[GENES].values.astype(int)[tr_all].sum(1), 0, 3).astype(int)
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr_all)), st)):
        tr = tr_all[a]
        Ytr = dfu[GENES].values.astype(int)[tr]
        torch.manual_seed(seed * 1000 + f)
        pw = torch.tensor(np.where(Ytr.sum(0) > 0, (len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1.),
                          dtype=torch.float32, device=dev)
        net = Net(XU.shape[2], HID, len(GENES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, len(tr) // BATCH))
        rng = np.random.RandomState(seed * 100 + f)
        lf = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(EP):
            net.train(); perm = rng.permutation(len(tr))
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                rr = perm[i:i + BATCH]
                bt = torch.from_numpy(gather(tr[rr], rng)).to(dev, torch.float32)
                yb = torch.from_numpy(Ytr[rr].astype(np.float32)).to(dev)
                loss = lf(net(bt), yb).mul(pw).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        # held-out 的 LUAD 部分
        te_all = tr_all[b]
        te_luad = te_all[isLUAD_U[te_all]]
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            if len(te_luad):
                bt = torch.from_numpy(gather(te_luad, np.random.RandomState(7))).to(dev, torch.float32)
                p = net(bt).float().cpu().numpy()
                pos = {c: i for i, c in enumerate(te_luad)}
                for i, c in enumerate(luad_idx):
                    if c in pos:
                        oof[i] = p[pos[c]]
            bt = torch.from_numpy(gather_ext(EXT_ROWS, np.random.RandomState(11))).to(dev, torch.float32)
            ext_acc += net(bt).float().cpu().numpy() / 5
    return oof, ext_acc


auc = lambda y, p: float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan
YLUAD = dfu[GENES].values.astype(int)[isLUAD_U]
maskA = isLUAD_U.copy()
maskB = np.ones(len(dfu), bool)

res = {}
for name, mk in [("A_luad_only", maskA), ("B_luad_plus_lusc", maskB)]:
    n_luad = int(isLUAD_U.sum())          # OOF 只在 LUAD 子集上, 不是全部行
    res[name] = {"oof": np.zeros((len(SEEDS), n_luad, len(GENES)), np.float32),
                 "ext": np.zeros((len(SEEDS), len(metaL), len(GENES)), np.float32)}
    for seed in SEEDS:
        oof, ext = run(mk, seed)
        res[name]["oof"][seed] = oof
        res[name]["ext"][seed] = ext
        a = np.mean([auc(YLUAD[:, j], oof[:, j]) for j in range(len(GENES))])
        e = np.mean([auc(YE[:, j], ext[:, j]) for j in range(len(GENES))])
        print(f"  [{name}] seed {seed}  LUAD-OOF {a:.4f}  CPTAC-LUAD {e:.4f}", flush=True)

print("\n" + "=" * 100)
print("=== LUAD-only vs LUAD+LUSC (UNI v1, 5 seed) ===")
print(f"{'cohort':<18}" + "".join(f"{g:>10}" for g in GENES) + f"{'mean':>10}{'CPTAC':>10}")
final = {}
for name in ["A_luad_only", "B_luad_plus_lusc"]:
    o = res[name]["oof"].mean(0)
    per = [auc(YLUAD[:, j], o[:, j]) for j in range(len(GENES))]
    e = res[name]["ext"].mean(0)
    per_e = [auc(YE[:, j], e[:, j]) for j in range(len(GENES))]
    final[name] = (per, per_e)
    print(f"{name:<18}" + "".join(f"{v:>10.4f}" for v in per)
          + f"{np.mean(per):>10.4f}{np.mean(per_e):>10.4f}")

print("\n=== 配对差 (B - A), 逐基因逐 seed ===")
for j, g in enumerate(GENES):
    dA = [auc(YLUAD[:, j], res["A_luad_only"]["oof"][s][:, j]) for s in range(len(SEEDS))]
    dB = [auc(YLUAD[:, j], res["B_luad_plus_lusc"]["oof"][s][:, j]) for s in range(len(SEEDS))]
    dd = np.array(dB) - np.array(dA)
    eA = [auc(YE[:, j], res["A_luad_only"]["ext"][s][:, j]) for s in range(len(SEEDS))]
    eB = [auc(YE[:, j], res["B_luad_plus_lusc"]["ext"][s][:, j]) for s in range(len(SEEDS))]
    de = np.array(eB) - np.array(eA)
    print(f"  {g:<7} 内部 {dd.mean():>+8.4f} (同号 {int((dd>0).sum())}/5)   外部 {de.mean():>+8.4f} (同号 {int((de>0).sum())}/5)")

dA = np.mean([auc(YLUAD[:, j], res["A_luad_only"]["oof"].mean(0)[:, j]) for j in range(len(GENES))])
dB = np.mean([auc(YLUAD[:, j], res["B_luad_plus_lusc"]["oof"].mean(0)[:, j]) for j in range(len(GENES))])
eA = np.mean(final["A_luad_only"][1]); eB = np.mean(final["B_luad_plus_lusc"][1])
print(f"\n  MEAN   内部 {dB-dA:>+8.4f}                外部 {eB-eA:>+8.4f}")

rows = []
for name in res:
    for si in range(len(SEEDS)):
        for j, g in enumerate(GENES):
            rows.append(dict(cohort=name, seed=si, gene=g,
                             oof=auc(YLUAD[:, j], res[name]["oof"][si][:, j]),
                             ext=auc(YE[:, j], res[name]["ext"][si][:, j])))
pd.DataFrame(rows).to_csv(os.path.join(C, "luad_vs_luadlusc.csv"), index=False)
print(f"\n[saved] {C}\\luad_vs_luadlusc.csv")
print("DONE", flush=True)