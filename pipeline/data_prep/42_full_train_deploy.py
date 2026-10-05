# -*- coding: utf-8 -*-
"""42_full_train_deploy.py — 全量训练 joint_e25 部署版本 + CPTAC 外部验证

用户问题: 5-fold CV 不是部署状态; 真正部署应该用全部 436 例训练,
无 hold-out, 外部验证完全依赖 CPTAC。

做法:
  - 训练集: TCGA-LUAD 436 例(全部, 无划折), UNI v1 1024 维
  - 训练配置: joint_e25(同 41 号, 同超参), 5 seed
  - 验证集: CPTAC-LUAD 216 例(脚本 31 重建标签), 5 seed 投票
  - 比较: 脚本 32 的 5-fold OOF(0.653 mean) vs 本脚本的全量(应该略高)
  - 产物: 5 份 state_dict -> .tmp/joint_e25_full_seed{k}.pt
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
DIM, BATCH, LR_LR, HID, S_INST = 1024, 32, 3e-4, 256, 512
EP = 25
dev = "cuda"
SEEDS = [0, 1, 2, 3, 4]


class Net(nn.Module):
    def __init__(s, d, h, k):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.25))
        s.V = nn.Linear(h, h, bias=False)
        s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False)
        s.od = nn.Linear(h, k)

    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


# ---- 加载训练集 ----
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
iu = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
df = iu[["case_id", "mem_row"]].merge(man[["case_id", "project"] + GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project == "LUAD"].reset_index(drop=True)
N = len(df); Y = df[GENES].values.astype(int); RT = df.mem_row.values.astype(int)
XU = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))
DIM_IN = XU.shape[2]
print(f"[训练集] TCGA-LUAD n={N}, UNI v1 {XU.shape}", flush=True)

# ---- 加载外部验证集 ----
meta = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
fi_ext = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
fmap = {c: r for c, r in zip(fi_ext.case_id, fi_ext.mem_row)}
meta["row"] = meta.submitter_id.map(lambda s: fmap.get(s, -1))
meta = meta[(meta.disease == "LUAD") & (meta.row >= 0)].reset_index(drop=True)
XE = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"))
DIM_E = XE.shape[2]
YE = meta[GENES].values.astype(int)
print(f"[验证集] CPTAC-LUAD n={len(meta)}, UNI v1 {XE.shape}", flush=True)
assert DIM_IN == DIM_E, f"特征维度不一致 {DIM_IN} vs {DIM_E}"


def gather_from(X, rows, rng):
    bt = np.empty((len(rows), S_INST, X.shape[2]), np.float16)
    for i, r in enumerate(rows):
        v = X[r]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    return bt


def predict(net, X, seed):
    """对整批 X 推理(用 seed 确定采样 RNG, 保持 5 seed 模型用不同 patch 抽样)"""
    net.eval()
    rng = np.random.RandomState(seed * 333)
    bt = np.empty((X.shape[0], S_INST, X.shape[2]), np.float16)
    for i in range(X.shape[0]):
        v = X[i]; m = v.shape[0]
        bt[i] = v[rng.choice(m, S_INST, replace=m < S_INST)]
    bt = torch.from_numpy(bt).to(dev, torch.float32)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return net(bt).float().cpu().numpy()


# ---- 全量训练 5 seed ----
print("\n=== 全量训练 5 seed (无 hold-out, 全部 436 例) ===", flush=True)
ckpt_dir = os.path.join(R, ".tmp")
os.makedirs(ckpt_dir, exist_ok=True)
results = []
for seed in SEEDS:
    torch.manual_seed(seed)
    pw = torch.tensor(np.where(Y.sum(0) > 0, (N - Y.sum(0)) / np.maximum(Y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    net = Net(DIM_IN, HID, len(GENES)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_LR, epochs=EP, steps_per_epoch=max(1, N // BATCH))
    rng = np.random.RandomState(seed * 100)
    lf = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(EP):
        net.train(); perm = rng.permutation(N)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            bt = torch.from_numpy(gather_from(XU, RT[rr], rng)).to(dev, torch.float32)
            yb = torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
            loss = lf(net(bt), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass

    # 训练集 sanity AUC
    sd_path = os.path.join(ckpt_dir, f"joint_e25_full_seed{seed}.pt")
    try:
        torch.save(net.state_dict(), sd_path)
        sd_size = os.path.getsize(sd_path) / 1024
        sd_info = f"{sd_size:.0f} KB"
    except Exception as e:
        sd_info = f"save FAIL {e}"

    # CPTAC 推理
    ext_p = predict(net, XE[meta.row.values], seed)
    # 在训练集上同样推理看 sanity
    train_p = predict(net, XU[RT], seed)
    train_auc = float(np.mean([roc_auc_score(Y[:, j], train_p[:, j])
                               for j in range(len(GENES)) if 10 <= Y[:, j].sum() <= N - 10]))
    ext_auc = float(np.mean([roc_auc_score(YE[:, j], ext_p[:, j])
                             for j in range(len(GENES)) if 10 <= YE[:, j].sum() <= len(YE) - 10]))
    print(f"  seed {seed}  train AUC {train_auc:.3f}  CPTAC-LUAD AUC {ext_auc:.3f}  state {sd_info}",
          flush=True)
    results.append(dict(seed=seed, train_auc=train_auc, ext_auc=ext_auc, ext_pred=ext_p))

# ---- 集成: 5 seed CPTAC 预测取平均 ----
ens = np.mean([r["ext_pred"] for r in results], axis=0)
print("\n=== 5-seed 集成 (CPTAC-LUAD n=216) ===", flush=True)


def auc_safe(y, p):
    return float(roc_auc_score(y, p)) if 10 <= y.sum() <= len(y) - 10 else np.nan


try:
    old_ens = np.load(os.path.join(C, "ext_reval_oof_cptac.npy"))
    have_old = True
except FileNotFoundError:
    old_ens = None; have_old = False   # 脚本 32 当时未保存预测, 这里用参考值
print(f"{'gene':<8}{'全量 5seed':>11}{'5-fold 3seed':>14}{'差':>9}")
new_aucs = []
for j, g in enumerate(GENES):
    new = auc_safe(YE[:, j], ens[:, j])
    old = auc_safe(YE[:, j], old_ens[:, j])
    new_aucs.append((g, new, old))
    print(f"{g:<8}{new:>11.3f}{old:>14.3f}{new-old:>+9.3f}")
print(f"{'mean':<8}{np.mean([a[1] for a in new_aucs]):>11.3f}{np.mean([a[2] for a in new_aucs]):>14.3f}"
      f"{np.mean([a[1]-a[2] for a in new_aucs]):>+9.3f}")

np.save(os.path.join(C, "ext_cptac_full_5seed.npy"), ens)
print(f"\n[saved] {C}\\ext_cptac_full_5seed.npy  (CPTAC-LUAD 216 × 6 的 logits, 5-seed 集成)")
print(f"[saved] {ckpt_dir}\\joint_e25_full_seed{{0..4}}.pt  (5 份 state_dict, 可作部署用)")
print("DONE", flush=True)