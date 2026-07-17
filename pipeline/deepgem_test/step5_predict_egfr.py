"""
deepgem_test/step5_predict_egfr.py
────────────────────────────────
加载 DeepGEM 已训练好的 EGFR pickle (modelTCGA_ExcisionalBiopsy_EGFR.pickle)
+ 前面 step3 出来的 .pkl patch 特征
+ 前面 step4 出来的 internal.pickle 拿到 label

跑 forward 拿 9 维 logits (patch-level + WSI-level),
softmax 后计算 positive probability vs EGFR label

输出每 case 的 positive probability,以及 accuracy。

⚠️ 注意:DeepGEM 的 forward 会触发 prototype + patch_classifier 两条路径,
   详见 main/engine.py L87-L108。我们这里跑测试,只关心 wsi_classifier_output。
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).parent.resolve()
DEEPGEM_MAIN = HERE.parent.parent / "DeepGEM" / "main"
CKPT = HERE.parent.parent / "DeepGEM" / "checkpoints" / "DeepGEM_TCGA" / "modelTCGA_ExcisionalBiopsy_EGFR.pickle"


def load_internal_pickle(path: Path, wsi_type: str, gene: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[wsi_type][gene]["test"]


def make_nested_tensor(features: np.ndarray, feature_len: int = 500):
    """模拟 util.misc.NestedTensor 的 (tensor, mask)"""
    n = features.shape[0]
    if n > feature_len:
        feats = features[:feature_len]
        mask = np.ones(feature_len, dtype=bool)
    elif n < feature_len:
        pad_n = feature_len - n
        feats = np.concatenate([features, np.zeros((pad_n, features.shape[1]))])
        mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad_n, dtype=bool)])
    else:
        feats = features
        mask = np.ones(feature_len, dtype=bool)
    return (torch.from_numpy(feats.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", default=str(HERE / "data" / "internal_3case.pickle"))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    ap.add_argument("--feat-dir", default=str(HERE / "deepgem_feat"))
    args = ap.parse_args()

    # DeepGEM 路径加进 sys.path,以便 import models / util.misc
    sys.path.insert(0, str(DEEPGEM_MAIN))
    from util.misc import NestedTensor
    from models.model_deepgem import DeepGEM

    # 1) label
    info = load_internal_pickle(Path(args.internal), args.wsi_type, args.gene)
    pids, labels, feat_fps = info["pid"], info["label"], info["feat_fp"]
    print(f"[data] {len(pids)} cases, gene={args.gene}")
    for p, l in zip(pids, labels):
        print(f"  {p}: label={l} ({'MUTANT' if l==1 else 'WT'})")

    # 2) 加载模型 + 加载权重
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    param = ckpt["parameter"]
    sd = ckpt["checkpoint"]
    feature_len = param["feature_len"]

    # 检查 patch_dim
    for k in sd.keys():
        if "patch_to_embedding.weight" in k:
            patch_dim = sd[k].shape[1]
            print(f"[ckpt] patch_dim={patch_dim}, feature_len={feature_len}, "
                  f"hidden_dim={param['hidden_dim']}, depth={param['depth']}")
            break

    model = DeepGEM(
        num_classes=2,
        patch_dim=patch_dim,
        dim=param["hidden_dim"],
        depth=param["depth"],
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  first 3 missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"  first 3 unexpected: {unexpected[:3]}")
    model.eval()

    # 3) 对每个 case 跑 forward
    print(f"\n[forward] {args.gene} predictions")
    print(f"  {'pid':<14} {'label':<8} {'neg_prob':<10} {'pos_prob':<10} {'pred':<5} {'hit':<4}")
    print(f"  {'-'*60}")

    correct = 0
    for pid, label, fp in zip(pids, labels, feat_fps):
        # 读 patch 特征
        with open(fp, "rb") as f:
            patch_list = pickle.load(f)
        feats = np.stack([p["val"] for p in patch_list]).astype(np.float32)  # [N, 768]
        if feats.shape[1] != patch_dim:
            print(f"  [WARN] {pid}: feat dim {feats.shape[1]} != model patch_dim {patch_dim}")

        x, m = make_nested_tensor(feats, feature_len)
        sample = NestedTensor(x, m)

        with torch.no_grad():
            logits = model(sample)  # [1, 2]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # [2]
        pred = int(np.argmax(probs))
        hit = "+" if pred == label else "-"
        if pred == label:
            correct += 1
        print(f"  {pid:<14} {label:<8} {probs[0]:<10.4f} {probs[1]:<10.4f} {pred:<5} {hit:<4}")

    n = len(pids)
    print(f"\n[summary] correct={correct}/{n}, accuracy={correct/n:.2%}")
    print(f"\n[interpret]")
    print(f"  - 这不是 trained model 的真实表现,只是 sanity check:管线通了吗？")
    print(f"  - 若要 model 训好的效果,需保证 patch 来自同 source (CTransPath 训在 TCGA LUAD 上)")
    print(f"  - 现在用你的 UNI-extracted fallback (step3 截断到 768) 跑,效果会很差")
    print(f"  - 如果要准:必须 step2 用 CTransPath 抽,不要用 UNI 截断")


if __name__ == "__main__":
    main()
