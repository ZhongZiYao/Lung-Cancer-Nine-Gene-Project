"""
deepgem_test/step5_slice_ensemble.py  (Plan C)

对每个 case 的 3 个 SM slice 各自跑 DeepGEM EGFR 模型, 然后 mean 3 logits → 1 case logit.
Compare with step5_predict_egfr.py 单一 combined forward 的结果.

输出:
  pid / label / pred_combined / prob_combined / pred_ensemble / prob_ensemble / hit
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


def load_one_slice(path):
    """读 1 个 slice.feats
    支持 2 种形态:
      - list-of-dict 的单个 .pkl (combined 模式): 返回 np.stack([{'val': arr}].val)
      - per-patch 目录 (含 *.pkl): stack 所有
    返回 np.ndarray [N, 768]
    """
    if path.is_file():
        with open(path, "rb") as f:
            patch_list = pickle.load(f)
        return np.stack([p["val"] for p in patch_list]).astype(np.float32)
    # directory: load all p0..pN.pkl
    files = sorted(path.glob("*.pkl"), key=lambda p: int(p.stem.split("_")[0][1:]))
    arrs = []
    for f in files:
        with open(f, "rb") as inf:
            d = pickle.load(inf)
        arrs.append(d["val"].astype(np.float32))
    return np.stack(arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal", default=str(HERE / "data" / "internal_3case.pickle"))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--feat-dir", default=str(HERE / "deepgem_feat"))
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    args = ap.parse_args()

    sys.path.insert(0, str(DEEPGEM_MAIN))
    from util.misc import NestedTensor
    from models.model_deepgem import DeepGEM

    # 1) label
    info = load_internal_pickle(Path(args.internal), args.wsi_type, args.gene)
    pids, labels = info["pid"], info["label"]
    print(f"[data] {len(pids)} cases, gene={args.gene}")

    # 2) 加载模型
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    param = ckpt["parameter"]
    sd = ckpt["checkpoint"]
    feature_len = param["feature_len"]
    for k in sd.keys():
        if "patch_to_embedding.weight" in k:
            patch_dim = sd[k].shape[1]
            break
    print(f"[ckpt] patch_dim={patch_dim}, feature_len={feature_len}, "
          f"hidden_dim={param['hidden_dim']}, depth={param['depth']}")
    model = DeepGEM(num_classes=2, patch_dim=patch_dim,
                    dim=param["hidden_dim"], depth=param["depth"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    # 3) 找每个 case 的 slice .pkl
    # per-patch 模式: feat_dir/<case_id>/s<i>_<hash>/p<i>_f768.pkl
    feat_dir = Path(args.feat_dir)
    case_slices = {}
    if feat_dir.exists():
        for case_dir in sorted([d for d in feat_dir.iterdir() if d.is_dir()]):
            case_id = case_dir.name
            for slice_dir in sorted([d for d in case_dir.iterdir() if d.is_dir()]):
                try:
                    slice_idx = int(slice_dir.name.split("_")[0][1:])
                except (ValueError, IndexError):
                    continue
                case_slices.setdefault(case_id, {})[slice_idx] = slice_dir
    print(f"\n[feats] case -> slices:")
    for c, slices in case_slices.items():
        print(f"  {c}: {len(slices)} slices ({sorted(slices.keys())})")

    # 4) Forward 每个 case 的每个 slice,ensemble logits
    print(f"\n[forward] {args.gene} predictions (3 case × ~3 slice each)")
    print(f"  {'pid':<14} {'label':<5} | "
          f"{'combined':<20} | {'ensemble':<20}")
    print(f"  {'':14} {'':5} | "
          f"{'neg':<8} {'pos':<8} {'pred':<4} | "
          f"{'neg':<8} {'pos':<8} {'pred':<4} | hit")
    print(f"  {'-'*90}")

    n = len(pids)
    correct_combined = 0
    correct_ensemble = 0

    combined_log = []
    ensemble_log = []

    for pid, label in zip(pids, labels):
        if pid not in case_slices:
            print(f"  [skip] {pid}: no slices")
            continue
        slices = case_slices[pid]
        slice_idxs_sorted = sorted(slices.keys())

        # === A. Combined (旧 step5): 1 个 case 1 个 combined .pkl ===
        # 这里我们跳过 c,s combined, 直接看 ensemble

        # === B. Slice-level ensemble ===
        slice_logits_list = []
        slice_probs_list = []
        for s_idx in slice_idxs_sorted:
            slice_pkl = slices[s_idx]
            feats = load_one_slice(slice_pkl)
            if feats.shape[1] != patch_dim:
                print(f"  [WARN] {pid} slice {s_idx}: dim mismatch")
                continue
            x, m = make_nested_tensor(feats, feature_len)
            sample = NestedTensor(x, m)
            with torch.no_grad():
                logits = model(sample)  # [1, 2]
            slice_logits_list.append(logits[0].cpu().numpy())  # [2]
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            slice_probs_list.append(probs)

        if not slice_logits_list:
            continue
        from scipy.special import softmax
        # mean logits
        mean_logits = np.mean(slice_logits_list, axis=0)
        mean_probs = softmax(mean_logits)
        ensemble_pred = int(np.argmax(mean_probs))
        ensemble_hit = "+" if ensemble_pred == label else "-"
        if ensemble_pred == label:
            correct_ensemble += 1

        # === 也算下 combined(所有 patch 拼一个 case)对比 ===
        # 找 combined .pkl
        combined_pkl = feat_dir / f"{pid}.pkl"
        if combined_pkl.exists():
            all_feats = load_one_slice(combined_pkl)
            x, m = make_nested_tensor(all_feats, feature_len)
            sample = NestedTensor(x, m)
            with torch.no_grad():
                logits = model(sample)
            combined_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            combined_pred = int(np.argmax(combined_probs))
            combined_hit = "+" if combined_pred == label else "-"
            if combined_pred == label:
                correct_combined += 1
        else:
            combined_probs = (0.5, 0.5)
            combined_pred = -1
            combined_hit = "?"

        print(f"  {pid:<14} {label:<5} | "
              f"{combined_probs[0]:<8.4f} {combined_probs[1]:<8.4f} {combined_pred:<4} | "
              f"{mean_probs[0]:<8.4f} {mean_probs[1]:<8.4f} {ensemble_pred:<4} | "
              f"e={ensemble_hit}/c={combined_hit}")

        combined_log.append({"pid": pid, "label": label,
                             "pred": combined_pred,
                             "pos_prob": float(combined_probs[1])})
        ensemble_log.append({"pid": pid, "label": label,
                             "pred": ensemble_pred,
                             "pos_prob": float(mean_probs[1])})

    n_actual = len(ensemble_log)
    print(f"\n[summary]")
    print(f"  combined accuracy: {correct_combined}/{n_actual} = "
          f"{correct_combined/max(n_actual,1):.2%}")
    print(f"  ensemble accuracy: {correct_ensemble}/{n_actual} = "
          f"{correct_ensemble/max(n_actual,1):.2%}")
    print(f"\n[interpret]")
    print(f"  ensemble = 对每个 case 的多个 SM 各自 forward, 然后 mean logits")
    print(f"  combined = 把所有 slice 的 patches 拼 1 个 case forward (旧 step5)")
    print(f"  小样本 3 case, 1 正 2 负, AUC 评估意义有限。")


if __name__ == "__main__":
    main()
