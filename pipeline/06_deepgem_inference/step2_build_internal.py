"""合并 per-patch 特征 → per-case .pkl,并写 internal.pickle。

输入:
    --feat        : deepgem_feat/<case>/<SM>/<x>_<y>.pkl  (step1 产物)
    --labels      : project_resources/data_public/9gene_panel_LUAD.csv
    --cases       : 想跑的 case_id 列表(json 或一行一个 case_id 文本)
                    默认: 当前 feat 目录下所有有 feat 的 case
    --gene        : 要测的基因,如 EGFR / KRAS
    --wsi-type    : WSI 类型,DeepGEM 用 ExcisionalBiopsy (ex-biopsy 还是切除活检)
    --out-feat    : combined_feat 输出目录
    --out-pickle  : internal.pickle 输出路径

输出:
    combined_feat/<case_id>.pkl  ← list[dict{val,feat_name}],全部 patch 的 768 维展平
    internal.pickle              ← 满足 DeepGEMDataset.build 期望的格式:
        {
          "ExcisionalBiopsy": {
            "EGFR": {
              "test": {"pid": [...], "label": [...]}
            }
          }
        }
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_case_features(feat_root: Path, case_id: str) -> list[dict]:
    """读 case_id 名下所有 .pkl,返回 DeepGEMDataset 兼容的 list[dict]"""
    per_patch = []
    case_dir = feat_root / case_id
    if not case_dir.exists():
        return []
    for pkl_path in sorted(case_dir.rglob("*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        # step1 写的是 list[dict{val,feat_name}],展平
        if isinstance(d, list):
            per_patch.extend(d)
        else:
            per_patch.append(d)
    return per_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=Path, default=Path("./deepgem_feat"))
    ap.add_argument("--labels", type=Path,
                    default=Path("./project_resources/data_public/9gene_panel_LUAD.csv"))
    ap.add_argument("--gene", default="EGFR",
                    help="要预测的基因,必须在 labels CSV 里")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy",
                    help="DeepGEM 模型按 WSI 类型分; TCGA 里 ex-biopsy 选 ExcisionalBiopsy")
    ap.add_argument("--cases", type=Path, default=None,
                    help="case_id 列表(json). 默认 = feat 目录下所有 case")
    ap.add_argument("--out-feat", type=Path, default=Path("./deepgem_combined_feat"))
    ap.add_argument("--out-pickle", type=Path,
                    default=Path("./deepgem_internal.pickle"))
    args = ap.parse_args()

    # ---- label ----
    if not args.labels.exists():
        sys.exit(f"[FATAL] labels csv not found: {args.labels}")
    df = pd.read_csv(args.labels)
    if args.gene not in df.columns:
        sys.exit(f"[FATAL] gene '{args.gene}' not in {args.labels}: {list(df.columns)}")
    label_map = dict(zip(df["case_id"], df[args.gene].astype(int)))
    print(f"[INFO] loaded {len(label_map)} case labels from {args.labels.name}")

    # ---- case 列表 ----
    if args.cases is not None:
        case_ids = json.loads(args.cases.read_text())
        case_ids = [c for c in case_ids if c]
        print(f"[INFO] using {len(case_ids)} cases from --cases json")
    else:
        case_ids = sorted(p.name for p in args.feat.iterdir() if p.is_dir())
        print(f"[INFO] auto-discovered {len(case_ids)} cases from {args.feat}")

    # ---- 每个 case 合并 per-patch → per-case .pkl ----
    args.out_feat.mkdir(parents=True, exist_ok=True)
    pids, labels = [], []
    missing_label = 0
    missing_feat = 0
    for case_id in case_ids:
        feats = load_case_features(args.feat, case_id)
        if not feats:
            print(f"[WARN] {case_id}: no features under {args.feat}/{case_id}")
            missing_feat += 1
            continue
        if case_id not in label_map:
            print(f"[WARN] {case_id}: no label in {args.labels.name}, skipping")
            missing_label += 1
            continue
        out_path = args.out_feat / f"{case_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(feats, f)
        pids.append(case_id)
        labels.append(int(label_map[case_id]))
        print(f"  {case_id}: {len(feats)} patches  label={labels[-1]}")

    # ---- 写 internal.pickle (DeepGEMDataset.build 期望格式) ----
    payload = {
        args.wsi_type: {
            args.gene: {
                "test": {
                    "pid": pids,
                    "label": labels,
                }
            }
        }
    }
    with open(args.out_pickle, "wb") as f:
        pickle.dump(payload, f)

    n_pos = sum(labels)
    print()
    print(f"[OK] {len(pids)} cases → {args.out_feat}/")
    print(f"[OK] 内部:     EGFR+={n_pos}  EGFR-={len(labels)-n_pos}")
    if missing_feat:
        print(f"[WARN] {missing_feat} case(s) 没有 patch 特征")
    if missing_label:
        print(f"[WARN] {missing_label} case(s) 没有 label,跳过")
    print(f"[OK] internal.pickle → {args.out_pickle}")


if __name__ == "__main__":
    main()
