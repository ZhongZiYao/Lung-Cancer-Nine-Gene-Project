"""
step3_build_deepgem_inputs.py
─────────────────────────────
从 9gene_panel_LUAD.csv 构造 DeepGEM test.py 期望的 internal.pickle,
并构造一个 dummy checkpoint (随机初始化 DeepGEM)。

用法:
    python step3_build_deepgem_inputs.py --cases TCGA-05-4382 TCGA-05-4245 TCGA-05-4249 --gene EGFR
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch

# DeepGEM 路径
DEEPGEM = Path("DeepGEM/main").resolve()
sys.path.insert(0, str(DEEPGEM))


def load_labels(csv_path: Path) -> dict:
    """case_id -> {gene: 0/1 dict}"""
    label_df = {}
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            pid = parts[0]
            vals = parts[1:]
            # CSV 列顺序: EGFR,KRAS,ALK,ROS1,TP53,BRAF,PIK3CA,ERBB2,NRAS,RET
            label_df[pid] = {g: int(v) for g, v in zip(header[1:11], vals)}
    return label_df


def make_internal_pickle(cases: list[str], labels: dict, gene: str,
                         feat_dir: Path, wsi_type: str = "ExcisionalBiopsy") -> dict:
    """构造 DeepGEM 期望的多层 dict 格式"""
    # 选中目标 gene 的 labels
    target_labels = [labels[c][gene] for c in cases]
    feat_fps = [str(feat_dir / f"{c}.pkl") for c in cases]

    data = {wsi_type: {}}
    # 给 9 个基因都建空 train/val(避免 KeyError)
    genes = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF",
             "PIK3CA", "ERBB2", "NRAS", "RET"]
    for g in genes:
        if g == gene:
            data[wsi_type][g] = {
                "train": {"pid": [], "label": [], "feat_fp": []},
                "val":   {"pid": [], "label": [], "feat_fp": []},
                "test":  {"pid": cases, "label": target_labels, "feat_fp": feat_fps},
            }
        else:
            # 其他基因:train/val/test 全部空,但要给 1 个 case 让内部循环不崩
            data[wsi_type][g] = {
                "train": {"pid": [], "label": [], "feat_fp": []},
                "val":   {"pid": [], "label": [], "feat_fp": []},
                "test":  {"pid": [], "label": [], "feat_fp": []},
            }
    return data


def make_dummy_checkpoint(out_path: Path, ckpt_dim: int = 768):
    """DeepGEM 期望的 checkpoint 格式: pickle {'parameter': {...}, 'checkpoint': state_dict}

    用随机初始化 DeepGEM。
    ⚠️ DeepGEM 默认 patch_dim=768 我们改成 1024(UNI 维度)
    """
    from models.model_deepgem import DeepGEM

    # ↓↓↓ 用 1024 因为你后面用 UNI 特征;DeepGEM 限制是 patch_dim,
    # 而我们之前 step2 截断到 768, 所以这里其实可以是 768
    # ⚠️ sanity check, 改这俩关键就能跑
    model = DeepGEM(
        num_classes=2,
        patch_dim=ckpt_dim,
        dim=32,
        depth=6,
        num_queries=5,
    )

    # 触发一次 forward 确保参数都建好了
    dummy_input = torch.randn(1, 500, ckpt_dim)
    from util.misc import NestedTensor
    samples = NestedTensor(dummy_input, torch.zeros(500, dtype=torch.bool))
    out = model(samples)
    print(f"[dummy] forward OK, output shape: {out.shape}")

    ckpt = {
        "parameter": {
            "hidden_dim": 32,
            "depth": 6,
            "feature_len": 500,
            "batch_size": 1,
        },
        "checkpoint": model.state_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[dummy] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True,
                    help="3 个 case id, e.g. TCGA-05-4382 TCGA-05-4245 TCGA-05-4249")
    ap.add_argument("--gene", default="EGFR",
                    help="要测的基因,默认 EGFR")
    ap.add_argument("--csv", default="transfer_package/data/MC3/9gene_panel_LUAD.csv")
    ap.add_argument("--feat-dir", default="pipeline/deepgem_feat")
    ap.add_argument("--out-internal",
                    default="DeepGEM/main/data/internal/internal_3case.pickle")
    ap.add_argument("--out-ckpt",
                    default="DeepGEM/main/checkpoints/dummy_3case.pickle")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    ap.add_argument("--patch-dim", type=int, default=1024,
                    help="DeepGEM 模型 patch_dim 输入维度")
    args = ap.parse_args()

    labels = load_labels(Path(args.csv))
    missing = [c for c in args.cases if c not in labels]
    if missing:
        print(f"[ERR] cases not in CSV: {missing}")
        return

    # 1) build internal.pickle
    data = make_internal_pickle(
        args.cases, labels, args.gene,
        Path(args.feat_dir), wsi_type=args.wsi_type,
    )
    out_int = Path(args.out_internal)
    out_int.parent.mkdir(parents=True, exist_ok=True)
    with open(out_int, "wb") as f:
        pickle.dump(data, f)
    print(f"[internal.pickle] saved -> {out_int}")

    # 2) dummy checkpoint
    make_dummy_checkpoint(Path(args.out_ckpt), ckpt_dim=args.patch_dim)

    # 3) 报告 labels
    print(f"\n[{args.gene} labels for {len(args.cases)} cases]:")
    for c in args.cases:
        v = labels[c][args.gene]
        print(f"  {c}: {v} (1={args.gene} mutant, 0=WT)")


if __name__ == "__main__":
    main()
