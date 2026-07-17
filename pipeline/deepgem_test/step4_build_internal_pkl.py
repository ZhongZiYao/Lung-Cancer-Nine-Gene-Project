"""
deepgem_test/step4_build_internal_pkl.py
───────────────────────────────────────
读取 9gene_panel_LUAD.csv,根据 3 case id 构造:
  1) DeepGEM 期望的 internal.pickle 多层 dict
  2) 并打印每个 case 的 EGFR 真实 label

输出位置:deepgem_test/data/internal_3case.pickle
"""
import argparse
import csv
import pickle
from pathlib import Path


HERE = Path(__file__).parent.resolve()
DEFAULT_CSV = HERE.parent.parent / "transfer_package" / "data" / "MC3" / "9gene_panel_LUAD.csv"
DEFAULT_OUT = HERE / "data" / "internal_3case.pickle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--cases", nargs="*",
                    default=["TCGA-05-4382", "TCGA-05-4249", "TCGA-05-4395"])
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    ap.add_argument("--feat-dir", default=str(HERE / "deepgem_feat"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # 读 CSV
    label_df = {}
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        # 9-gene 列顺序看 header
        gene_cols = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF",
                     "PIK3CA", "ERBB2", "NRAS", "RET", "MET"]
        for row in reader:
            pid = row["case_id"]
            label_df[pid] = {g: int(row[g]) for g in gene_cols if g in row}

    # 9 个基因都构造(避免 DeepGEM 内部 KeyError)
    genes = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF",
             "PIK3CA", "ERBB2", "NRAS", "RET", "TP53"]
    data = {args.wsi_type: {}}

    for g in genes:
        if g == args.gene:
            # 目标基因有 label
            target_labels = []
            feat_fps = []
            for c in args.cases:
                if c not in label_df:
                    print(f"[ERR] {c} not in CSV; abort")
                    return
                target_labels.append(label_df[c][g])
                feat_fps.append(str(Path(args.feat_dir) / f"{c}.pkl"))
            data[args.wsi_type][g] = {
                "train": {"pid": [], "label": [], "feat_fp": []},
                "val":   {"pid": [], "label": [], "feat_fp": []},
                "test":  {"pid": args.cases, "label": target_labels, "feat_fp": feat_fps},
            }
        else:
            data[args.wsi_type][g] = {
                "train": {"pid": [], "label": [], "feat_fp": []},
                "val":   {"pid": [], "label": [], "feat_fp": []},
                "test":  {"pid": [], "label": [], "feat_fp": []},
            }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        pickle.dump(data, f)

    # 打印
    print(f"\n[{args.gene} labels for {len(args.cases)} cases]:")
    for c in args.cases:
        v = label_df[c][args.gene]
        sign = "MUTANT" if v == 1 else "WT"
        print(f"  {c}: {v} ({sign})")

    print(f"\n[internal.pickle] saved -> {out_p}")


if __name__ == "__main__":
    main()
