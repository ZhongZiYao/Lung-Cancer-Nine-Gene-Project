"""Prepare DeepGEM-format input from newly downloaded WSI datasets.

After downloading NSCLC-Radiogenomics / CPTAC-LUAD / TCGA-LUAD,
this script:
    1. Collects WSI file paths
    2. Builds a DeepGEM-style sample.csv (pid, label) using
       the dataset's clinical/mutation table
    3. Optionally invokes step1-step4 of DeepGEM data_prepare

For each dataset, you need:
    - WSI folder (full of .svs / .tiff / .dcm)
    - Mutation label table (CSV with columns: case_id, gene, label)
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
DEEPGEM_DATA_PREPARE = ROOT / "DeepGEM" / "data_prepare"


GENES = ["EGFR", "KRAS", "TP53", "ALK", "ROS1", "LRP1B",
         "BRAF", "MET", "HER2"]


def find_wsi(folder: Path):
    """Yield (case_id, wsi_path) for each WSI file."""
    exts = [".svs", ".tiff", ".tif", ".ndpi", ".scn", ".mrxs", ".vms", ".bif"]
    for f in folder.rglob("*"):
        if f.suffix.lower() in exts and f.is_file():
            # case_id from filename - strip the slide/section suffix
            case_id = f.stem.split("-")[0:3]
            yield "-".join(case_id), f


def build_sample_csv(wsi_dir: Path, mutation_csv: Path, out_csv: Path,
                     wsi_type: str = "ExcisionalBiopsy"):
    """Build DeepGEM-style sample.csv from a mutation table.

    mutation_csv columns expected: case_id, EGFR, KRAS, TP53, ALK, ROS1, LRP1B, BRAF, MET, HER2
    Values: 1 (mutant), 0 (wild-type), NaN (unknown)
    """
    mut = pd.read_csv(mutation_csv)
    mut = mut.set_index("case_id")
    rows = []
    for case_id, wsi_path in find_wsi(wsi_dir):
        if case_id not in mut.index:
            continue
        row = {"pid": case_id, "wsi_path": str(wsi_path)}
        for gene in GENES:
            if gene in mut.columns:
                v = mut.loc[case_id, gene]
                row[f"label_{gene}"] = int(v) if pd.notna(v) else -1
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(df)} rows -> {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", type=Path, required=True)
    ap.add_argument("--mutation-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT / "data" / "deepgem_input.csv")
    ap.add_argument("--wsi-type", default="ExcisionalBiopsy")
    args = ap.parse_args()

    build_sample_csv(args.wsi_dir, args.mutation_csv, args.out_csv, args.wsi_type)


if __name__ == "__main__":
    main()