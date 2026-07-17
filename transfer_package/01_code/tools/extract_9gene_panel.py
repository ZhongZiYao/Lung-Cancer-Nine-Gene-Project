"""
extract_9gene_panel.py — 从 MC3 MAF 抽出 TCGA-LUAD 9 基因突变 panel,导出 CSV

输入: mc3.v0.2.8.PUBLIC.maf.gz (从 tools/download_mc3_maf.py 得到)
输出: 9gene_panel_LUAD.csv + 控制台打印每基因阳性样本数

判定规则:
  任意 PASS + Missense_Mutation/In_Frame_Del/In_Frame_Ins/... 蛋白功能改变类变异 → 该基因阳性
  只看 TCGA-LUAD (project = TCGA-LUAD, alias TCGA-LUAD)

需要 9 基因(合同 + NCCN + DeepGEM 共有):
  EGFR, KRAS, ALK, ROS1, TP53, BRAF, PIK3CA, ERBB2(=HER2), NRAS, RET, MET
  → 11 个,留有冗余;实际可减到 9

用法:
  python tools/extract_9gene_panel.py --maf D:/pan-caner/data/MC3/mc3.v0.2.8.PUBLIC.maf.gz
  python tools/extract_9gene_panel.py --maf ... --out D:/pan-caner/data/MC3/9gene_panel_LUAD.csv
  python tools/extract_9gene_panel.py --maf ... --genes EGFR,KRAS,ALK  # 只输出几个
  python tools/extract_9gene_panel.py --maf ... --include-coad-reads  # 不止 LUAD,把所有 project 都加上 9 基因栏
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

# 9 基因 panel (DeepGEM 6 + 商业的 HER2/ERBB2 + BRAF + NRAS + RET + MET = 11,留冗余)
DEFAULT_GENES = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF", "PIK3CA", "ERBB2", "NRAS", "RET", "MET"]

# MC3 用 HUGO;确认 ERBB2 即 HER2(分子生物通用命名)

# 这些 Variant_Classification 都算该基因的"功能改变性"突变(阳性)
POSITIVE_CLASSES = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}


def is_pass_filter(val: str) -> bool:
    v = (val or "").strip().upper()
    # PASS / . / 缺省 都视为通过
    return v in {"PASS", "", "."}


def extract_case_id(barcode: str) -> str:
    """TCGA barcode e.g. TCGA-05-4398-01A → TCGA-05-4398 (case_id,patient-level)"""
    parts = (barcode or "").split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return barcode or "UNKNOWN"


def is_luad(barcode: str, whitelist: set[str] | None = None) -> bool:
    """LUAD 判定:
      - 若提供 whitelist(case_id 集合,如 cBioPortal 的 TCGA-LUAD 病例列表),用它过滤(推荐)
      - 否则 fallback: barcode 第 2 段 site code == '05'(legacy site 05 全是 LUAD,
        只覆盖 ~30 cases,严重不全,只为离线 fallback)
    """
    parts = (barcode or "").split("-")
    if len(parts) < 3 or parts[0] != "TCGA":
        return False
    case_id = "-".join(parts[:3])
    if whitelist is not None:
        return case_id in whitelist
    return parts[1] == "05"


def stream_maf(maf_path: Path):
    """MC3 MAF 是 gzipped TSV,首行是注释(以#开头),首列依次为 MAF 标准列"""
    opener = gzip.open if str(maf_path).endswith(".gz") else open
    with opener(maf_path, "rt", encoding="utf-8", errors="replace") as f:
        # 跳过注释
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            header = line.rstrip("\n").split("\t")
            break
        if header is None:
            raise RuntimeError(f"MAF {maf_path} has no header after comment lines")
        col_idx = {name: i for i, name in enumerate(header)}
        required = ["Hugo_Symbol", "Variant_Classification", "FILTER", "Tumor_Sample_Barcode"]
        for r in required:
            if r not in col_idx:
                raise RuntimeError(f"MAF missing required col {r}; got {header[:10]}...")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            yield {
                "Hugo_Symbol": cols[col_idx["Hugo_Symbol"]],
                "Variant_Classification": cols[col_idx["Variant_Classification"]],
                "FILTER": cols[col_idx["FILTER"]] if "FILTER" in col_idx else "PASS",
                "Tumor_Sample_Barcode": cols[col_idx["Tumor_Sample_Barcode"]],
            }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract 9-gene mutation panel from MC3 MAF (TCGA-LUAD)")
    ap.add_argument("--maf", required=True, help="MC3 MAF .maf[.gz] path")
    ap.add_argument("--out", default="/mnt/d/pan-caner/data/MC3/9gene_panel_LUAD.csv", help="output CSV path")
    ap.add_argument("--genes", help="comma-separated gene list to use (default: 11-gene panel)")
    ap.add_argument(
        "--luad-whitelist",
        default="/mnt/d/pan-caner/data/MC3/tcga_luad_cases.json",
        help="JSON file with {'case_ids':[...]} or {'sample_ids':[...]} for TCGA-LUAD cases "
             "(from cBioPortal /api/studies/luad_tcga/samples). Pass empty string '' to disable.",
    )
    args = ap.parse_args()

    genes = [g.strip() for g in args.genes.split(",")] if args.genes else DEFAULT_GENES
    maf = Path(args.maf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 加载 LUAD case_id 白名单
    whitelist: set[str] | None = None
    if args.luad_whitelist:
        wl_path = Path(args.luad_whitelist)
        if wl_path.is_file():
            with open(wl_path) as f:
                wl = json.load(f)
            case_ids = wl.get("case_ids", []) if isinstance(wl, dict) else wl
            sample_ids = wl.get("sample_ids", [])
            whitelist = set(case_ids) | set(sample_ids)
            print(f"[whitelist] {len(whitelist):,} TCGA-LUAD case/sample ids from {wl_path}")
        else:
            print(f"[warn] whitelist file {wl_path} not found, falling back to site-05 only")

    # case_id -> {gene -> True (阳性)}
    matrix: dict[str, dict[str, bool]] = defaultdict(lambda: {g: False for g in genes})
    n_rows = 0
    n_luad = 0
    n_pass = 0
    n_kept = 0
    gene_counter = defaultdict(int)

    print(f"[read] {maf}")
    for row in stream_maf(maf):
        n_rows += 1
        if not is_luad(row["Tumor_Sample_Barcode"], whitelist):
            continue
        n_luad += 1
        if not is_pass_filter(row["FILTER"]):
            continue
        n_pass += 1
        if row["Hugo_Symbol"] not in genes:
            continue
        if row["Variant_Classification"] not in POSITIVE_CLASSES:
            continue
        case_id = extract_case_id(row["Tumor_Sample_Barcode"])
        matrix[case_id][row["Hugo_Symbol"]] = True
        gene_counter[row["Hugo_Symbol"]] += 1
        n_kept += 1
        if n_rows % 200_000 == 0:
            print(f"  ... read {n_rows/1e6:.1f}M rows, {n_kept:,} kept")

    print(f"\n[stats]")
    print(f"  total MAF rows          = {n_rows:,}")
    print(f"  rows in TCGA-LUAD       = {n_luad:,}")
    print(f"  rows PASS               = {n_pass:,}")
    print(f"  kept (9-gene POSITIVE)  = {n_kept:,}")
    print(f"  unique TCGA-LUAD cases  = {len(matrix):,}")

    # 写 CSV
    header = ["case_id"] + genes + ["any_9gene_positive", "n_genes_mutated"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        # 按 case_id 排序
        for cid in sorted(matrix):
            row = matrix[cid]
            flags = [int(row[g]) for g in genes]
            n_pos = sum(flags)
            w.writerow([cid] + flags + [int(n_pos > 0), n_pos])

    print(f"\n[wrote] {out}")

    # 控制台汇总(每基因阳性病例数)
    print(f"\n[per-gene positive case counts]")
    print(f"  {'gene':<8} {'pos_cases':>10} {'prev(%)':>9}")
    pos_per_gene = {g: sum(1 for cid in matrix if matrix[cid].get(g)) for g in genes}
    total_cases = len(matrix)
    for g in genes:
        c = pos_per_gene[g]
        pct = 100 * c / max(1, total_cases)
        print(f"  {g:<8} {c:>10,} {pct:>8.2f}%")

    print(f"\n[donwload (Tip)] Now you can train on these {total_cases:,} LUAD cases:")
    print(f"   python tools/build_tcga_dataset.py --label-csv {out} --out /mnt/d/pan-caner/data/TCGA-LUAD-9gene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
