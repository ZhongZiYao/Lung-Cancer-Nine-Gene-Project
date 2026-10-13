# -*- coding: utf-8 -*-
"""
01_build_tcga_labels.py
从 MC3 MAF 构建「全 TCGA case」x 11 基因 突变标签表。

设计要点:
  - 不在此脚本里筛癌种。癌种/队列过滤放到 join 阶段 (02/03),
    这样新增特征来源时不需要重跑 MAF 解析 (MC3 有 360 万行)。
  - 判定规则沿用 extract_9gene_panel.py,保证与历史 421 例口径一致:
      FILTER in {PASS, "", "."}  且  Variant_Classification in POSITIVE_CLASSES
输出: data/labels/tcga_9gene_panel_all.csv
"""
import gzip, csv, sys, os
from collections import defaultdict

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
MAF = os.path.join(R, "data", "MC3", "mc3.v0.2.8.PUBLIC.maf.gz")
OUT = os.path.join(R, "data", "labels", "tcga_9gene_panel_all.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

GENES = ["EGFR","KRAS","ALK","ROS1","TP53","BRAF","PIK3CA","ERBB2","NRAS","RET","MET"]
POSITIVE_CLASSES = {
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
    "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
}

def case_of(barcode):
    p = (barcode or "").split("-")
    return "-".join(p[:3]) if len(p) >= 3 and p[0] == "TCGA" else None

def main():
    matrix = defaultdict(lambda: {g: 0 for g in GENES})
    sample_of_case = defaultdict(set)
    n_rows = n_keep = 0
    with gzip.open(MAF, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            if line.startswith("#"): continue
            header = line.rstrip("\n").split("\t"); break
        ci = {c: i for i, c in enumerate(header)}
        need = ["Hugo_Symbol","Variant_Classification","FILTER","Tumor_Sample_Barcode"]
        for r in need:
            assert r in ci, f"missing {r}"
        for line in f:
            n_rows += 1
            cols = line.rstrip("\n").split("\t")
            if len(cols) < len(header): continue
            bc = cols[ci["Tumor_Sample_Barcode"]]
            cid0 = case_of(bc)
            if not cid0: continue
            if cid0 not in matrix:          # 登记真阴性样本
                matrix[cid0] = {g: 0 for g in GENES}
                sample_of_case[cid0].add(bc)
            if cols[ci["Hugo_Symbol"]] not in GENES: continue
            if cols[ci["Variant_Classification"]] not in POSITIVE_CLASSES: continue
            flt = cols[ci["FILTER"]].strip().upper()
            if flt not in {"PASS","","."}: continue
            bc = cols[ci["Tumor_Sample_Barcode"]]
            cid = case_of(bc)
            if not cid: continue
            matrix[cid][cols[ci["Hugo_Symbol"]]] = 1
            sample_of_case[cid].add(bc)
            n_keep += 1
            if n_rows % 1_000_000 == 0:
                print(f"  ...{n_rows/1e6:.1f}M rows, kept {n_keep:,}, cases {len(matrix):,}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] + GENES + ["any_positive","n_genes_mutated","n_samples"])
        for cid in sorted(matrix):
            row = matrix[cid]; flags = [row[g] for g in GENES]
            w.writerow([cid] + flags + [int(sum(flags) > 0), sum(flags), len(sample_of_case[cid])])

    N = len(matrix)
    print(f"\n[done] rows={n_rows:,} kept={n_keep:,} cases={N:,}")
    print(f"[wrote] {OUT}")
    print(f"\n{'gene':<8}{'pos_cases':>10}{'prev%':>9}")
    for g in GENES:
        c = sum(matrix[k][g] for k in matrix)
        print(f"{g:<8}{c:>10,}{100*c/N:>8.2f}%")

if __name__ == "__main__":
    main()

