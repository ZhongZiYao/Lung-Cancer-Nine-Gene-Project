# -*- coding: utf-8 -*-
"""01b_gene_panel_survey.py — 在 MC3 全量上统计 30+ 肺癌相关基因的阳性数,用于数据驱动地选基因"""
import gzip, os, sys
from collections import defaultdict
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
MAF = os.path.join(R,"data","MC3","mc3.v0.2.8.PUBLIC.maf.gz")
OUT = os.path.join(R,"data","labels","tcga_gene_survey_all.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 候选池: 原 9/11 基因 + LUAD/LUSC 真实驱动基因 + 高频伴随突变
GENES = ["EGFR","KRAS","ALK","ROS1","TP53","BRAF","PIK3CA","ERBB2","NRAS","RET","MET",
         "STK11","NF1","KEAP1","AKT1","SMARCA4","RB1","PTEN","ATM","NFE2L2","SETBP1",
         "FAT1","CDKN2A","FGFR1","MAP3K1","RBM10","ACSS2","SMARCB1","KMT2D","ARID1A","CTNNB1"]
POS = {"Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
       "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation"}
case_of = lambda b: "-".join(b.split("-")[:3]) if len(b.split("-"))>=3 and b.startswith("TCGA") else None

mat = defaultdict(lambda: {g:0 for g in GENES})
n=0
with gzip.open(MAF,"rt",encoding="utf-8",errors="replace") as f:
    hdr=None
    for line in f:
        if line.startswith("#"): continue
        hdr=line.rstrip("\n").split("\t"); break
    ci={c:i for i,c in enumerate(hdr)}
    for line in f:
        n+=1
        c=line.rstrip("\n").split("\t")
        cid=case_of(c[ci["Tumor_Sample_Barcode"]])
        if cid is None: continue
        if cid not in mat: mat[cid]={g:0 for g in GENES}
        g=c[ci["Hugo_Symbol"]]
        if g not in GENES: continue
        if c[ci["Variant_Classification"]] not in POS: continue
        if c[ci["FILTER"]].strip().upper() not in {"PASS","","."}: continue
        mat[cid][g]=1

N=len(mat)
with open(OUT,"w",encoding="utf-8") as f:
    f.write("case_id,"+",".join(GENES)+"\n")
    for cid in sorted(mat):
        f.write(cid+","+",".join(str(mat[cid][g]) for g in GENES)+"\n")
print(f"[done] rows={n:,} cases={N:,} -> {OUT}")
print(f"\n{'gene':<9}{'pos':>7}{'prev%':>8}   5-fold 60/15/15 可行性")
for g in sorted(GENES, key=lambda x:-sum(mat[k][x] for k in mat)):
    c=sum(mat[k][g] for k in mat)
    p=100*c/N
    ok = "OK" if c>=50 else ("marginal" if c>=30 else "skip")
    print(f"{g:<9}{c:>7}{p:>7.2f}%   {ok}")
