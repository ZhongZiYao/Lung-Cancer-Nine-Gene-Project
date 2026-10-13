# -*- coding: utf-8 -*-
"""08_update_manifest_genes.py — 用 31 基因 survey 表刷新 case_manifest 的基因列"""
import os, numpy as np, pandas as pd
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
L = os.path.join(R,"data","labels")
sv = pd.read_csv(os.path.join(L,"tcga_gene_survey_all.csv"))
man = pd.read_csv(os.path.join(L,"case_manifest.csv"))
old = [c for c in man.columns if c not in ("case_id","project","any_positive","n_genes","n_feat_src",
            "has_slide_titan","has_slide_gigapath","has_patch_cpath","has_patch_local416","has_label")]
man = man.drop(columns=[c for c in old if c in man.columns])
GENES = [c for c in sv.columns if c != "case_id"]
man = man.merge(sv, on="case_id", how="left")
man["n_genes"]  = man[GENES].sum(axis=1, min_count=1)
man["any_positive"] = (man["n_genes"].fillna(0) > 0).astype(int)
man["has_label"] = man[GENES].notna().any(axis=1).astype(int)
man.to_csv(os.path.join(L,"case_manifest.csv"), index=False)
print(f"[wrote] case_manifest.csv  case={len(man)}  有标签={int(man.has_label.sum())}  基因列={len(GENES)}")

# 可训练队列(有 patch 特征 + 有标签)
tr = man[man.has_label==1 & man.has_patch_cpath.eq(1) & man.project.isin(["LUAD","LUSC"])]
print(f"\n[可训练队列] NSCLC={len(tr)}  LUAD={int((tr.project=='LUAD').sum())}  LUSC={int((tr.project=='LUSC').sum())}")
print(f"\n{'gene':<9}{'NSCLC+':>7}{'LUAD+':>7}{'LUSC+':>7}  入选")
PICK = ["KRAS","TP53","EGFR","STK11","KEAP1","NF1","BRAF"]
for g in GENES:
    if g not in PICK: continue
    y=tr[g].fillna(0).astype(int); lu=tr.project=="LUAD"
    print(f"{g:<9}{int(y.sum()):>7}{int(y[lu].sum()):>7}{int(y[~lu].sum()):>7}  "
          f"{'OK(>=50 both)' if y[lu].sum()>=50 and y[~lu].sum()>=50 else ('LUAD-ok' if y[lu].sum()>=30 else 'skip')}")
