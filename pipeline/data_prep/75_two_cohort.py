import sys, os, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
GEN = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]

uni = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
cpt = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
lab = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
print("=== 同编码器(UNI v1) 的两个 LUAD 队列 ===", flush=True)
print("TCGA uni cache : %d 例  (列=%s)" % (len(uni), [c for c in uni.columns][:5]), flush=True)
print("CPTAC uni cache: %d 例  (列=%s)" % (len(cpt), [c for c in cpt.columns][:5]), flush=True)
print("cptac labels  : %d 例  (列=%s)" % (len(lab), list(lab.columns)[:10]), flush=True)
print("\ncptac labels 里 disease 分布: %s" % dict(lab.disease.value_counts()) if "disease" in lab.columns else "no disease col", flush=True)
print("cptac id 前缀: %s" % dict(pd.Series([str(x)[:3] for x in lab.submitter_id]).value_counts()), flush=True)
if "n_maf" in lab.columns:
    print("n_maf 分布: %s" % dict(pd.cut(lab.n_maf, [-1,0,1,5,20,100,10000]).value_counts().sort_index()), flush=True)
    print("  n_maf==0 (无突变记录) 的例数 = %d" % int((lab.n_maf == 0).sum()), flush=True)

cp = set(cpt.case_id) & set(lab.submitter_id)
lu = lab[lab.submitter_id.str.startswith("C3L")]
cpl = set(cpt.case_id) & set(lu.submitter_id)
print("\nCPTAC-LUAD(C3L) 标签: %d ; 与 UNI 特征交集: %d" % (len(lu), len(cpl)), flush=True)
sub = lu[lu.submitter_id.isin(cp)].copy()
sub = sub[["submitter_id"] + GEN].rename(columns={"submitter_id": "case_id"})
print("可用 CPTAC-LUAD 训练例数 = %d" % len(sub), flush=True)
print("  阳性数: %s" % {g: int(sub[g].sum()) for g in GEN}, flush=True)
if "n_maf" in lu.columns:
    print("  这 %d 例里 n_maf>0 的比例: %.1f%%" % (len(sub), 100.0*(lu[lu.submitter_id.isin(cp)].n_maf > 0).mean()), flush=True)
sub.to_csv(os.path.join(C, "cptac_luad_usable.csv"), index=False)

print("\n=== 与 TCGA 对比 ===", flush=True)
man = pd.read_csv(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\labels\case_manifest.csv")
labu = man[(man.project=="LUAD") & (man.has_label==1)].set_index("case_id")
tlu = sorted(set(uni.case_id) & set(labu.index))
print("TCGA-LUAD (UNI v1): %d 例, 阳性 %s" % (len(tlu), {g: int(labu.loc[tlu, g].sum()) for g in GEN}), flush=True)
print("CPTAC-LUAD (UNI v1): %d 例, 阳性 %s" % (len(sub), {g: int(sub[g].sum()) for g in GEN}), flush=True)
print("\n合并后 = %d 例 (比单 TCGA 多 %.0f%%)" % (len(tlu)+len(sub), 100.0*len(sub)/len(tlu)), flush=True)