# -*- coding: utf-8 -*-
"""62_label_gap.py - how many slides actually lack labels"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
labeled = set(man[man.has_label == 1].case_id)
proj = dict(zip(man.case_id, man.project))

srcs = {
    "gigap": os.path.join(C, "NSCLC_gigap_sub512_index.csv"),
    "uni": os.path.join(C, "NSCLC_uni_sub512_index.csv"),
    "chief": os.path.join(C, "NSCLC_chief_sub512_index.csv"),
    "uni2h_luad": os.path.join(C, "NSCLC_uni2h_luad_sub512_index.csv"),
}
allids = {}
for k, f in srcs.items():
    if os.path.exists(f):
        allids[k] = list(pd.read_csv(f).case_id)

print("=== slide x label coverage ===")
for k, ids in allids.items():
    c = pd.Series([proj.get(i, "NOT_IN_MANIFEST") for i in ids]).value_counts().to_dict()
    lab = sum(1 for i in ids if i in labeled)
    unl = [i for i in ids if i not in labeled]
    uu = pd.Series([proj.get(i, "NOT_IN_MANIFEST") for i in unl]).value_counts().to_dict()
    print("  %-10s total=%-5d labeled=%-5d unlabeled=%-5d" % (k, len(ids), lab, len(unl)))
    print("      cohort split      : %s" % c)
    print("      UNLABELED cohort  : %s" % uu)
    if unl:
        print("      unlabeled sample  : %s" % unl[:5])

g = allids["gigap"]
print("\n=== NSCL C detail (gigap cache) ===")
for p in ["LUAD", "LUSC"]:
    ids = [i for i in g if proj.get(i) == p]
    lab = [i for i in ids if i in labeled]
    print("  %s: slides=%-5d labeled=%-5d (%.1f%%)" % (p, len(ids), len(lab), 100.0 * len(lab) / max(1, len(ids))))
nf = [i for i in g if proj.get(i) == "NOT_IN_MANIFEST"]
print("  NOT_IN_MANIFEST     : slides=%-5d" % len(nf))
print("      sample           : %s" % nf[:5])
lab_all = [i for i in man[man.has_label == 1].case_id]
print("\n=== 反向: 有标签但没 gigap 特征 ===")
miss = [i for i in lab_all if proj.get(i) in ("LUAD", "LUSC") and i not in set(g)]
print("  LUAD+LUSC 标签总数=%d, 其中缺 gigap 特征=%d" % (len([i for i in lab_all if proj.get(i) in ("LUAD", "LUSC")]), len(miss)))
print("      sample           : %s" % miss[:5])
pd.Series(nf).to_csv(os.path.join(C, "nsclc_slides_not_in_manifest.csv"), index=False, header=["case_id"])
print("\n[saved] nsclc_slides_not_in_manifest.csv (%d)" % len(nf))