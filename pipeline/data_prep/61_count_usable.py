# -*- coding: utf-8 -*-
"""61_count_usable.py  """
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
print("[manifest] %d , %d case, project=%s" % (len(man), man.case_id.nunique(),
      dict(man.project.value_counts())), flush=True)
print("[] case_id/project/has_label + : %s" % [c for c in man.columns if c not in ("case_id","project","has_label")], flush=True)

print("\n===  ===")
for proj in ["LUAD", "LUSC"]:
    d = man[man.project == proj]
    print("\n-- %s: %d , has_label==1  %d" % (proj, len(d), int((d.has_label == 1).sum())), flush=True)
    h = d[d.has_label == 1]
    print("   6 : %s" % {g: int(h[g].notna().sum()) for g in GENES}, flush=True)
    complete = h.dropna(subset=GENES)
    print("   : %d " % len(complete), flush=True)
    print("   (complete): %s" % {g: {int(k): int(v) for k, v in complete[g].value_counts().items()} for g in GENES}, flush=True)

lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
comp = lab.dropna(subset=GENES)
print("\n===  55  ===")
print("55/59/60  (LUAD & has_label==1 & dropna[TP53]): %d " % len(lab), flush=True)
print(" 6 : %d " % len(comp), flush=True)
print("   : %s" % {g: int(lab[g].isna().sum()) for g in GENES}, flush=True)
bad = lab[lab[GENES].isna().any(axis=1)]
if len(bad):
    print("    %d, : %s" % (len(bad),
          dict(bad[GENES].isna().sum(axis=1).value_counts().sort_index())), flush=True)
print("   : %s" % {g: int(comp[g].sum()) for g in GENES}, flush=True)

print("\n=== "" ===")
idxf = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
gset = set(idxf.case_id)
print("gigap  case: %d" % len(gset), flush=True)
for tag, s in [("LUAD 6 ", set(comp.index)), ("LUAD dropna[TP53] ()", set(lab.index))]:
    print("   %-28s  gigap : %d" % (tag, len(s & gset)), flush=True)
for enc in ["uni", "chief", "uni2h_luad"]:
    f = os.path.join(C, "NSCLC_%s_sub512_index.csv" % enc)
    if os.path.exists(f):
        s = set(pd.read_csv(f).case_id)
        print("   %-28s  %s : %d" % ("LUAD 6 ", enc, len(set(comp.index) & s)), flush=True)
        print("   %-28s  %s : %d" % ("LUAD dropna[TP53] ()", enc, len(set(lab.index) & s)), flush=True)

print("\n=== CPTAC  ===")
cp = pd.read_csv(os.path.join(C, "cptac_labels_rebuilt.csv"))
print("cptac_labels_rebuilt.csv: %d " % len(cp), flush=True)
if "project" in cp.columns:
    print("   project: %s" % dict(cp.project.value_counts()), flush=True)
    c6 = cp.dropna(subset=GENES)
    print("   6 : %d" % len(c6), flush=True)
    print("   : %s" % {g: int(c6[g].sum()) for g in GENES if g in c6.columns}, flush=True)