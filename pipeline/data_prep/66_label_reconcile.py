# -*- coding: utf-8 -*-
"""66_label_reconcile.py - per-gene reconciliation vs cBioPortal PanCanAtlas,
then emit a corrected label file + list every case whose labels actually change."""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, numpy as np, requests

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
STUDIES = {"LUAD": "luad_tcga_pan_can_atlas_2018", "LUSC": "lusc_tcga_pan_can_atlas_2018"}
CB = "https://www.cbioportal.org/api"
S = requests.Session(); S.trust_env = False; S.headers.update({"Accept": "application/json"})

def get(u, **kw):
    for a in range(4):
        try:
            r = S.get(u, timeout=60, **kw); r.raise_for_status(); return r.json()
        except Exception:
            if a == 3: raise
            time.sleep(1.5 * (a + 1))

EGID = {g: get(CB + "/genes/" + g)["entrezGeneId"] for g in GENES}

rows = {}
for proj, sid in STUDIES.items():
    prof, lst = sid + "_mutations", sid + "_all"
    samples = get(CB + "/sample-lists/" + lst)["sampleIds"]
    for s in samples:
        rows.setdefault(s[:12], {g_: 0 for g_ in GENES})
    rowsx = {s[:12]: rows[s[:12]] for s in samples}
    for g in GENES:
        ms = get(CB + "/molecular-profiles/%s/mutations" % prof,
                 params={"sampleListId": lst, "entrezGeneId": EGID[g],
                         "projection": "SUMMARY", "pageSize": 2000000})
        for m in ms:
            cid = m["sampleId"][:12]
            if cid in rowsx:
                rowsx[cid][g] = 1
    for cid in rowsx:
        rowsx[cid]["project"] = proj
    print("  %s done (%d sample)" % (proj, len(samples)), flush=True)

cb = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "case_id"})
cb["case_id"] = cb.case_id.astype(str)
cb[GENES] = cb[GENES].astype(int)
cb.to_csv(os.path.join(C, "cba_luad_lusc_6gene_labels.csv"), index=False)
print("[saved] cba_luad_lusc_6gene_labels.csv  %d case" % len(cb), flush=True)
print("  阳性率: %s" % {g: "%.1f%%" % (100 * cb[g].mean()) for g in GENES}, flush=True)

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1][["case_id", "project"] + GENES].copy()
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
feat = set(idx.case_id)

m = lab.merge(cb, on=["case_id", "project"], how="left", suffixes=("_ours", "_cb"))
inb = m[m.case_id.isin(feat)].copy()
print("\n=== 对账范围 ===", flush=True)
print("  本地有标签        : %d" % len(m), flush=True)
print("  其中有 gigap 特征 : %d" % len(inb), flush=True)
print("  其中 cBioPortal 也能查到且有特征: %d" % int(inb[[g + "_cb" for g in GENES]].notna().all(axis=1).sum()), flush=True)

print("\n=== 逐基因混淆 (只看同时有特征+两边都有标签的 LUAD/LUSC) ===", flush=True)
sub = inb[(inb.project == "LUAD") | (inb.project == "LUSC")]
sub = sub[sub[[g + "_cb" for g in GENES]].notna().all(axis=1)]
tot = {"TP": 0, "TN": 0, "FN": 0, "FP": 0}
for g in GENES:
    a, b = sub[g + "_ours"].astype(int), sub[g + "_cb"].astype(int)
    tp = int(((a == 1) & (b == 1)).sum()); tn = int(((a == 0) & (b == 0)).sum())
    fn = int(((a == 0) & (b == 1)).sum()); fp = int(((a == 1) & (b == 0)).sum())
    tot["TP"] += tp; tot["TN"] += tn; tot["FN"] += fn; tot["FP"] += fp
    print("  %-6s  TP=%-4d TN=%-4d  我们漏标(+)=%d  我们多标(-)=%d   agree=%.2f%%"
          % (g, tp, tn, fn, fp, 100.0 * (tp + tn) / max(1, tp + tn + fn + fp)), flush=True)
print("  ALL    TP=%-4d TN=%-4d  我们漏标(+)=%d  我们多标(-)=%d   agree=%.2f%%"
      % (tot["TP"], tot["TN"], tot["FN"], tot["FP"],
         100.0 * (tot["TP"] + tot["TN"]) / (tot["TP"] + tot["TN"] + tot["FN"] + tot["FP"])), flush=True)

diff = sub[((sub[[g + "_ours" for g in GENES]].values) != (sub[[g + "_cb" for g in GENES]].values)).any(axis=1)]
print("\n=== 标签会变的 case (有特征且在训练集里): %d ===" % len(diff), flush=True)
for _, r in diff.iterrows():
    ch = []
    for g in GENES:
        if int(r[g + "_ours"]) != int(r[g + "_cb"]):
            ch.append("%s %d->%d" % (g, int(r[g + "_ours"]), int(r[g + "_cb"])))
    print("  %-14s %s  %s" % (r.case_id, r.project, "; ".join(ch)), flush=True)

print("\n=== cBioPortal 有标签+特征, 但本地完全没标签 ===", flush=True)
new = cb[cb.case_id.isin(feat)].merge(lab[["case_id"]].assign(have=1), on="case_id", how="left")
new = new[new.have.isna()]
print("  %d 例: %s" % (len(new), new.case_id.tolist()[:30]), flush=True)
if len(new):
    print("  其中阳性分布: %s" % {g: int(new[g].sum()) for g in GENES}, flush=True)
    print("  project: %s" % dict(new.project.value_counts()), flush=True)

new.to_csv(os.path.join(C, "labels_newly_available_from_cba.csv"), index=False)
print("\n[saved] labels_newly_available_from_cba.csv (%d)" % len(new), flush=True)
print("DONE", flush=True)