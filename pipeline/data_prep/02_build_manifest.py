# -*- coding: utf-8 -*-
"""
02_build_manifest.py
把「三个特征来源」x「MC3 11基因标签」join 成一张 case 级 manifest。

输出: data/labels/case_manifest.csv
列:   case_id, project, <11 genes>, any_positive, n_genes,
      has_slide_titan, has_slide_gigapath, has_patch_cpath, n_slide_*
癌种判定优先级: Prov-GigaPath 官方 cancer type abbreviation > TCGA TSS 集合
"""
import os, re, glob, pickle, collections
import pandas as pd, numpy as np

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
GENES = ["EGFR","KRAS","ALK","ROS1","TP53","BRAF","PIK3CA","ERBB2","NRAS","RET","MET"]
tnorm = lambda s: (lambda m: f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(
    re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(s)))

# ---------- 1. 标签 ----------
lab = pd.read_csv(os.path.join(R,"data","labels","tcga_9gene_panel_all.csv"))
lab["case_id"] = lab["case_id"].map(tnorm)
print(f"[1] MC3 标签 case = {len(lab)}")

# ---------- 2. 特征来源 ----------
src = collections.defaultdict(set)   # case -> {source}

tk = pickle.load(open(os.path.join(R,"models","TITAN","TCGA_TITAN_features.pkl"),"rb"))
for f in tk["filenames"]:
    c = tnorm(f)
    if c: src[c].add("slide_titan")
print(f"[2] slide_titan  slide={len(tk['filenames'])}")

gp = pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"] = gp["_PATIENT"].map(tnorm)
gp["slide"] = gp["filename"].map(tnorm)
proj = {}
for c, a in zip(gp["case"], gp["cancer type abbreviation"]):
    if c: proj[c] = a
for c in gp["case"].dropna().unique(): src[c].add("slide_gigapath")
print(f"[2] slide_gigapath  row={len(gp)}  LUAD={sum(1 for v in proj.values() if v=='LUAD')}  LUSC={sum(1 for v in proj.values() if v=='LUSC')}")

for p in glob.glob(os.path.join(R,"data","public_features","CPathPatchFeature","nsclc","uni","pt_files","*.pt")):
    b = os.path.basename(p); c = "-".join(b.split("-")[:3])
    if c.startswith("TCGA"): src[c].add("patch_cpath")
print(f"[2] patch_cpath  file={len(glob.glob(os.path.join(R,'data','public_features','CPathPatchFeature','nsclc','uni','pt_files','*.pt')))}")

for p in glob.glob(os.path.join(R,"feat_416","*.npz")):
    c = os.path.basename(p).replace(".npz","")
    if c.startswith("TCGA"): src[c].add("patch_local416")
print(f"[2] patch_local416  file={len(glob.glob(os.path.join(R,'feat_416','*.npz')))}")

# ---------- 3. 癌种 fallback (TCGA TSS) ----------
LUAD_TSS = {"05","33","35","38","44","49","50","53","55","56","62","64","66","67","69",
            "71","73","75","77","78","80","83","85","86","91","4B","L9","MP","50"}
LUSC_TSS = {"18","21","22","94","95","96","97","98","99","MN","NJ","J2","O1"}
L421 = set(pd.read_csv(os.path.join(R,"data","MC3","9gene_panel_LUAD.csv"))["case_id"])
for c in L421: proj.setdefault(c, "LUAD")
# cpath_patch 只含 TCGA-NSCLC,故有该来源且非 LUAD 者判为 LUSC
for c, f in src.items():
    if "patch_cpath" in f and proj.get(c) not in ("LUAD","LUSC"): proj[c] = "LUSC"

def classify(c):
    if c in proj: return proj[c]
    t = c.split("-")[1] if len(c.split("-"))>1 else ""
    if c in L421: return "LUAD"
    if t in LUSC_TSS: return "LUSC"
    if t in LUAD_TSS: return "LUAD"
    return "unknown"

# ---------- 4. 组装 ----------
rows = []
for c, feats in src.items():
    if not c: continue
    r = {"case_id": c, "project": classify(c)}
    for g in GENES: r[g] = np.nan
    r["has_slide_titan"]   = int("slide_titan" in feats)
    r["has_slide_gigapath"]= int("slide_gigapath" in feats)
    r["has_patch_cpath"]   = int("patch_cpath" in feats)
    r["has_patch_local416"]= int("patch_local416" in feats)
    r["n_feat_src"] = len(feats)
    rows.append(r)
man = pd.DataFrame(rows)

m = lab.set_index("case_id")
man = man.set_index("case_id")
common = man.index.intersection(m.index)
man.loc[common, GENES] = m.loc[common, GENES].values
man["any_positive"] = man[GENES].sum(axis=1, min_count=1)
man["n_genes"] = man[GENES].sum(axis=1, min_count=1)
man["has_label"] = man[GENES].notna().any(axis=1).astype(int)

out = os.path.join(R,"data","labels","case_manifest.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
man.reset_index().sort_values(["project","case_id"]).to_csv(out, index=False)
print(f"\n[3] manifest case = {len(man)}  有标签 = {int(man['has_label'].sum())}  -> {out}")
print(f"\n[4] 癌种分布: {dict(man['project'].value_counts())}")

# ---------- 5. 覆盖矩阵 + 逐基因 ----------
print("\n=== 特征来源覆盖 (按癌种) ===")
for p in ["LUAD","LUSC","unknown"]:
    s = man[man["project"]==p]
    if not len(s): continue
    print(f"  {p:8s} case={len(s):5d}  titan={int(s.has_slide_titan.sum()):5d}  gigapath={int(s.has_slide_gigapath.sum()):5d}  cpath_patch={int(s.has_patch_cpath.sum()):5d}  local416={int(s.has_patch_local416.sum()):5d}")

print("\n=== 有标签的 NSCLC 逐基因阳性数 ===")
for grp, sel in [("LUAD only", man.project=="LUAD"),
                 ("LUSC only", man.project=="LUSC"),
                 ("NSCLC (LUAD+LUSC)", man.project.isin(["LUAD","LUSC"]))]:
    s = man[sel & (man.has_label==1)]
    if not len(s): continue
    print(f"\n  {grp}  n={len(s)}")
    for g in GENES:
        c = int(s[g].sum())
        flag = "" if c>=30 else ("  <-- 稀缺" if c>=10 else "  <-- 不可用")
        print(f"    {g:<7}{c:>6}  ({100*c/len(s):5.1f}%){flag}")

