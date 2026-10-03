# -*- coding: utf-8 -*-
"""09_build_cptac_labels.py
桥接: 本地 MAF 文件名 == manifest.file_id -> case_id(UUID) -> submitter_id(C3L/C3N) -> 切片特征
输出: data/labels/cptac_gene_labels.csv  (按 submitter_id 索引)
"""
import os, re, gzip, json, glob, csv
from collections import defaultdict
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R,"data","CPTAC"); MD = os.path.join(C,"cptac_maf")
OUT = os.path.join(R,"data","labels","cptac_gene_labels.csv")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR","BRAF","PIK3CA","ROS1","ALK"]
POS = {"Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
       "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation"}
UU = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)

man = json.load(open(os.path.join(C,"cptac3_maf_manifest.json"), encoding="utf-8"))
fid2cases = {m["file_id"].lower(): m.get("case_ids", []) for m in man}
cm = {k.lower(): v for k, v in json.load(open(os.path.join(C,"cptac_case_mapping.json"), encoding="utf-8")).items()}

mat = defaultdict(lambda: {g: 0 for g in GENES})
nfiles = nlinked = 0
for p in sorted(glob.glob(os.path.join(MD,"*.maf.gz"))):
    fid = UU.search(os.path.basename(p)).group(1).lower()
    case_uuids = fid2cases.get(fid)
    if not case_uuids: continue
    subs = {cm[u.lower()]["submitter_id"] for u in case_uuids
            if u.lower() in cm and cm[u.lower()].get("submitter_id")}
    if not subs: continue
    nlinked += 1
    nfiles += 1
    with gzip.open(p,"rt",encoding="utf-8",errors="replace") as f:
        for line in f:
            if line.startswith("#"): continue
            hdr = line.rstrip("\n").split("\t"); break
        ci = {c:i for i,c in enumerate(hdr)}
        if "Hugo_Symbol" not in ci or "Variant_Classification" not in ci: continue
        has_flt = "FILTER" in ci   # CPTAC 的 aliquot-merged MAF 无 FILTER 列,视为全通过
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < len(hdr): continue
            if c[ci["Hugo_Symbol"]] not in GENES: continue
            if c[ci["Variant_Classification"]] not in POS: continue
            if has_flt and c[ci["FILTER"]].strip().upper() not in {"PASS","","."}: continue
            for s in subs: mat[s][c[ci["Hugo_Symbol"]]] = 1

with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["submitter_id","prefix","disease_type"]+GENES)
    for s in sorted(mat):
        pre = s.split("-")[0]
        dt = next((cm[u.lower()]["disease_type"] for u in fid2cases.get("",[]) if False), None)
        w.writerow([s, pre, ""] + [mat[s][g] for g in GENES])

# 与切片特征 join
feat = {os.path.basename(x).split("-")[0]+"-"+os.path.basename(x).split("-")[1]
        for x in glob.glob(os.path.join(R,"data","public_features","CPathPatchFeature","cptac_nsclc","uni","pt_files","*.pt"))}
print(f"[1] 本地 MAF {len(glob.glob(os.path.join(MD,'*.maf.gz')))} 个, 成功桥接到 submitter_id 的 {nlinked} 个")
print(f"[2] 解析出 {len(mat)} 个带基因标签的 CPTAC 病例")
print(f"[3] 切片特征 case = {len(feat)}")
c3l = {s for s in mat if s.startswith('C3L')}
print(f"    其中 C3L = {len(c3l)}   ∩ 切片特征 = {len(c3l & feat)}  <== 可直接外验的病例数")
print(f"\n[4] 外验队列逐基因阳性数 (C3L ∩ 特征):")
ev = sorted(c3l & feat)
for g in GENES:
    n = sum(mat[s][g] for s in ev)
    print(f"    {g:<8}{n:>5}  ({100*n/max(1,len(ev)):.1f}%)")
print(f"\n[wrote] {OUT}")

