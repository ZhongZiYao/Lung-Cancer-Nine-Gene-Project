# -*- coding: utf-8 -*-
"""65_cbioportal_verify.py - cross-check our labels against cBioPortal PanCanAtlas
and find lung cohorts on GDC that have BOTH slide images and open somatic MAF."""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, requests

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
OUT = os.path.join(C, "cbioportal_gene_matrix.csv")
STUDIES = {"LUAD": "luad_tcga_pan_can_atlas_2018", "LUSC": "lusc_tcga_pan_can_atlas_2018"}
CB = "https://www.cbioportal.org/api"
S = requests.Session(); S.trust_env = False
S.headers.update({"Accept": "application/json"})

def get(u, **kw):
    for a in range(4):
        try:
            r = S.get(u, timeout=40, **kw); r.raise_for_status(); return r.json()
        except Exception as e:
            if a == 3: raise
            time.sleep(1.5 * (a + 1))

print("=== 1. gene ids ===", flush=True)
EGID = {}
for g in GENES:
    j = get(CB + "/genes/" + g)
    EGID[g] = j.get("entrezGeneId")
    print("  %-6s -> %s" % (g, EGID[g]), flush=True)

mat = {}
for proj, sid in STUDIES.items():
    prof = sid + "_mutations"
    lst = sid + "_all"
    samples = get(CB + "/sample-lists/" + lst)["sampleIds"]
    print("\n=== 2. %s (%s): %d sample ===" % (proj, sid, len(samples)), flush=True)
    pos = set()
    for g in GENES:
        ms = get(CB + "/molecular-profiles/%s/mutations" % prof,
                 params={"sampleListId": lst, "entrezGeneId": EGID[g], "projection": "SUMMARY",
                         "pageSize": 2000000})
        hit = {m["sampleId"] for m in ms}
        pos |= hit
        print("   %-6s mutated cases = %d" % (g, len(hit)), flush=True)
    mat[proj] = pos

rows = {}
for proj, pos in mat.items():
    samples = get(CB + "/sample-lists/%s_all" % STUDIES[proj])["sampleIds"]
    for s in samples:
        cid = s[:12]
        if cid not in rows:
            rows[cid] = {"project": proj, "cbioportal_any6": 0}
        rows[cid]["cbioportal_any6"] = int(s in pos)
    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "case_id"})
df["case_id"] = df.case_id.astype(str)
df["project"] = df["project"].astype(str)
df["cbioportal_any6"] = df.cbioportal_any6.astype(int)
print("\n=== 3. cBioPortal 结果: %d case 有 6 基因任一突变 ===" % int(df.cbioportal_any6.sum()), flush=True)
df.to_csv(OUT, index=False)
print("  saved ->", OUT, flush=True)

print("\n=== 4. 与本地 manifest 对账 ===", flush=True)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1][["case_id", "project"] + GENES].copy()
m = lab.merge(df, on=["case_id", "project"], how="left")
m["cbioportal_any6"] = m.cbioportal_any6.fillna(-1).astype(int)
have = m[m.cbioportal_any6 >= 0]
print("  本地有标签: %d ; 其中 cBioPortal 也能查到: %d" % (len(m), len(have)), flush=True)
print("  本地全阴(WildType) 例子: %s" % m[(m.cbioportal_any6 == 0)][m[GENES].sum(axis=1) == 0].case_id.head(5).tolist(), flush=True)
dis = have[(have.cbioportal_any6 == 1) & (have[GENES].sum(axis=1) == 0)]
print("  ** 冲突: 本地全阴 但 cBioPortal 说有 6 基因突变 = %d 例" % len(dis), flush=True)
if len(dis): print(dis[["case_id", "project"]].head(20).to_string(), flush=True)

print("\n=== 5. 本地未标注的 21 张切片, cBioPortal 判它们是什么 ===", flush=True)
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
lset = set(man[man.has_label == 1].case_id)
unl = [i for i in idx.case_id if i not in lset]
u = df[df.case_id.isin(unl)][["case_id", "project", "cbioportal_any6"]]
print("  查到 %d / %d :" % (len(u), len(unl)), flush=True)
print(u.to_string(index=False), flush=True)

print("\n=== 6. GDC: 同时有诊断切片 + 开放 Masked Somatic Mutation 的项目 ===", flush=True)
GDC = "https://api.gdc.cancer.gov/projects"
flt = {"op": "and", "content": [
    {"op": "in", "content": {"field": "data_categories.data_category", "value": ["Diagnostic Slide Image"]}},
    {"op": "in", "content": {"field": "data_categories.data_type", "value": ["Masked Somatic Mutation"]}}]}
try:
    r = S.post(GDC, json={"filters": flt, "facets": "program.name", "size": 1000}, timeout=90)
    j = r.json()
    d = j.get("data", {})
    aggs = d.get("aggregations", {})
    proj = [b["key"] for b in aggs.get("project_id", {}).get("buckets", [])]
    print("  项目数 = %d" % d.get("pagination", {}).get("total"), flush=True)
    lung = [p for p in proj if any(k in p.upper() for k in
            ["LUAD", "LUSC", "LUNG", "NSCLC", "APOLLO", "CPTAC", "NLST", "HCMI", "MATCH", "CGCI"])]
    print("  肺癌相关: %s" % lung, flush=True)
    for b in sorted(aggs.get("project_id", {}).get("buckets", []), key=lambda x: -x["doc_count"])[:15]:
        print("    %-24s %d case" % (b["key"], b["doc_count"]), flush=True)
except Exception as e:
    print("  GDC ERR", str(e)[:120], flush=True)
print("DONE", flush=True)