# -*- coding: utf-8 -*-
"""69_alchem_build_index.py - build the ALCHEMIST-ALCH case index (labels + slide list)

ALCHEMIST-ALCH is the only non-TCGA lung cohort that has BOTH open-access diagnostic
slide images AND open-access masked somatic mutation MAFs, with full coverage of our
six genes. Verified via GDC API:
  - 1,349 SVS / 1,175 cases, access=open
  - 1,188 Masked Somatic Mutation files, access=open
  - >=1,147 cases have both (inclusion-exclusion lower bound)

This script produces the join table the extraction pipeline needs:
  case_id | 6 gene labels | slide file uuid/name/size | sample_type | histology
It does NOT download any SVS. Slides are ~1.5 GB each, so the download is streamed
one-at-a-time by 70_alchem_stream_extract.py.

GDC field notes (these cost me a wrong query, recording them so nobody repeats it):
  - slide images are files.data_type == "Slide Image"; their data_category is
    "Biospecimen", NOT "Diagnostic Slide Image" (that string is portal-UI only).
  - the project facet for file counts is project.project_id, not project_id.
  - gene symbol field on ssm_occurrences is ssm.consequence.transcript.gene.symbol
    and it is LOWERCASE.
"""
import gzip, io as _io, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, requests

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
AL = os.path.join(R, "data", "alchemist")
os.makedirs(AL, exist_ok=True)
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
PROJ = "ALCHEMIST-ALCH"
GDC = "https://api.gdc.cancer.gov"
S = requests.Session(); S.trust_env = False
S.headers.update({"Content-Type": "application/json"})


def post(path, payload):
    for a in range(4):
        try:
            r = S.post(GDC + path, json=payload, timeout=120); r.raise_for_status(); return r.json()
        except Exception as e:
            if a == 3: raise
            print("   retry %d (%s)" % (a + 1, str(e)[:60]), flush=True); time.sleep(2 * (a + 1))


def fetch(path, params):
    # GDC GET needs filters JSON-encoded; requests would otherwise flatten the dict
    params = dict(params)
    if isinstance(params.get("filters"), dict):
        params["filters"] = json.dumps(params["filters"])
    for a in range(4):
        try:
            r = S.get(GDC + path, params=params, timeout=120)
            if r.status_code != 200:
                raise RuntimeError("HTTP %s %s" % (r.status_code, r.text[:200]))
            return r.json()
        except Exception as e:
            if a == 3: raise
            time.sleep(2 * (a + 1))


# ---------------- 1. slide files ----------------
print("=== 1. slide images ===", flush=True)
fl = {"op": "and", "content": [
      {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJ]}},
      {"op": "in", "content": {"field": "files.data_type", "value": ["Slide Image"]}},
      {"op": "in", "content": {"field": "files.access", "value": ["open"]}}]}
slides, frm, size = [], 0, 400
while True:
    j = fetch("/files", {"filters": fl, "size": size, "from": frm,
                         "fields": "file_id,file_name,file_size",
                         "expand": "cases", "format": "JSON"})
    h = j.get("data", {}).get("hits", [])
    slides += h; frm += size
    print("   got %d (total %s)" % (len(slides), j.get("data", {}).get("pagination", {}).get("total")), flush=True)
    if len(h) < size: break
print("  slide files = %d" % len(slides), flush=True)
sdf = pd.DataFrame(slides)
sdf["file_id"] = sdf.file_id.str.split("/").str[-1]
def _cid(x):
    if isinstance(x, list) and x:
        return x[0].get("submitter_id") if isinstance(x[0], dict) else x[0]
    if isinstance(x, dict):
        return x.get("submitter_id")
    return x
sdf["case_id"] = sdf["cases"].apply(_cid)
sdf = sdf[["file_id", "file_name", "file_size", "case_id"]]
assert sdf.case_id.notna().all(), "some slides have no case_id"
print("  样例:", sdf.head(2).to_dict("records"), flush=True)
print("  case 数 = %d ; 总大小 %.0f GB" % (sdf.case_id.nunique(), sdf.file_size.sum() / 1e9), flush=True)

# ---------------- 2. case metadata: sample_type + histology ----------------
print("\n=== 2. case metadata ===", flush=True)
fl2 = {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJ]}}
cases, frm = [], 0
while True:
    j = fetch("/cases", {"filters": fl2, "size": size, "from": frm,
                         "fields": "submitter_id,samples.submitter_id,samples.sample_type,"
                                   "diagnoses.primary_diagnosis,demographic.gender",
                         "format": "JSON"})
    h = j.get("data", {}).get("hits", [])
    cases += h; frm += size
    if len(h) < size: break
print("  cases = %d" % len(cases), flush=True)

rows = []
for c in cases:
    cid = c["submitter_id"]
    samples = c.get("samples", [])
    if isinstance(samples, str): samples = []
    stypes = sorted({s.get("sample_type") for s in samples if s.get("sample_type")})
    sampids = [s.get("submitter_id") for s in samples if s.get("submitter_id")]
    dx = c.get("diagnoses", [])
    dx = dx[0].get("primary_diagnosis") if isinstance(dx, list) and dx else None
    rows.append(dict(case_id=cid, n_samples=len(sampids), sample_types="|".join(stypes),
                     diagnosis=dx))
cdf = pd.DataFrame(rows)
print("  sample_type 组合分布:" if len(cdf) else "", flush=True)
print(cdf.sample_types.value_counts().head(12).to_string(), flush=True)
print("\n  组织学分布:", flush=True)
print(cdf.diagnosis.value_counts().head(12).to_string(), flush=True)

# ---------------- 3. tumor slide filter ----------------
TUMORY = ("Primary Tumor", "Recurrent Tumor", "Metastatic", "Solid Tissue")
def is_tumor(row):
    st = set(row.sample_types.split("|"))
    return len(st & set(TUMORY)) > 0 and "Blood Derived Normal" not in st

m = sdf.merge(cdf, on="case_id", how="left")
m["tumor_only"] = m.apply(is_tumor, axis=1)
print("\n  切片: 共 %d ; 属于有肿瘤样本的 case 的切片 %d"
      % (len(m), int(m.tumor_only.sum())), flush=True)
print("  纯肿瘤(不含 Blood Derived Normal)的 case 数 = %d"
      % m[m.tumor_only].case_id.nunique(), flush=True)

# ---------------- 4. six-gene labels per case ----------------
print("\n=== 3. 6 基因标签 (ssm_occurrences) ===", flush=True)
lab = {}
for g in GENES:
    flg = {"op": "and", "content": [
           {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJ]}},
           {"op": "in", "content": {"field": "ssm.consequence.transcript.gene.symbol",
                                    "value": [g.lower()]}}]}
    hits, frm = [], 0
    while True:
        j = post("/ssm_occurrences", {"filters": flg, "size": 1000, "from": frm,
                                      "fields": "case.submitter_id", "format": "JSON"})
        h = j.get("data", {}).get("hits", [])
        hits += h; frm += 1000
        if len(h) < 1000: break
    cs = set()
    for h in hits:
        c = h.get("case", {})
        cid = c.get("submitter_id") if isinstance(c, dict) else None
        if cid: cs.add(cid)
    lab[g] = cs
    print("   %-6s 突变 case = %d" % (g, len(cs)), flush=True)

allc = sorted(cdf.case_id)
ldf = pd.DataFrame([dict(case_id=c, **{g: int(c in lab[g]) for g in GENES}) for c in allc])

# ---------------- 5. save ----------------
m.to_csv(os.path.join(AL, "alchem_slides.csv"), index=False)
ldf.to_csv(os.path.join(AL, "alchem_labels_6gene.csv"), index=False)
with open(os.path.join(AL, "alchem_maf_fileids.json"), "w") as f:
    flm = {"op": "and", "content": [
           {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJ]}},
           {"op": "in", "content": {"field": "files.data_type", "value": ["Masked Somatic Mutation"]}},
           {"op": "in", "content": {"field": "files.access", "value": ["open"]}}]}
    mf = []
    frm = 0
    while True:
        j = fetch("/files", {"filters": flm, "size": 500, "from": frm,
                             "fields": "file_id,file_name,file_size", "expand": "cases", "format": "JSON"})
        h = j.get("data", {}).get("hits", []); mf += h; frm += 500
        if len(h) < 500: break
    def _c(x):
        cs = x.get("cases", [])
        if isinstance(cs, list) and cs and isinstance(cs[0], dict):
            return cs[0].get("submitter_id")
        return None
    json.dump([{"file_id": x["file_id"].split("/")[-1], "file_name": x["file_name"], "case_id": _c(x)}
               for x in mf], f, indent=1)
    print("\n  MAF 文件 = %d" % len(mf), flush=True)

print("\n=== 4. 可用交集 ===", flush=True)
tumor_cases = set(m[m.tumor_only].case_id)
any6 = set(ldf[(ldf[GENES].sum(axis=1) > 0)].case_id)
for g in GENES:
    print("  %-6s 阳性 case = %-5d (其中纯肿瘤 %d)" % (g, int(ldf[g].sum()),
          len({c for c in lab[g] if c in tumor_cases})), flush=True)
print("  有肿瘤切片的 case = %d" % len(tumor_cases), flush=True)
print("  有肿瘤切片 且 6基因至少一个阳性 = %d" % len(tumor_cases & any6), flush=True)
print("\n[saved] %s" % AL, flush=True)
print("DONE", flush=True)