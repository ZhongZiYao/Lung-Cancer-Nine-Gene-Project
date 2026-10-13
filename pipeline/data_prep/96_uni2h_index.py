"""96_uni2h_index.py - 扫所有解压的 h5, 抽 case_id 与 CPTAC 标签 join, 评估可用规模"""
import os, sys, json, h5py, glob, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\public_features\MahmoodLab_UNI2-h\CPTAC_extracted"
LABELS = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\cptac_labels_rebuilt.csv"
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]

# CPTAC case id 形式: C3L-xxxxx-yy, 也可能是 C3N-...
def parse_cptac_id(name):
    """Return (case_id, block) or (None, None) for non-CPTAC files.

    The tar files are organised as e.g. CPTAC/<cohort>/C3L-00001-21.h5 where
    '21' is the block number. case_id is the cohort + 5-digit number
    (e.g. C3L-00001), block is the suffix.
    """
    rel = name.replace("\\", "/")
    parts = rel.split("/")
    base = os.path.splitext(parts[-1])[0]
    for prefix in ("C3L-", "C3N-"):
        if base.startswith(prefix):
            pieces = base.split("-")
            if len(pieces) >= 3:
                return f"{pieces[0]}-{pieces[1]}", pieces[2]
            return None
    return None

lab = pd.read_csv(LABELS)
lab["submitter_id"] = lab["submitter_id"].astype(str)
lab_dict = dict(zip(lab.submitter_id, lab[GENES + (["disease"] if "disease" in lab.columns else [])].to_dict("records")))
print("标签表中 CPTAC-LUAD case 数 =", int((lab.disease == "LUAD").sum()) if "disease" in lab.columns else len(lab), flush=True)
print("标签总 case 数 =", len(lab), flush=True)

files = sorted(glob.glob(os.path.join(C, "**", "*.h5"), recursive=True))
print("h5 文件总数 =", len(files), flush=True)

rows = []
for f in files:
    rel = os.path.relpath(f, C)
    cid = parse_cptac_id(rel)
    with h5py.File(f, "r") as h:
        feat = h.get("features")
        if feat is None:
            continue
        n_patches = int(feat.shape[1])
    cohort = "CPTAC-LUAD" if cid and cid.startswith("C3L") else ("CPTAC-other" if cid else "non-CPTAC")
    rec = dict(path=rel, n_patches=n_patches, cohort=cohort, case_id=cid or "")
    if cid and cid in lab_dict:
        rec["has_label"] = True
        for g in GENES:
            rec[g] = int(lab_dict[cid].get(g, -1))
        rec["disease"] = lab_dict[cid].get("disease", "")
    else:
        rec["has_label"] = False
        for g in GENES:
            rec[g] = -1
        rec["disease"] = ""
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(C, "_index.csv"), index=False)
print("\n[written]", os.path.join(C, "_index.csv"), "rows =", len(df), flush=True)

print("\n=== 按 cohort 统计 ===", flush=True)
print(df.groupby("cohort").agg(
    n_files=("path", "count"),
    n_cases=("case_id", "nunique"),
    n_labeled=("has_label", "sum"),
    median_patches=("n_patches", "median"),
    total_patches=("n_patches", "sum"),
).to_string(), flush=True)

print("\n=== C3L-LUAD 标签覆盖率 ===", flush=True)
c3l = df[df.cohort == "CPTAC-LUAD"]
print("  C3L h5 文件数:", len(c3l), flush=True)
print("  C3L case 数:", c3l.case_id.nunique(), flush=True)
print("  已有 6 基因标签的 case 数:", c3l[c3l.has_label].case_id.nunique(), flush=True)
print("  6 基因标签覆盖率:", c3l[c3l.has_label].case_id.nunique() / c3l.case_id.nunique(), flush=True)
print("  patch 总数:", int(c3l.n_patches.sum()), flush=True)
print("  patch 中位数:", int(c3l.n_patches.median()), flush=True)
print("  patch 最少/最多:", int(c3l.n_patches.min()), "/", int(c3l.n_patches.max()), flush=True)

print("\n=== 6 基因逐位覆盖率 (有标签的 C3L) ===", flush=True)
labeled = c3l[c3l.has_label]
print("  病例数:", len(labeled), flush=True)
for g in GENES:
    valid = (labeled[g] != -1).sum()
    pos = ((labeled[g] == 1)).sum()
    print(f"  {g:6s}: 有标签={valid:4d}/{len(labeled):4d} ({100*valid/len(labeled):.1f}%)  阳性={pos:3d}", flush=True)

# 9gene 检查：是否覆盖 6 个基因
print("\n=== 全部 6 个基因都有非 -1 标签的 case 数 ===", flush=True)
all6 = labeled[(labeled[GENES] != -1).all(axis=1)]
print("  ", len(all6), "例", flush=True)
print("  中位 patch 数:", int(c3l[c3l.case_id.isin(all6.case_id)].n_patches.median()), flush=True)
all6.to_csv(os.path.join(C, "_cptac_luad_labeled.csv"), index=False)
print("[written] _cptac_luad_labeled.csv", flush=True)
print("DONE", flush=True)