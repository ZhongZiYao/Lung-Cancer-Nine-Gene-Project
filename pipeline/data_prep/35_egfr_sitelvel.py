# -*- coding: utf-8 -*-
"""35_egfr_sitelvel.py — EGFR 位点级标签构建 + 可行性摸底

动机: 二元 EGFR(有/无突变)在本队列 AUC 0.745, 是净信号最低的基因之一。
文献显示 EGFR 位点级 (L858R / Ex19del) 可能更容易也可能更难:
  - Saha AACR 2024: L858R 0.79 / Ex19del 0.77 (从不吸烟 LUAD)
  - Saha Modern Pathology 2026: L858R / E746_A750del 达 0.84-0.93
  - 但同一篇里 KRAS G12D 0.43 / G12V 0.55 -> 特定密码子形态信号为负
先摸底: 各亚型病例数够不够建模, 不够就直接判定这条路不值得做。
"""
import os, sys, re, gzip
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
MAF = os.path.join(R, "data", "MC3", "mc3.v0.2.8.PUBLIC.maf.gz")
POS = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
       "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}
# 必须与 01b_gene_panel_survey.py 的 case_of 完全一致: 用 split 取前三段, 不用正则。
# 严格正则会丢掉样本 ID 非 4 位的病例(实测少算 13 例 EGFR+)。
tn = lambda b: ("-".join(b.split("-")[:3]) if len(b.split("-")) >= 3 and str(b).startswith("TCGA") else None)

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
idx = pd.read_csv(os.path.join(R, "data", "patch_cache", "NSCLC_gigap_sub512_index.csv"))
m = idx.merge(man[["case_id", "project"]], on="case_id", how="inner")
coh = set(m[m.project == "LUAD"].case_id)
print(f"[LUAD 队列] 有 Prov-GigaPath 特征的病例 = {len(coh)}", flush=True)

per_case = defaultdict(Counter)
conseq = defaultdict(Counter)
with gzip.open(MAF, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("#"):
            continue
        hdr = line.rstrip("\n").split("\t"); break
    ci = {c: i for i, c in enumerate(hdr)}
    have_hgvsp = "HGVSp_Short" in ci
    print(f"MAF 列数 {len(hdr)}  HGVSp_Short 可用: {have_hgvsp}", flush=True)
    for line in f:
        c = line.rstrip("\n").split("\t")
        if len(c) < len(hdr):
            continue
        if c[ci["Hugo_Symbol"]] != "EGFR":
            continue
        if c[ci["Variant_Classification"]] not in POS:
            continue
        if "FILTER" in ci and c[ci["FILTER"]].strip().upper() not in {"PASS", "", "."}:
            continue
        cs = tn(c[ci["Tumor_Sample_Barcode"]])
        if cs not in coh:
            continue
        hv = c[ci["HGVSp_Short"]] if have_hgvsp else c[ci["One_Consequence"]]
        oc = c[ci["One_Consequence"]] if "One_Consequence" in ci else c[ci["Variant_Classification"]]
        per_case[cs][hv] += 1
        conseq[cs][oc] += 1

print(f"\n有 EGFR 致病变异的 LUAD 病例: {len(per_case)} / {len(coh)}", flush=True)

def site_class(hs):
    s = hs.upper().replace("P.", "")
    if "E746_A750" in s or "DELETION" in s and "E746" in s:
        return "Ex19del"
    if "L858R" in s:
        return "L858R"
    if "T790M" in s:
        return "T790M"
    if "L861" in s or "G719" in s or "S768" in s or "L768" in s:
        return "other_sens"
    if "EX20" in s or "S768I" in s:
        return "Ex20ins"
    return "other_EGFR"

lab = {}
for cs, cnt in per_case.items():
    sites = set()
    for h in cnt:
        sites.add(site_class(h))
    lab[cs] = sites

names = ["Ex19del", "L858R", "T790M", "other_sens", "Ex20ins", "other_EGFR"]
print("\n=== EGFR 亚型分布 (一个病例可属多类) ===", flush=True)
for nm in names:
    n = sum(1 for cs, s in lab.items() if nm in s)
    print(f"  {nm:<12}{n:>4} 例  ({n/len(coh):.1%} of cohort)", flush=True)
any_egfr = len(per_case)
wt = len(coh) - any_egfr
print(f"  {'WT':<12}{wt:>4} 例  ({wt/len(coh):.1%})")

rows = []
for cs in coh:
    s = lab.get(cs, set())
    rows.append(dict(case_id=cs, EGFR=1 if cs in per_case else 0,
                     **{n: int(n in s) for n in names},
                     **{f"co_{n}": int(n in s) for n in names}))
df = pd.DataFrame(rows)
out = os.path.join(R, "data", "labels", "egfr_sitelvel_luad.csv")
df.to_csv(out, index=False)
print(f"\n[saved] {out}   shape={df.shape}", flush=True)

print("\n=== 可建模性判定 ===", flush=True)
print(f"{'任务':<28}{'阳性':>6}{'阴性':>7}{'可建模?':>10}")
for nm in names:
    n = int(df[nm].sum())
    neg = len(df) - n
    ok = "是" if n >= 30 and neg >= 30 else "否(样本不足)"
    print(f"{'vs WT: '+nm:<28}{n:>6}{neg:>7}{ok:>10}")
n = int(df["EGFR"].sum())
print(f"{'vs WT: 任意EGFR':<28}{n:>6}{len(df)-n:>7}{'是' if n>=30 else '否':>10}")
print("\n注: 位点级任务里'阴性'应排除携带其它 EGFR 变异的病例, 真正干净的 WT 对照"
      f" = {int((df.EGFR==0).sum())} 例", flush=True)
print("DONE", flush=True)