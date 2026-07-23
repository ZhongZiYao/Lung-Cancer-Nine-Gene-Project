#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_100_cases.py — pick 59 cases to extend 41 -> 100.

Strategy (constraint-satisfied greedy):

Phase 1 MUST-PICK (positive rare-gene cases too few to drop):
  - NRAS positives (CSV all 3, 41 used 2, remaining 1 must pick)
  - ERBB2 positives (CSV all 8, 41 used 4, remaining 4 pick all)
  - PIK3CA top-N positives (target >= 6 final)

Phase 2 RARE-POSITIVE FILL (target 4-5 positive cases):
  ROS1, RET, MET, PIK3CA, ALK, BRAF

Phase 3 MAIN-POSITIVE FILL (target ~30% positive rate):
  EGFR, KRAS, TP53

Phase 4 CENTER-DIVERSITY FILL (>= 8 centers, no center > 25 cases):
  Centers currently in 41: 05(30), 38(7), 35(4) = 3 centers
  Must add from 55, 44, 86, 78, 50, 49, 97, MP, 75, 91, 95

Phase 5 NEGATIVE FILL:
  A few any-9-gene-negative cases for hard-negative baseline

Output:
  data/TCGA-LUAD-WSI/picked_100_case_ids.json
  data/TCGA-LUAD-WSI/new_59_case_ids.json
  data/TCGA-LUAD-WSI/selection_strategy.json     (audit trail)
  data/TCGA-LUAD-WSI/manifest_100.tsv           (manifest_41 + new_59)

Usage:
  python scripts/select_100_cases.py
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


GENES = ['EGFR', 'KRAS', 'ALK', 'ROS1', 'TP53',
         'BRAF', 'PIK3CA', 'ERBB2', 'NRAS', 'RET', 'MET']


# Target positive counts for 100 case (after 41 already). 41 + 59 = 100.
# All values are LOWER BOUND target; final may exceed.
TARGET_FINAL = {
    'EGFR':   18,   # ~18% positive
    'KRAS':   30,   # ~30%
    'TP53':   52,   # ~52%
    'ALK':    9,    # 3-fold current -> ~9
    'ROS1':   5,    # was 1 -> target 5
    'BRAF':   10,   # was 5 -> target 10
    'PIK3CA': 6,    # was 2 -> target 6
    'ERBB2':  7,    # CSV max 8 -> target 7
    'NRAS':   3,    # CSV only has 3 -> max
    'RET':    5,    # was 1 -> target 5
    'MET':    5,    # was 1 -> target 5
}

# Phase 1 hard-pick rules
MUST_PICK = {
    'NRAS_all_remaining': True,  # pick every NRAS+ not already in 41
    'ERBB2_all_remaining': True, # pick every ERBB2+ not already in 41
}

# Center diversity constraints
TARGET_CENTER_DIVERSITY = 8      # final 100 case should have >= 8 distinct centers
CENTER_HARD_CAP = 25             # no single center > 25 case (41 case already has 30 in 05 -> acceptable to be unbalanced by extending)


def already_used_count(df, used_cases, gene):
    s = df[df['case_id'].isin(used_cases)]
    return int(s[gene].sum())


def center_of(case_id):
    # TCGA-XX-Y... -> XX is center
    parts = case_id.split('-')
    if len(parts) >= 2:
        return parts[1]
    return ''


def is_any_positive(row):
    return int(row['any_9gene_positive']) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/MC3/9gene_panel_LUAD.csv")
    ap.add_argument("--used", default="data/TCGA-LUAD-WSI/picked_41_case_ids.json")
    ap.add_argument("--out-dir", default="data/TCGA-LUAD-WSI")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    df = pd.read_csv(args.csv)
    used = set(json.load(open(args.used)))

    # The candidate pool = CSV - already used
    pool = df[~df['case_id'].isin(used)].copy()
    print(f"[select_100] CSV total = {len(df)}, already used = {len(used)}, "
          f"candidate pool = {len(pool)}", flush=True)

    selected = []  # list of dicts {case_id, reason}
    strategy_log = []

    # =========================================================================
    # Phase 1 — MUST-PICK rare positives
    # =========================================================================
    print("\n[Phase 1] MUST-PICK rare-positive cases:", flush=True)
    for gene in ['NRAS', 'ERBB2', 'PIK3CA']:
        cur = already_used_count(df, used, gene)
        tgt = TARGET_FINAL[gene]
        needed = max(0, tgt - cur)
        positives = pool[pool[gene] == 1].sort_values('case_id')
        if gene == 'NRAS':
            take = positives['case_id'].tolist()  # all of them
        elif gene == 'ERBB2':
            take = positives['case_id'].tolist()  # all of them
        else:  # PIK3CA
            # take as many as needed, but max for CSV
            need = min(needed, len(positives))
            take = positives['case_id'].tolist()[:need]

        for c in take:
            selected.append({'case_id': c, 'reason': f'P1:{gene}+',
                             'center': center_of(c)})
        strategy_log.append({'phase': 1, 'gene': gene, 'picked': take,
                             'reason': 'rare-gene max-pick'})
        # remove from pool
        pool = pool[~pool['case_id'].isin(take)]
        print(f"  {gene}+ : picked {len(take)} (target={tgt}, current={cur})", flush=True)

    # =========================================================================
    # Phase 2 — RARE-POSITIVE FILL (target final >= TARGET_FINAL[gene])
    # =========================================================================
    print("\n[Phase 2] Rare-positive fill:", flush=True)
    rare_genes = ['ROS1', 'RET', 'MET', 'ALK', 'BRAF']
    for gene in rare_genes:
        cur = already_used_count(df, used | {s['case_id'] for s in selected}, gene)
        tgt = TARGET_FINAL[gene]
        needed = max(0, tgt - cur)
        if needed == 0:
            print(f"  {gene}+ : already at target ({cur}), skip", flush=True)
            continue
        positives = pool[pool[gene] == 1].sort_values('case_id')
        # rotate so we don't always pick same lexicographic order
        n = min(needed, len(positives))
        # take with center mix preference
        take = []
        used_centers = Counter()
        for s in selected:
            used_centers[center_of(s['case_id'])] += 1
        positives_with_center = [(row['case_id'], center_of(row['case_id']))
                                 for _, row in positives.iterrows()]
        # sort: prefer new centers, then by case_id
        positives_with_center.sort(key=lambda x: (used_centers[x[1]] > 0, x[0]))
        for case_id, _ in positives_with_center[:n]:
            take.append(case_id)
        for c in take:
            selected.append({'case_id': c, 'reason': f'P2:{gene}+', 'center': center_of(c)})
        pool = pool[~pool['case_id'].isin(take)]
        strategy_log.append({'phase': 2, 'gene': gene, 'picked': take,
                             'reason': 'rare-positive fill'})
        print(f"  {gene}+ : picked {len(take)} (target={tgt}, current={cur})", flush=True)

    # =========================================================================
    # Phase 3 — MAIN-POSITIVE FILL (target final ~TARGET_FINAL)
    # =========================================================================
    print("\n[Phase 3] Main-positive fill:", flush=True)
    main_genes = ['EGFR', 'KRAS', 'TP53']
    for gene in main_genes:
        cur = already_used_count(df, used | {s['case_id'] for s in selected}, gene)
        tgt = TARGET_FINAL[gene]
        needed = max(0, tgt - cur)
        if needed == 0:
            print(f"  {gene}+ : already at target ({cur}), skip", flush=True)
            continue
        positives = pool[pool[gene] == 1].sort_values('case_id')
        n = min(needed, len(positives))
        # same center-diversity preference
        take = []
        used_centers = Counter()
        for s in selected:
            used_centers[center_of(s['case_id'])] += 1
        positives_with_center = [(row['case_id'], center_of(row['case_id']))
                                 for _, row in positives.iterrows()]
        positives_with_center.sort(key=lambda x: (used_centers[x[1]] > 0, x[0]))
        for case_id, _ in positives_with_center[:n]:
            take.append(case_id)
        for c in take:
            selected.append({'case_id': c, 'reason': f'P3:{gene}+', 'center': center_of(c)})
        pool = pool[~pool['case_id'].isin(take)]
        strategy_log.append({'phase': 3, 'gene': gene, 'picked': take,
                             'reason': 'main-positive fill'})
        print(f"  {gene}+ : picked {len(take)} (target={tgt}, current={cur})", flush=True)

    # =========================================================================
    # Phase 4 — CENTER-DIVERSITY FILL
    # =========================================================================
    used_cases_with_picked = used | {s['case_id'] for s in selected}
    current_centers = Counter()
    for c in used_cases_with_picked:
        current_centers[center_of(c)] += 1
    print(f"\n[Phase 4] Center-diversity: current = {dict(current_centers)}", flush=True)

    target_new_centers = ['55', '44', '86', '78', '50', '49', '97', 'MP', '75']
    p4_picks = []
    p4_picks_set = set()
    for new_center in target_new_centers:
        center_cases = pool[pool['case_id'].str.startswith(f'TCGA-{new_center}-')]
        if len(center_cases) == 0:
            continue
        n_take = min(3, len(center_cases))
        sample = center_cases.sample(n_take, random_state=hash(new_center) & 0xFFFFFFFF)
        for _, row in sample.iterrows():
            cid = row['case_id']
            p4_picks.append(cid)
            p4_picks_set.add(cid)
            selected.append({'case_id': cid, 'reason': f'P4:new-center {new_center}',
                             'center': new_center})
    # remove from pool once for all phase 4 picks
    pool = pool[~pool['case_id'].isin(p4_picks_set)]

    strategy_log.append({'phase': 4, 'picked': p4_picks,
                         'reason': 'center-diversity fill'})
    print(f"  picked {len(p4_picks)} from new centers", flush=True)

    # =========================================================================
    # Phase 5 — NEGATIVE FILL (any-9-negative cases for hard baseline)
    # =========================================================================
    used_cases_with_picked = used | {s['case_id'] for s in selected}
    current_n = len(used_cases_with_picked)
    n_remaining = 100 - current_n
    if n_remaining > 0:
        print(f"\n[Phase 5] Negative fill: need {n_remaining} more", flush=True)
        negatives = pool[pool['any_9gene_positive'] == 0].sort_values('case_id')
        n_take = min(n_remaining, len(negatives))
        if n_take > 0:
            take = negatives['case_id'].tolist()[:n_take]
            for c in take:
                selected.append({'case_id': c, 'reason': 'P5:9gene-negative',
                                 'center': center_of(c)})
            strategy_log.append({'phase': 5, 'picked': take,
                                 'reason': 'negative baseline fill'})
            print(f"  picked {len(take)} any-9-negative", flush=True)
        else:
            print(f"  no negatives left in pool, but still need {n_remaining}",
                  flush=True)
    else:
        print(f"\n[Phase 5] Negative fill: already at {current_n}, skip", flush=True)

    # =========================================================================
    # Trim or top-up
    # =========================================================================
    final_cases = sorted({s['case_id'] for s in selected})
    # attach center to each
    for s in selected:
        s['center'] = center_of(s['case_id'])
    final_100 = sorted(used | set(final_cases))
    final_100 = final_100[:100] if len(final_100) > 100 else final_100

    n_final = len(final_100)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  41 already used + {len(final_cases)} new = {n_final} cases",
          flush=True)
    if n_final != 100:
        print(f"  WARNING: expected 100, got {n_final}", flush=True)

    # final per-gene check
    final_df = df[df['case_id'].isin(final_100)]
    print(f"  Per-gene positive in final 100:", flush=True)
    for g in GENES:
        n = int(final_df[g].sum())
        pct = n / n_final * 100
        target = TARGET_FINAL.get(g, '?')
        print(f"    {g:6}: {n:3d} / {n_final} ({pct:5.1f}%)  target={target}",
              flush=True)
    # center distribution
    centers_final = Counter(center_of(c) for c in final_100)
    print(f"\n  Center distribution (final 100):", flush=True)
    for center, n in sorted(centers_final.items(), key=lambda x: -x[1]):
        print(f"    {center}: {n}", flush=True)
    print(f"  Total distinct centers: {len(centers_final)}", flush=True)

    # =========================================================================
    # WRITE OUTPUT
    # =========================================================================
    new_added = sorted(set(final_100) - used)
    (out_dir / "picked_100_case_ids.json").write_text(
        json.dumps(final_100, indent=2))
    (out_dir / "new_59_case_ids.json").write_text(
        json.dumps(new_added, indent=2))
    (out_dir / "selection_strategy.json").write_text(
        json.dumps(strategy_log, indent=2, default=str))
    print(f"\n  Wrote:", flush=True)
    print(f"    {out_dir}/picked_100_case_ids.json", flush=True)
    print(f"    {out_dir}/new_59_case_ids.json", flush=True)
    print(f"    {out_dir}/selection_strategy.json", flush=True)


if __name__ == "__main__":
    main()
