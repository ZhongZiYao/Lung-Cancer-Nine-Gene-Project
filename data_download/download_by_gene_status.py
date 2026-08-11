"""download_by_gene_status.py

根据 9gene_panel_LUAD.csv 中的 EGFR 状态,从 IDC 拉 TCGA-LUAD WSI。

用法示例:
  # 拉 5 EGFR+ + 5 EGFR-
  python data_download/download_by_gene_status.py --n-each 5

  # 拉所有 EGFR+
  python data_download/download_by_gene_status.py --n-each all --gene-status pos

  # 拉特定 case
  python data_download/download_by_gene_status.py --cases TCGA-05-4402 TCGA-38-4627

数据落到: ./data/TCGA-LUAD-WSI/dicom/tcga_luad/<case_id>/

依赖: pip install idc-index
"""
from __future__ import annotations

import argparse
import os
import sys

# Force utf-8 to avoid Windows gbk crash on sql files used by idc-index
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\9gene_panel_LUAD.csv'))
    ap.add_argument('--gene', default='EGFR', choices=['EGFR','KRAS','ALK','ROS1','TP53','BRAF','PIK3CA','ERBB2','NRAS','RET','MET'])
    ap.add_argument('--n-each', default='5',
                    help='每个类别 (pos/neg) 拉几个 — 整数或 "all"')
    ap.add_argument('--cases', nargs='+', default=None,
                    help='手动指定 case,会跳过 csv 选择 (优先级高)')
    ap.add_argument('--gene-status', choices=['pos', 'neg', 'both'], default='both',
                    help='选了哪些状态 (--cases 模式时忽略)')
    ap.add_argument('--out', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\data\TCGA-LUAD-WSI\dicom\tcga_luad'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pickled-from', type=Path,
                    default=Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\project_resources\data_public\picked_50_case_ids.json'),
                    help='仅从 picked_50 文件夹里挑 (避免下载 IDC 上 case 全集)')
    args = ap.parse_args()

    # ---- import idc-index (fix utf-8 errors) ----
    try:
        from idc_index import IDCClient
    except ImportError:
        sys.exit('[FATAL] idc-index not installed.  pip install idc-index')

    # ---- 选择 case ----
    if args.cases:
        case_ids = list(args.cases)
        print(f'[INFO] using {len(case_ids)} user-specified cases')
    else:
        # 检查 csv + 用 picked_50 过滤
        df = pd.read_csv(args.csv)
        if args.gene not in df.columns:
            sys.exit(f'[FATAL] gene "{args.gene}" not in {args.csv}: {list(df.columns)}')
        if args.pickled_from and args.pickled_from.exists():
            picked = set(__import__('json').load(open(args.pickled_from)))
            df_picked = df[df['case_id'].isin(picked)]
            print(f'[INFO] csv {len(df)} cases  →  picked_50 {len(df_picked)}')
        else:
            df_picked = df
            print(f'[INFO] csv {len(df)} cases (no picked_50 filter)')

        df_pos = df_picked[df_picked[args.gene] == 1]
        df_neg = df_picked[df_picked[args.gene] == 0]
        chosen = []
        n_each = None if args.n_each == 'all' else int(args.n_each)
        if args.gene_status in ('pos', 'both'):
            n = min(n_each or len(df_pos), len(df_pos))
            chosen.extend(df_pos.case_id.iloc[:n].tolist())
            print(f'[INFO] pick {n} EGFR+ from {len(df_pos)} available')
        if args.gene_status in ('neg', 'both'):
            n = min(n_each or len(df_neg), len(df_neg))
            chosen.extend(df_neg.case_id.iloc[:n].tolist())
            print(f'[INFO] pick {n} EGFR- from {len(df_neg)} available')
        case_ids = chosen

    print(f'[INFO] total cases to download: {len(case_ids)}')
    for c in case_ids:
        print(f'  {c}')
    if not case_ids:
        sys.exit('[FATAL] no cases selected')

    # ---- IDC query ----
    args.out.mkdir(parents=True, exist_ok=True)
    client = IDCClient()
    selection = 'collection_id=tcga_luad&Modality=SM&rows=5000'
    print(f'[INFO] querying IDC for tcga_luad SM series...')
    df_idc = client.get_index_with_selection(selection)
    print(f'[OK] {len(df_idc)} SM series total in tcga_luad')

    df_target = df_idc[df_idc['PatientID'].isin(case_ids)]
    print(f'[INFO] {len(df_target)} series for {len(case_ids)} target cases')
    if len(df_target) == 0:
        print(f'[FATAL] No SM series for any case in IDC')
        return

    # 详细每个 case 状态
    for c in case_ids:
        n = (df_target['PatientID'] == c).sum()
        marker = 'OK' if n > 0 else 'X (no SM)'
        print(f'  {c:18s}  {n} SM series  {marker}')

    # ---- build PatientID selection ----
    pid_selector = 'PatientID=' + ' OR '.join(f'"{c}"' for c in case_ids)
    full_selection = f'{selection}&{pid_selector}'
    print(f'[INFO] downloading to {args.out}  workers={args.workers}')
    print(f'  (this calls s5cmd/s5 over IDC public S3 bucket; ~3-7 GB total)')
    client.download_from_selection(
        full_selection,
        download_dir=str(args.out),
        workers=args.workers,
    )
    print(f'[OK] done. Next: 跑 pipeline/extract_patches_direct.py 来切 patches')


if __name__ == '__main__':
    main()
