#!/usr/bin/env python3
"""download_one_case.py — 在 WSL 里下载单个 TCGA-LUAD case 的最大 SM series

仅下 1 个 case 的最大 SM series (而不是 case 下所有 SM series):
  - 节省带宽/存储 (用户决策: 只要最大的 SM, 大的 dcm)
  - 下完一个 case 后立刻 patch + 特征 + 删 dicom

用法:
  python download_one_case.py <case_id> [--target-dir <DIR>]

依赖: idc-index >= 0.12.4
"""
import argparse
import os
import sys
from pathlib import Path

# Force utf-8 on Windows for safety
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from idc_index import IDCClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case_id', help='TCGA case id, e.g. TCGA-05-4402')
    # Windows 默认路径 (在 WSL 中用 --target-dir /mnt/d/... 覆盖)
    ap.add_argument('--target-dir', default='D:\\Lungcancer\\Lung-Cancer-Nine-Gene-Project\\data\\TCGA-LUAD-WSI\\dicom\\tcga_luad')
    ap.add_argument('--modality', default='SM')
    ap.add_argument('--use-s5cmd-sync', action='store_true', default=True)
    args = ap.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # case_dir 由 idc-index 自动创建 (dirTemplate=%PatientID),不要预先建

    c = IDCClient()
    # Query SM series for this case, then pick biggest
    sql = (
        f"SELECT SeriesInstanceUID, series_size_MB, instanceCount "
        f"FROM index "
        f"WHERE collection_id = 'tcga_luad' "
        f"  AND PatientID = '{args.case_id}' "
        f"  AND Modality = '{args.modality}'"
    )
    df = c.sql_query(sql)
    if df.empty:
        sys.exit(f"[FATAL] no {args.modality} series for {args.case_id} in IDC")

    # Sort by size desc, pick biggest
    df = df.sort_values('series_size_MB', ascending=False)
    biggest_uid = df.iloc[0]['SeriesInstanceUID']
    biggest_mb = df.iloc[0]['series_size_MB']
    print(f"[INFO] case={args.case_id}: {len(df)} {args.modality} series")
    print(f"[INFO] biggest SM = {biggest_uid} ({biggest_mb:.1f} MB)")

    # Download that single series. idc-index will create:
    # <downloadDir>/tcga_luad/<PatientID>/<Studth default dirTemplate creates:
    # <downloadDir>/<collection_id>/<PatientID>/<StudyInstanceUID>/<Modality>_<SeriesInstanceUID>/<dcm>
    # 所以 downloadDir 应该是 PROJECT/data/.../dicom,idc-index 会自动加 tcga_luad/<case>/<study>/<series>/
    if target_dir.name.lower() == 'tcga_luad':
        target_dir = target_dir.parent
        print(f"[INFO] adjusted downloadDir to {target_dir} (to avoid tcga_luad/tcga_luad nesting)")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading to: {target_dir}")
    c.download_dicom_series(
        seriesInstanceUID=biggest_uid,
        downloadDir=str(target_dir),
        show_progress_bar=True,
        use_s5cmd_sync=True,
        dirTemplate='%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID',
    )

    print(f"[OK] done for {args.case_id}")


if __name__ == '__main__':
    main()