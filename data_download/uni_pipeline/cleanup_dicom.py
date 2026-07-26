#!/usr/bin/env python3
"""cleanup_dicom_after.py — pipeline 跑完后清理 dicom 的工具

策略:
  - 默认: 只保留 patches 已经抽完且 dicom 已下的 case (已经在跑策略)
  - 选项: 跑完后, 删除所有 dicom 释放 168 GB 空间
  - 选项: 只保留 top-N case 的 dicom, 其余删
"""
import argparse
import shutil
from pathlib import Path

PROJECT = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project')
DICOM_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'dicom'
PATCHES_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'patches_direct'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只看, 不删')
    ap.add_argument('--keep-n', type=int, default=0, help='只保留前 N 个最大 case (0=全删)')
    args = ap.parse_args()

    if not DICOM_ROOT.exists():
        print(f"[INFO] {DICOM_ROOT} 不存在")
        return

    cases = []
    for case_dir in DICOM_ROOT.iterdir():
        if not case_dir.is_dir():
            continue
        size = sum(f.stat().st_size for f in case_dir.rglob('*') if f.is_file())
        cases.append((case_dir.name, size))

    cases.sort(key=lambda x: x[1], reverse=True)
    total_size = sum(s for _, s in cases)
    print(f"共 {len(cases)} case, 总 {total_size/1024**3:.2f} GB")

    if args.keep_n > 0:
        keep = set(name for name, _ in cases[:args.keep_n])
        to_delete = [(n, s) for n, s in cases if n not in keep]
        print(f"保留 top {args.keep_n} (大) case, 删 {len(to_delete)} case")
    else:
        to_delete = cases
        print(f"全删 {len(cases)} case")

    delete_size = sum(s for _, s in to_delete)
    print(f"释放空间: {delete_size/1024**3:.2f} GB")

    if args.dry_run:
        print("\n[DRY-RUN] 不实际删除")
        return

    confirm = input("确认删除? (yes/no): ")
    if confirm.lower() != 'yes':
        print("取消")
        return

    for name, size in to_delete:
        case_dir = DICOM_ROOT / name
        shutil.rmtree(case_dir, ignore_errors=True)
        print(f"  删除 {name} ({size/1024**3:.2f} GB)")

    print(f"\n[DONE] 释放 {delete_size/1024**3:.2f} GB")


if __name__ == '__main__':
    main()