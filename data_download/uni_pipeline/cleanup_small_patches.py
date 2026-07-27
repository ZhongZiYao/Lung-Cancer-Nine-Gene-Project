#!/usr/bin/env python3
"""cleanup_small_patches.py — 删除 256x256 老 patches

策略:
  - 每个 case 下, 删除 < 50KB 的 patch (256x256 老 grid patches)
  - 保留 >= 50KB 的 patch (1120x1120 新 patches)
  - 如果 case 下全是 < 50KB, 删除整个 case 目录

用法:
  D:\conda\envs\lung9gene\python.exe D:\tmp\cleanup_small_patches.py [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

P = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\data\TCGA-LUAD-WSI\patches_direct')
SIZE_THRESHOLD = 50_000  # bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='只看, 不删')
    args = ap.parse_args()

    total_del = 0
    total_del_size = 0
    cases_removed = 0
    cases_kept = 0

    for case_dir in sorted(P.iterdir()):
        if not case_dir.is_dir():
            continue
        # 扫描所有 patches
        jpgs = list(case_dir.rglob('*.jpg'))
        if not jpgs:
            continue
        small = [p for p in jpgs if p.stat().st_size < SIZE_THRESHOLD]
        large = [p for p in jpgs if p.stat().st_size >= SIZE_THRESHOLD]
        if not small:
            # 全部 >= 50KB, 保留
            cases_kept += 1
            continue
        # 有 small patch, 需要删
        if not large:
            # 全是 small -> 删整个 case 目录
            sz = sum(p.stat().st_size for p in small)
            if args.dry_run:
                print(f'  [DRY] would remove dir {case_dir.name} ({len(small)} patches, {sz/1024/1024:.1f} MB)')
            else:
                shutil.rmtree(case_dir, ignore_errors=True)
                print(f'  [RM-DIR] {case_dir.name} ({len(small)} patches, {sz/1024/1024:.1f} MB)')
            cases_removed += 1
            total_del += len(small)
            total_del_size += sz
        else:
            # 混合 -> 只删 small
            sz = sum(p.stat().st_size for p in small)
            if args.dry_run:
                print(f'  [DRY] would remove {len(small)} small patches in {case_dir.name} (keep {len(large)} large)')
            else:
                for p in small:
                    p.unlink(missing_ok=True)
                # 也删空的 subdirs
                for sub in case_dir.iterdir():
                    if sub.is_dir() and not any(sub.iterdir()):
                        sub.rmdir()
                print(f'  [RM] {case_dir.name}: removed {len(small)} small (keep {len(large)} large)')
            total_del += len(small)
            total_del_size += sz
            cases_kept += 1  # case 还留着

    print()
    if args.dry_run:
        print(f'[DRY-RUN] would delete {total_del} small patches, {total_del_size/1024/1024/1024:.2f} GB')
        print(f'[DRY-RUN] would remove {cases_removed} full case dirs')
    else:
        print(f'[DONE] deleted {total_del} small patches, {total_del_size/1024/1024/1024:.2f} GB')
        print(f'[DONE] removed {cases_removed} full case dirs, kept {cases_kept} cases')


if __name__ == '__main__':
    main()