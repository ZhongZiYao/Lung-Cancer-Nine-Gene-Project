#!/usr/bin/env python3
"""redownload_4_cases.py — 专门重抽 4 个纯 256 case

目标 case (全是 256x256 老 patches, 已被 cleanup 删):
  - TCGA-05-4244
  - TCGA-05-4245
  - TCGA-05-4249
  - TCGA-05-4250

流程: 下载 -> 抽 patch (1120x1120) -> 删 dicom
其他 case 不动
"""
import subprocess
import time
from pathlib import Path

CASES = ['TCGA-05-4244', 'TCGA-05-4245', 'TCGA-05-4249', 'TCGA-05-4250']

PROJECT = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project')
DICOM_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'dicom'
PATCHES_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'patches_direct'

PY = r'D:\conda\envs\lung9gene\python.exe'
DL_SCRIPT = r'D:\tmp\download_one_case.py'
EXTRACT_SCRIPT = r'D:\tmp\extract_one_case_openslide.py'


def run(cmd, timeout=900):
    print(f'\n>>> {" ".join(cmd[:4])} ...')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        last = '\n'.join(r.stdout.strip().splitlines()[-6:])
        print(last)
        if r.returncode != 0:
            print(f'STDERR: {r.stderr.strip()[-500:]}')
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f'TIMEOUT after {timeout}s')
        return False


def cleanup(case):
    d = DICOM_ROOT / case
    if d.exists():
        import shutil
        sz = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
        print(f'[RM] {case} dicom {sz/1024/1024:.1f} MB')


def main():
    t0 = time.time()
    for case in CASES:
        print(f'\n{"="*60}\n=== {case} ===\n{"="*60}')
        # 1. 下载
        ok = run([PY, DL_SCRIPT, case, '--target-dir', str(DICOM_ROOT)], timeout=900)
        if not ok:
            print(f'[SKIP] {case} download failed, continue')
            continue
        # 2. 抽 patch
        ok = run([PY, EXTRACT_SCRIPT, '--case', case,
                  '--dicom-root', str(DICOM_ROOT),
                  '--patches-root', str(PATCHES_ROOT),
                  '--max-patches', '1000'], timeout=900)
        if not ok:
            print(f'[FAIL] {case} extract failed, dicom NOT deleted')
            continue
        # 3. 删 dicom 释放空间
        cleanup(case)
        # 4. 验证
        n = sum(1 for _ in (PATCHES_ROOT / case).rglob('*.jpg'))
        print(f'[OK] {case}: {n} patches')
    print(f'\n[DONE] total {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()