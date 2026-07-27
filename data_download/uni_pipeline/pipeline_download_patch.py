#!/usr/bin/env python3
"""pipeline_download_patch.py — 简化的下载+抽 patch 流水线 (不抽 UNI)

策略:
  - 421 case 按 n_genes_mutated 排序
  - 每个 case: 下载最大 SM → openslide 抽 patch (限 1000) → 保留 dicom + patches
  - 不跑 UNI (CPU 太慢),保留 dicom + patches 供将来 GPU 跑 features

输出:
  - data/TCGA-LUAD-WSI/dicom/<case>/<study>/SM_<series>/<dcm>
  - data/TCGA-LUAD-WSI/patches_direct/<case>/<slice_uid>/*.jpg
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project')
DICOM_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'dicom'
PATCHES_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'patches_direct'
LOG_FILE = Path(r'D:\tmp\pipeline_dp.log')
PROGRESS_FILE = Path(r'D:\tmp\pipeline_dp_progress.json')

TARGETS_CSV = r'D:\tmp\download_targets.csv'
PY = r'D:\conda\envs\lung9gene\python.exe'

DOWNLOAD_SCRIPT = r'D:\tmp\download_one_case.py'
EXTRACT_SCRIPT = r'D:\tmp\extract_one_case_openslide.py'

MAX_PATCHES_PER_CASE = 1000


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def free_gb():
    total, used, free = shutil.disk_usage('D:\\')
    return free / 1024**3


def patches_exist(case_id):
    """case 是否已抽过 patch"""
    case_patch_dir = PATCHES_ROOT / case_id
    if not case_patch_dir.exists():
        return False
    n = sum(1 for _ in case_patch_dir.rglob('*.jpg'))
    return n > 50  # 至少 50 个 patch 才算成功


def dicom_exist(case_id):
    case_dicom = DICOM_ROOT / case_id
    if not case_dicom.exists():
        return False
    # 至少有 1 个 SM series
    return any(case_dicom.rglob('SM_*/*.dcm'))


def run_step(cmd, timeout=900, log_prefix=''):
    log(f"{log_prefix}CMD: {' '.join(cmd[:4])} ... [timeout={timeout}s]")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.stdout:
            tail = '\n'.join(r.stdout.strip().splitlines()[-8:])
            log(f"{log_prefix}OUT: {tail}")
        if r.returncode != 0 and r.stderr:
            log(f"{log_prefix}ERR: {r.stderr.strip()[-500:]}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"{log_prefix}TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        log(f"{log_prefix}EXCEPTION: {e}")
        return False


def download_one(case_id):
    return run_step([PY, DOWNLOAD_SCRIPT, case_id,
                     '--target-dir', str(DICOM_ROOT)], timeout=900,
                    log_prefix=f"[{case_id} DL] ")


def extract_one(case_id):
    return run_step([PY, EXTRACT_SCRIPT, '--case', case_id,
                     '--dicom-root', str(DICOM_ROOT),
                     '--patches-root', str(PATCHES_ROOT),
                     '--max-patches', str(MAX_PATCHES_PER_CASE)], timeout=900,
                    log_prefix=f"[{case_id} PATCH] ")

def delete_dicom_after_patch(case_id):
    """case 抽完 patch 后, 删除 dicom 释放空间
    仅在 patches 数量 >= 50 时才删 (避免误删导致数据丢失)"""
    case_patch_dir = PATCHES_ROOT / case_id
    n_patches = sum(1 for _ in case_patch_dir.rglob('*.jpg')) if case_patch_dir.exists() else 0
    if n_patches < 50:
        log(f"[{case_id} RM] skip delete, patches={n_patches} < 50")
        return
    case_dicom_dir = DICOM_ROOT / case_id
    if not case_dicom_dir.exists():
        log(f"[{case_id} RM] skip delete, dicom not exists")
        return
    size = sum(f.stat().st_size for f in case_dicom_dir.rglob('*') if f.is_file())
    shutil.rmtree(case_dicom_dir, ignore_errors=True)
    log(f"[{case_id} RM] deleted dicom ({size/1024/1024:.1f} MB), patches preserved ({n_patches})")

def save_progress(progress):
    try:
        PROGRESS_FILE.write_text(repr(progress), encoding='utf-8')
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--targets', default=TARGETS_CSV)
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    ap.add_argument('--skip-existing', action='store_true', default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.targets)
    log(f"[INFO] {len(df)} total targets, free={free_gb():.1f} GB")

    todo = []
    for _, r in df.iterrows():
        case = r['PatientID']
        if args.skip_existing and patches_exist(case):
            continue
        todo.append(case)
    todo = todo[args.start:]
    if args.max_cases:
        todo = todo[:args.max_cases]
    log(f"[INFO] {len(todo)} to do (start={args.start})")

    PATCHES_ROOT.mkdir(parents=True, exist_ok=True)
    n_done = n_fail = n_skip = 0
    t_start = time.time()
    for i, case in enumerate(todo):
        elapsed = time.time() - t_start
        avg_per_case = elapsed / max(1, n_done)
        eta = avg_per_case * (len(todo) - i) if n_done else 0
        log(f"=== [{i+1}/{len(todo)}] case={case} | free={free_gb():.1f}GB | "
            f"elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min ===")
        if free_gb() < args.min_free_gb:
            log(f"[WARN] free<{args.min_free_gb} GB, sleep 10min")
            time.sleep(600)
        # 下 dicom (skip 如果已有)
        if not dicom_exist(case):
            if not download_one(case):
                n_fail += 1
                continue
        else:
            log(f"[{case} DL] skip (dicom already exist)")
        # 抽 patch
        if not extract_one(case):
            n_fail += 1
            continue
        # 抽完 patch 后删除 dicom 释放空间
        delete_dicom_after_patch(case)
        n_done += 1
        save_progress({'done': n_done, 'fail': n_fail, 'last_case': case,
                       'elapsed_sec': time.time() - t_start})
        if (n_done + n_fail) % 5 == 0:
            log(f">>> PROGRESS: {n_done} done, {n_fail} fail, free={free_gb():.1f} GB")
    log(f"[DONE] {n_done} ok, {n_fail} fail, total={(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()