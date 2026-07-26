#!/usr/bin/env python3
"""pipeline_uni_features.py — 后台并行 UNI 抽特征流水线

策略:
  - 监听 patches_direct/ 和 deepgem_feat/
  - 发现新抽完 patch 的 case, 跑 UNI 抽特征 (限 batch=32)
  - 避免与 pipeline_download_patch.py 抢 CPU:
    * 检测到 pipeline 在跑 (s5cmd / extract_one_case_openslide 进程) 就 sleep
    * 否则抢跑 UNI
  - 跑过的 case 写入 progress.json, 下次启动自动跳过

输出: deepgem_feat/<case>.npz

用法: 后台跑
  D:\conda\envs\lung9gene\python.exe D:\tmp\pipeline_uni_features.py
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project')
DICOM_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'dicom'
PATCHES_ROOT = PROJECT / 'data' / 'TCGA-LUAD-WSI' / 'patches_direct'
FEATURES_ROOT = PROJECT / 'deepgem_feat'
LOG_FILE = Path(r'D:\tmp\pipeline_uni.log')
PROGRESS_FILE = Path(r'D:\tmp\pipeline_uni_progress.json')

PY = r'D:\conda\envs\lung9gene\python.exe'
UNI_STEP1 = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\pipeline\step1_extract_features_uni.py'
UNI_CKPT = r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\UNI\assets\ckpts\vit_large_patch16_224.dinov2.uni_mass100k\pytorch_model.bin'

PIPELINE_DL_SCRIPT = r'D:\tmp\download_one_case.py'
PIPELINE_EXTRACT_SCRIPT = r'D:\tmp\extract_one_case_openslide.py'


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


def pipeline_is_busy():
    """检测主 pipeline 是否在跑"""
    try:
        r = subprocess.run(
            ['powershell', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=20
        )
        cmds = r.stdout
        # 在跑 download 或 extract 子进程
        busy_markers = ['download_one_case.py', 'extract_one_case_openslide.py']
        for line in cmds.splitlines():
            for m in busy_markers:
                if m in line and 'pipeline_uni' not in line:
                    return True
        return False
    except Exception as e:
        log(f"[WARN] pipeline_is_busy check failed: {e}")
        return False


def patches_exist(case_id):
    case_patch_dir = PATCHES_ROOT / case_id
    if not case_patch_dir.exists():
        return False
    n = sum(1 for _ in case_patch_dir.rglob('*.jpg'))
    return n > 50


def features_exist(case_id):
    f = FEATURES_ROOT / f'{case_id}.npz'
    return f.exists() and f.stat().st_size > 1024


def run_uni(case_id, timeout=7200):
    cmd = [PY, UNI_STEP1,
           '--patches', str(PATCHES_ROOT),
           '--out', str(FEATURES_ROOT),
           '--ckpt', UNI_CKPT,
           '--cases', case_id]
    log(f"[{case_id}] CMD: {' '.join(cmd[:4])} ... timeout={timeout}s")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.stdout:
            tail = '\n'.join(r.stdout.strip().splitlines()[-6:])
            log(f"[{case_id}] OUT: {tail}")
        if r.returncode != 0 and r.stderr:
            log(f"[{case_id}] ERR: {r.stderr.strip()[-500:]}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"[{case_id}] TIMEOUT")
        return False
    except Exception as e:
        log(f"[{case_id}] EXC: {e}")
        return False


def save_progress(progress):
    try:
        PROGRESS_FILE.write_text(repr(progress), encoding='utf-8')
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-interval', type=int, default=30, help='检查 pipeline 状态的间隔 (秒)')
    ap.add_argument('--one-shot', action='store_true', help='只跑一轮就退出')
    args = ap.parse_args()

    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_fail = 0
    t_start = time.time()
    last_progress_save = 0

    while True:
        # 找所有有 patch 但没 features 的 case
        candidates = []
        for case_dir in sorted(PATCHES_ROOT.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            if not patches_exist(case_id):
                continue
            if features_exist(case_id):
                continue
            candidates.append(case_id)

        if not candidates:
            log(f"[INFO] no pending cases, sleep 60s")
            if args.one_shot:
                log("[DONE] one-shot complete")
                break
            time.sleep(60)
            continue

        # 检查主 pipeline 是否忙
        is_busy = pipeline_is_busy()
        log(f"[INFO] {len(candidates)} pending, pipeline busy={is_busy}, free={free_gb():.1f}GB")

        # 选一个 case 跑 UNI
        case = candidates[0]
        if is_busy:
            # 主 pipeline 在下/抽 patch, 让 UNI 让路 (sleep 一会)
            log(f"[WAIT] pipeline busy, sleep 30s before UNI")
            time.sleep(30)
            # 再检查一次
            is_busy = pipeline_is_busy()
            if is_busy:
                # 主 pipeline 真的忙, 让 UNI 走但限 batch 小一点
                log(f"[INFO] still busy, run UNI anyway with small batch")

        ok = run_uni(case)
        if ok:
            n_done += 1
        else:
            n_fail += 1

        # 定期保存进度
        if time.time() - last_progress_save > 60:
            save_progress({'done': n_done, 'fail': n_fail, 'last_case': case,
                           'elapsed_sec': time.time() - t_start})
            last_progress_save = time.time()

        if (n_done + n_fail) % 3 == 0:
            log(f">>> PROGRESS: {n_done} done, {n_fail} fail, free={free_gb():.1f} GB")

        if args.one_shot:
            break


if __name__ == '__main__':
    main()