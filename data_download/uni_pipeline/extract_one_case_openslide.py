#!/usr/bin/env python3
"""extract_one_case_openslide.py — 用 openslide (gdcm 后端) 流式读 DICOM WSI

内存占用极低(< 200 MB),即使 80 GB 大切片也不会 OOM。

约束(用户原话):
  - 多个 SM 选最大的 SM (用 TotalPixelMatrix 面积)
  - 多个 .dcm 选最大的 dcm
  - 输出 <case>/<slice_uid>/<x>_<y>.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from openslide import OpenSlide
except ImportError:
    sys.exit("openslide not installed (pip install openslide-python openslide-bin)")


def pick_largest_sm(case_dir):
    """挑 case 下最大的 SM series (按 TotalPixelMatrixRows*Cols)
    在 series 内选最大的 .dcm。
    返回 (dcm_path, slice_uid_str)

    目录可能是:
      <case>/SM_<uid>/<dcm>          (老格式)
      <case>/<study_uid>/SM_<uid>/<dcm>  (idc-index 下载格式)
    """
    if not case_dir.is_dir():
        return None
    # 递归找所有 SM_ 子目录
    series_dirs = sorted([d for d in case_dir.rglob('SM_*') if d.is_dir()])
    if not series_dirs:
        return None
    best_series = None
    best_tpm = 0
    best_dcm_size = 0
    best_dcm = None
    for sdir in series_dirs:
        dcms = sorted(sdir.glob('*.dcm'))
        if not dcms:
            continue
        sizes = [(d, d.stat().st_size) for d in dcms]
        sizes.sort(key=lambda x: x[1], reverse=True)
        rep_dcm = sizes[0][0]
        try:
            # openslide 直接打开能拿到 dims
            s = OpenSlide(str(rep_dcm))
            w, h = s.dimensions
            tpm = w * h
            s.close()
        except Exception as e:
            print(f"  [warn] {sdir.name}: openslide open fail: {e}")
            continue
        print(f"  series {sdir.name}: dim={w}x{h}={tpm} ({len(dcms)} dcm, biggest={sizes[0][1]/1024/1024:.1f} MB)")
        if tpm > best_tpm or (tpm == best_tpm and sizes[0][1] > best_dcm_size):
            best_tpm = tpm
            best_dcm_size = sizes[0][1]
            best_series = sdir
            best_dcm = rep_dcm
    if best_dcm is None:
        return None
    return best_dcm


def extract_case(case_id, dicom_root, patches_root, patch_size=1120, stride=1120,
                 bg_thresh=0.95, min_tissue=0.15, max_patches=1000):
    """
    抽样策略: "前 N 个有效 patch" (top-left 起点, row-major 扫描)
    - 网格按 stride=1120 平铺整个 dcm
    - 按行从上到下,每行从左到右遍历
    - 遇到背景 (>95% 白) 或无内容 (<15% 组织) 跳过
    - 累计保留 max_patches=1000 个就停止 (常见 row 17-30 终止)
    - 大 case 通常只切顶部一小部分 (中下部没切到)
    """
    case_dir = dicom_root / case_id
    dcm_path = pick_largest_sm(case_dir)
    if dcm_path is None:
        print(f"[SKIP] {case_id}: no SM series")
        return False
    print(f"[INFO] {case_id}: {dcm_path.parent.name}/{dcm_path.name} ({dcm_path.stat().st_size/1024/1024:.1f} MB)")

    s = OpenSlide(str(dcm_path))
    w_total, h_total = s.dimensions
    print(f"  dims={w_total}x{h_total}")

    # 计算 patch grid
    patches_x = max(1, (w_total - patch_size) // stride + 1)
    patches_y = max(1, (h_total - patch_size) // stride + 1)

    # slice_uid = dcm uuid (no .dcm)
    slice_uid = dcm_path.stem
    out_dir = patches_root / case_id / slice_uid
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob('*.jpg'):
        p.unlink()

    print(f"  patch grid = {patches_x}x{patches_y} = {patches_x*patches_y} patches")

    n_kept = 0
    n_bg = 0
    n_tissue = 0
    # 抽样策略: row-major 扫描 (从左上到右下), 前 N 个有效 patch
    for py in range(patches_y):
        for px in range(patches_x):
            # 限制 max_patches: 达到 1000 立即停止
            if n_kept >= max_patches:
                break
            x0 = px * stride
            y0 = py * stride
            x1 = min(x0 + patch_size, w_total)
            y1 = min(y0 + patch_size, h_total)
            cw = x1 - x0
            ch = y1 - y0
            try:
                img = s.read_region((x0, y0), 0, (cw, ch))
            except Exception as e:
                print(f"      [warn] read_region({x0},{y0}) fail: {e}")
                continue
            arr = np.array(img.convert('RGB'))
            # bg / tissue check
            gray = arr.mean(axis=2)
            white_frac = (gray > 220).mean()
            tissue_frac = (gray < 200).mean()
            if white_frac > bg_thresh:
                n_bg += 1
                continue
            if tissue_frac < min_tissue:
                n_tissue += 1
                continue
            out_path = out_dir / f"{x0}_{y0}.jpg"
            Image.fromarray(arr).save(str(out_path), quality=85)
            n_kept += 1
        if n_kept >= max_patches:
            print(f"    [LIMIT] reached max_patches={max_patches}, stop at row {py+1}/{patches_y}")
            break
        if (py + 1) % 20 == 0:
            print(f"    row {py+1}/{patches_y}: kept={n_kept} bg_skip={n_bg} tissue_skip={n_tissue}")
    s.close()
    print(f"[OK] {case_id}: kept {n_kept} patches (bg={n_bg}, tissue={n_tissue}) -> {out_dir}")
    return n_kept > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--dicom-root', default=r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\data\TCGA-LUAD-WSI\dicom')
    ap.add_argument('--patches-root', default=r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\data\TCGA-LUAD-WSI\patches_direct')
    ap.add_argument('--patch-size', type=int, default=1120)
    ap.add_argument('--stride', type=int, default=1120)
    ap.add_argument('--max-patches', type=int, default=1000,
                    help='每个 case 最多抽多少 patch (默认 1000)')
    args = ap.parse_args()
    extract_case(args.case, Path(args.dicom_root), Path(args.patches_root),
                 patch_size=args.patch_size, stride=args.stride,
                 max_patches=args.max_patches)


if __name__ == '__main__':
    main()