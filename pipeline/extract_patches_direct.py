"""
extract_patches_direct.py — 从 DICOM 直接切 patch,跳过 stitched PNG 中间产物

不拼大 PNG:
  - 读 series 的第一个 .dcm,得到 grid = ceil(TotalPixelMatrix / tile_size)
  - 按 (patch_y, patch_x) 网格遍历,每个 patch 1120x1120
  - 每个 patch = (1120/256) x (1120/256) = 5x5 个 tile 拼起来(边缘 tile 用 pad)
  - 单 tile 256x256x3 = 196 KB,内存安全
  - 切完一个 patch 直接丢弃,从不持有完整 canvas

用法:
    python extract_patches_direct.py \\
        --dicom data/TCGA-LUAD-WSI/dicom/tcga_luad \\
        --out data/TCGA-LUAD-WSI/patches_direct \\
        --case TCGA-05-4245 \\
        --case TCGA-05-4249 \\
        --patch-size 1120 --stride 1120

依赖: pydicom, numpy, pillow
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import pydicom
except ImportError:
    sys.exit("pydicom not installed")

# Windows long-path 兼容 (>260 字符路径走 UNC \\?\)
def _win_long(path_str: str) -> str:
    """如果路径超过 250 字符且是 Windows,加 \\?\ 前缀绕过 MAX_PATH"""
    if sys.platform != "win32":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if len(path_str) > 250 and path_str[1:3] == ":\\":
        return "\\\\?\\" + path_str
    return path_str

def _glob_dcm(dir_path: Path):
    """兼容 Windows 长路径的 glob"""
    return sorted(Path(p) for p in os.listdir(str(dir_path)) if p.endswith(".dcm")) if dir_path.is_dir() else []

DEFAULT_PATCH_SIZE = 256
DEFAULT_STRIDE = 256
DEFAULT_BG_THRESHOLD = 0.95
DEFAULT_MIN_TISSUE = 0.15


def find_all_volume_series(case_dir: Path, modality: str = "SM") -> list[tuple[Path, int]]:
    """返回所有 VOLUME series(不只是最大的) + 排序按面积降序

    TCGA-LUAD 一个 case 通常有 2~4 张切片(同一肿瘤不同部位),
    每个 series 下 .dcm 是金字塔(/NONE 全分辨率 + /RESAMPLED 低/中分辨率 + THUMBNAIL)。
    这里把所有 SM_<uid> 都返回,pick_primary_dcm 再选每张里的全分辨率 .dcm。

    注意:同一个 series 下 .dcm 文件名是 UUID,字典序排列,THUMBNAIL 可能排在最前。
    所以这里遍历 series 下所有 .dcm,只要有任意一个是 VOLUME (非 THUMBNAIL)
    且 TotalPixelMatrix > 0,就保留该 series。

    返回: [(series_dir, h*w 面积), ...]  面积降序
    """
    out = []
    # ⚠️ Windows long-path + iterdir 双重问题,改用 os.listdir + 路径前缀绕过
    case_str = str(case_dir)
    if not os.path.isdir(case_str):
        return out
    try:
        study_names = sorted(os.listdir(case_str))
    except Exception as e:
        print(f"  [err] cannot list {case_str}: {e}")
        return out
    for sname in study_names:
        sp = case_dir / sname
        if not sp.is_dir():
            continue
        try:
            series_names = sorted(os.listdir(str(sp)))
        except Exception:
            continue
        for sname2 in series_names:
            series = sp / sname2
            if not series.is_dir() or not series.name.startswith(f"{modality}_"):
                continue
            try:
                dcm_names = [n for n in os.listdir(str(series)) if n.endswith(".dcm")]
            except Exception:
                dcm_names = []
            files = sorted([series / n for n in dcm_names], key=lambda p: p.name)
            if not files:
                continue
            # 遍历所有 .dcm,找一个 VOLUME (非 THUMBNAIL) 拿 TPM
            best_h, best_w = 0, 0
            found_volume = False
            for fp in files:
                fp_str = _win_long(str(fp))
                try:
                    d = pydicom.dcmread(fp_str, stop_before_pixels=True)
                except Exception as e:
                    # ⚠️ Windows 上偶发 FileNotFoundError,跳到下一个
                    print(f"      [warn] skip {fp.name}: {type(e).__name__}: {e}")
                    continue
                it = list(d.get("ImageType", []))
                if "THUMBNAIL" in it:
                    continue
                if "VOLUME" not in it:
                    continue
                h = int(d.TotalPixelMatrixRows or 0)
                w = int(d.TotalPixelMatrixColumns or 0)
                if h > best_h:  # 最大的 VOLUME 决定 series 的 TPM
                    best_h, best_w = h, w
                found_volume = True
            if found_volume and best_h > 0 and best_w > 0:
                out.append((series, best_h * best_w))
    out.sort(key=lambda x: x[1], reverse=True)  # 大切片先切
    return out


def pick_primary_dcm(series_dir: Path) -> tuple[np.ndarray, "pydicom.Dataset"]:
    """挑 series 里**完整切片**的 .dcm

    同一 series 下 .dcm 大小 / n_frames 悬殊:
      - 几百 MB = 完整切片 (n_frames = grid_tile_count)
      - 几十 MB = 子采样版本 (n_frames < grid) 或 z-stack 焦平面堆叠(空白)
      - 几 MB = 元数据 / 单帧 thumbnail

    挑选规则:
      1. 优先选 n_frames == grid_tile_count (ceil(TotalPixelMatrix / tile_size)^2)
         的 .dcm(完整数据,无冗余)
      2. 没有完全匹配的 → 选 n_frames 最接近 grid_tile_count 的
         (假设是 grid 的子采样)
      3. 都没有 4D → 抛错
    """
    import os
    series_str = str(series_dir)
    if os.path.isdir(series_str):
        dcm_names = [n for n in os.listdir(series_str) if n.endswith(".dcm")]
    else:
        dcm_names = []
    files = sorted([series_dir / n for n in dcm_names], key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"no .dcm in {series_dir}")

    # 先读一个能读的 dcm 的 metadata 拿 TotalPixelMatrix + tile_size,算 grid_tile_count
    d_meta = None
    total_h, total_w, tile_h, tile_w = 0, 0, 256, 256
    for fp in files:
        try:
            d_meta = pydicom.dcmread(_win_long(str(fp)), stop_before_pixels=True)
            total_h = int(d_meta.TotalPixelMatrixRows or 0)
            total_w = int(d_meta.TotalPixelMatrixColumns or 0)
            tile_h = int(d_meta.Rows or 256)
            tile_w = int(d_meta.Columns or 256)
            if total_h > 0 and total_w > 0:
                break
        except Exception as e:
            print(f"      [warn] meta skip {fp.name}: {type(e).__name__}: {e}")
            continue
    if total_h == 0 or total_w == 0:
        raise RuntimeError(f"no readable TotalPixelMatrix in {series_dir}")
    n_rows = (total_h + tile_h - 1) // tile_h
    n_cols = (total_w + tile_w - 1) // tile_w
    grid_tile_count = n_rows * n_cols
    print(f"    grid target: {n_rows}x{n_cols} = {grid_tile_count} tiles")

    candidates = []  # (n_frames, size, pixel_array, dcm_dataset)
    for f in files:
        try:
            d = pydicom.dcmread(_win_long(str(f)))
            a = d.pixel_array
        except Exception as e:
            print(f"      [warn] pixel_array skip {f.name}: {type(e).__name__}: {e}")
            continue
        if a.ndim != 4:
            continue
        n_frames = a.shape[0]
        try:
            sz = os.path.getsize(_win_long(str(f)))
        except Exception:
            sz = 0
        candidates.append((n_frames, sz, a, d))

    if not candidates:
        raise RuntimeError(f"no 4D pixel_array in any .dcm under {series_dir}")

    # 按规则挑选
    exact = [c for c in candidates if c[0] == grid_tile_count]
    if exact:
        # 多个完全匹配 → 取最大的(避免重复备份但内容相同,选文件大的相对保险)
        chosen = max(exact, key=lambda c: c[1])
        reason = f"exact match ({chosen[0]} frames == grid)"
    else:
        # 无完全匹配 → 取最接近 grid_tile_count 的(n_frames 不超太多)
        # 按 |n_frames - grid| 升序,|.| 相同时按 n_frames 降序(越接近完整越好)
        candidates.sort(key=lambda c: (abs(c[0] - grid_tile_count), -c[0]))
        chosen = candidates[0]
        reason = (f"closest to grid ({chosen[0]} frames, "
                  f"diff={chosen[0] - grid_tile_count:+d})")

    n_frames, size, pixel_array, d0 = chosen
    print(f"    pick dcm: n_frames={n_frames}, size={size/1e6:.1f} MB ({reason})")
    return pixel_array, d0


def is_background_patch(patch_rgb: np.ndarray, bg_threshold: float) -> bool:
    """H&E 纯白/纯黑背景 → True"""
    if patch_rgb.size == 0:
        return True
    img = Image.fromarray(patch_rgb)
    hsv = np.asarray(img.convert("HSV"))
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    very_bright = (v > 230).mean()  # H&E 亮区域 V 经常 240~250,旧阈值>250 会漏
    very_dark = (v < 10).mean()
    return (very_bright + very_dark) > bg_threshold


def extract_one_case(case_dir: Path,
                     out_root: Path,
                     patch_size: int,
                     stride: int,
                     bg_threshold: float,
                     min_tissue: float,
                     verbose: bool = True) -> dict:
    case_id = case_dir.name
    series_list = find_all_volume_series(case_dir)
    if not series_list:
        return {"case_id": case_id, "ok": False, "err": "no VOLUME SM series"}

    case_out = out_root / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    all_manifest_rows = []
    per_series = []
    case_total_bytes = 0
    case_n_total = 0
    case_n_kept = 0
    case_n_bg = 0
    case_n_tis = 0

    for series_dir, area in series_list:
        if verbose:
            print(f"  [{case_id}] series {series_dir.name[:40]}...  area={area}")
        # 挑 series 里最大的 .dcm (完整切片数据)
        try:
            pixel_array, d0 = pick_primary_dcm(series_dir)
        except RuntimeError as e:
            if verbose:
                print(f"    [skip series] {e}")
            per_series.append({"series": series_dir.name, "ok": False, "err": str(e)})
            continue

        rows, stats = _extract_one_series(
            pixel_array, d0, series_dir, case_out,
            patch_size, stride, bg_threshold, min_tissue, verbose=verbose,
        )
        all_manifest_rows.extend(rows)
        per_series.append({
            "series": series_dir.name,
            "ok": True,
            "grid": stats["grid"],
            "canvas": stats["canvas"],
            "n_patches_grid": stats["n_patches_grid"],
            "n_total_patches": stats["n_total"],
            "n_kept": stats["n_kept"],
            "n_bg_dropped": stats["n_bg"],
            "n_tissue_dropped": stats["n_tissue"],
            "bytes": stats["bytes"],
        })
        case_total_bytes += stats["bytes"]
        case_n_total += stats["n_total"]
        case_n_kept += stats["n_kept"]
        case_n_bg += stats["n_bg"]
        case_n_tis += stats["n_tissue"]

    # 写 case-level manifest
    if all_manifest_rows:
        with open(case_out / "_manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(all_manifest_rows)

    if verbose:
        print(f"  [{case_id}] DONE: {len(series_list)} series, "
              f"kept={case_n_kept}/{case_n_total} "
              f"(bg={case_n_bg}, tissue={case_n_tis}), "
              f"{case_total_bytes/1e6:.1f} MB")

    return {
        "case_id": case_id,
        "ok": True,
        "n_series": len(series_list),
        "n_series_ok": sum(1 for s in per_series if s.get("ok")),
        "n_total_patches": case_n_total,
        "n_kept": case_n_kept,
        "n_bg_dropped": case_n_bg,
        "n_tissue_dropped": case_n_tis,
        "total_bytes": case_total_bytes,
        "per_series": per_series,
    }


def _extract_one_series(pixel_array: np.ndarray,
                        d0,
                        series_dir: Path,
                        case_out: Path,
                        patch_size: int,
                        stride: int,
                        bg_threshold: float,
                        min_tissue: float,
                        verbose: bool = True) -> tuple[list[dict], dict]:
    """从一张切片的 pixel_array 切 patch, 输出到 case_out/<slice_uid>/"""
    n_frames, tile_h, tile_w = pixel_array.shape[:3]
    if pixel_array.ndim == 4 and pixel_array.shape[-1] == 3:
        pass
    elif pixel_array.ndim == 4:
        pixel_array = np.repeat(pixel_array[..., None], 3, axis=-1)

    total_h = int(d0.TotalPixelMatrixRows)
    total_w = int(d0.TotalPixelMatrixColumns)
    n_rows = (total_h + tile_h - 1) // tile_h
    n_cols = (total_w + tile_w - 1) // tile_w
    canvas_h = n_rows * tile_h
    canvas_w = n_cols * tile_w

    if verbose:
        print(f"    grid {n_rows}x{n_cols} = {n_rows*n_cols} tiles, "
              f"canvas {canvas_h}x{canvas_w} (target {total_h}x{total_w})")

    n_patches_y = max(1, (canvas_h - patch_size) // stride + 1)
    n_patches_x = max(1, (canvas_w - patch_size) // stride + 1)

    # 输出目录: case_out/<slice_uid>/
    slice_uid = series_dir.name  # SM_<uid>
    out_dir = case_out / slice_uid
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    n_total = 0
    n_bg = 0
    n_tis = 0
    n_kept = 0
    total_bytes = 0

    for py_idx in range(n_patches_y):
        y0 = py_idx * stride
        y1 = min(y0 + patch_size, canvas_h)
        if y1 - y0 < patch_size // 2:
            break
        for px_idx in range(n_patches_x):
            x0 = px_idx * stride
            x1 = min(x0 + patch_size, canvas_w)
            if x1 - x0 < patch_size // 2:
                break

            n_total += 1

            tile_y0 = y0 // tile_h
            tile_y1 = (y1 - 1) // tile_h
            tile_x0 = x0 // tile_w
            tile_x1 = (x1 - 1) // tile_w

            patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            ok = True
            for ty in range(tile_y0, tile_y1 + 1):
                for tx in range(tile_x0, tile_x1 + 1):
                    frame_idx = ty * n_cols + tx
                    if frame_idx >= n_frames:
                        ok = False
                        break
                    tile = pixel_array[frame_idx]
                    ty0 = ty * tile_h
                    tx0 = tx * tile_w
                    iy0 = max(0, y0 - ty0)
                    iy1 = min(tile_h, y1 - ty0)
                    ix0 = max(0, x0 - tx0)
                    ix1 = min(tile_w, x1 - tx0)
                    py0 = max(0, ty0 - y0) + (iy0 - max(0, y0 - ty0))
                    px0 = max(0, tx0 - x0) + (ix0 - max(0, x0 - tx0))
                    py1 = py0 + (iy1 - iy0)
                    px1 = px0 + (ix1 - ix0)
                    if iy1 > iy0 and ix1 > ix0:
                        patch[py0:py1, px0:px1] = tile[iy0:iy1, ix0:ix1]
                if not ok:
                    break

            if not ok:
                continue

            if is_background_patch(patch, bg_threshold):
                n_bg += 1
                continue

            gray = patch.mean(axis=2)
            tissue_ratio = ((gray < 230) & (gray > 20)).mean()
            if tissue_ratio < min_tissue:
                n_tis += 1
                continue

            fname = f"{x0}_{y0}.jpg"
            fpath = out_dir / fname
            Image.fromarray(patch).save(fpath, "JPEG", quality=90)
            fs = fpath.stat().st_size
            total_bytes += fs
            n_kept += 1
            manifest_rows.append({
                "patch_name": fname,
                "slice_uid": slice_uid,
                "x": x0,
                "y": y0,
                "patch_size": patch_size,
                "tissue_ratio": round(float(tissue_ratio), 4),
                "file_size_bytes": fs,
                "source_series": slice_uid,
            })

    return manifest_rows, {
        "grid": f"{n_rows}x{n_cols}",
        "canvas": f"{canvas_h}x{canvas_w}",
        "n_patches_grid": f"{n_patches_y}x{n_patches_x}",
        "n_total": n_total,
        "n_kept": n_kept,
        "n_bg": n_bg,
        "n_tissue": n_tis,
        "bytes": total_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dicom", default="data/TCGA-LUAD-WSI/dicom/tcga_luad")
    ap.add_argument("--out", default="data/TCGA-LUAD-WSI/patches_direct")
    ap.add_argument("--case", action="append", help="指定 case_id,可多次给")
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--bg-threshold", type=float, default=DEFAULT_BG_THRESHOLD)
    ap.add_argument("--min-tissue", type=float, default=DEFAULT_MIN_TISSUE)
    args = ap.parse_args()

    dicom = Path(args.dicom)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.case:
        case_dirs = [dicom / c for c in args.case]
    else:
        case_dirs = sorted(d for d in dicom.iterdir()
                           if d.is_dir() and d.name.startswith("TCGA-"))

    print(f"[start] {len(case_dirs)} cases, "
          f"patch_size={args.patch_size}, stride={args.stride}")

    results = []
    for cd in case_dirs:
        if not cd.is_dir():
            print(f"  [skip] {cd}: not a dir")
            continue
        r = extract_one_case(cd, out, args.patch_size, args.stride,
                             args.bg_threshold, args.min_tissue)
        results.append(r)
        if r.get("ok"):
            n_series = r.get("n_series", "?")
            n_series_ok = r.get("n_series_ok", "?")
            print(f"  [OK] {r['case_id']}: kept {r['n_kept']}/{r['n_total_patches']} "
                  f"({r['total_bytes']/1e6:.1f} MB), "
                  f"series={n_series_ok}/{n_series}, "
                  f"bg={r['n_bg_dropped']}, tissue={r['n_tissue_dropped']}")
        else:
            print(f"  [SKIP] {r['case_id']}: {r.get('err','')}")

    summary = {
        "n_cases": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "total_patches_kept": sum(r.get("n_kept", 0) for r in results),
        "total_bytes": sum(r.get("total_bytes", 0) for r in results),
        "params": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "bg_threshold": args.bg_threshold,
            "min_tissue": args.min_tissue,
        },
        "per_case": results,
    }
    summary_path = out / "_patches_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {summary['n_ok']}/{summary['n_cases']} cases, "
          f"{summary['total_patches_kept']} patches total, "
          f"{summary['total_bytes']/1e6:.1f} MB")
    print(f"  summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())