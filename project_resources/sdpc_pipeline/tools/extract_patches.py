"""
extract_patches.py — WSL Ubuntu 跑的 .sdpc patch 提取流水线

读 D:/pan-caner/data/中日冰冻切片/<id>.sdpc
输出 D:/pan-caner/data/patches/<id>/<x>_<y>.jpg   (512x512 patches, 64px overlap, 20x level)
     D:/pan-caner/data/patches/_thumbs/<id>.jpg   (整张缩略图)
     D:/pan-caner/data/patches/manifest.csv       (每个 patch 路径 + 来源)

用法:
    /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/extract_patches.py --help
    /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/extract_patches.py --dry-run --limit 3
    /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/extract_patches.py --workers 8
    /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/extract_patches.py --workers 8 --limit 50

环境:
    脚本内部自动注入 opensdpc 的 .so 搜索路径并自重启,无需 source 也不需 export。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# IMPORTANT: must come before `import opensdpc` — see _opensdpc_runtime docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opensdpc_runtime import setup_opensdpc_libpath  # noqa: E402

setup_opensdpc_libpath()

import opensdpc  # noqa: E402
from PIL import Image
import numpy as np

# ---- 路径(WSL 视角) ----
SDPC_DIR = Path("/mnt/d/pan-caner/data/中日冰冻切片")
OUT_DIR = Path("/mnt/d/pan-caner/data/patches")
THUMB_DIR = OUT_DIR / "_thumbs"
MANIFEST = OUT_DIR / "manifest.csv"

# ---- 参数 ----
PATCH_SIZE = 512              # px
PATCH_OVERLAP = 64            # px
PATCH_LEVEL = 0               # 0 = level 0 (20x 全分辨率); 用 1 也可以加速
BG_THRESHOLD = 0.7            # 背景 > 70% 的 patch 丢掉
JPEG_QUALITY = 85
THUMB_LEVEL_OFFSET = 2        # patch level + 2 = 缩略图 level(省内存)

# 背景检测:HSV S 通道 + 灰度
def is_background(rgb: np.ndarray, thresh: float = BG_THRESHOLD) -> bool:
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        # 灰度均值
        gray = rgb.mean(axis=2)
        # 灰度 > 220 视为白色背景
        white_ratio = (gray > 220).mean()
        # 灰度 < 30 视为黑色背景
        black_ratio = (gray < 30).mean()
        return (white_ratio + black_ratio) > thresh
    return False


def open_slide(sdpc_path: str):
    return opensdpc.OpenSdpc(sdpc_path)


def extract_one(sdpc_path_str: str) -> dict:
    """处理一个 .sdpc: 存缩略图 + 切 patches + 写 manifest 行"""
    sdpc_path = Path(sdpc_path_str)
    slide_id = sdpc_path.stem
    out_dir = OUT_DIR / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    info = {
        "slide_id": slide_id,
        "sdpc_path": str(sdpc_path),
        "ok": False,
        "n_patches": 0,
        "thumb_ok": False,
        "level_count": None,
        "level_dimensions": None,
        "err": "",
    }

    try:
        slide = open_slide(str(sdpc_path))
        info["level_count"] = slide.level_count
        info["level_dimensions"] = str(slide.level_dimensions)

        # ---- 缩略图 ----
        thumb_level = max(0, slide.level_count - 1)
        try:
            thumb = slide.get_thumbnail(thumb_level)
            if not isinstance(thumb, Image.Image):
                thumb = Image.fromarray(thumb)
            thumb = thumb.convert("RGB")
            thumb.save(THUMB_DIR / f"{slide_id}.jpg", "JPEG", quality=JPEG_QUALITY)
            info["thumb_ok"] = True
        except Exception as e:
            info["err"] += f"thumb:{e};"

        # ---- patch 提取 ----
        if PATCH_LEVEL >= slide.level_count:
            patch_level = slide.level_count - 1
        else:
            patch_level = PATCH_LEVEL

        W, H = slide.level_dimensions[patch_level]
        stride = PATCH_SIZE - PATCH_OVERLAP
        if stride <= 0:
            raise ValueError(f"PATCH_OVERLAP {PATCH_OVERLAP} >= PATCH_SIZE {PATCH_SIZE}")

        manifest_rows = []
        for y in range(0, max(1, H - PATCH_SIZE + 1), stride):
            for x in range(0, max(1, W - PATCH_SIZE + 1), stride):
                x2 = min(x + PATCH_SIZE, W)
                y2 = min(y + PATCH_SIZE, H)
                if x2 - x < PATCH_SIZE // 2 or y2 - y < PATCH_SIZE // 2:
                    continue
                try:
                    rgb = slide.read_region((x, y), patch_level, (x2 - x, y2 - y))
                    if not isinstance(rgb, Image.Image):
                        rgb = Image.fromarray(rgb)
                    arr = np.asarray(rgb.convert("RGB"))
                except Exception as e:
                    continue

                if is_background(arr):
                    continue

                fname = f"{x}_{y}.jpg"
                rgb.save(out_dir / fname, "JPEG", quality=JPEG_QUALITY)
                manifest_rows.append({
                    "slide_id": slide_id,
                    "patch_path": str(out_dir / fname).replace("/mnt/d/", "D:/"),
                    "level": patch_level,
                    "x": x, "y": y, "w": x2 - x, "h": y2 - y,
                    "file_size_bytes": (out_dir / fname).stat().st_size,
                })

        info["n_patches"] = len(manifest_rows)
        info["ok"] = True

        # 写每 slide 的临时 manifest(主进程会合并)
        slide_manifest = out_dir / "_manifest.csv"
        with open(slide_manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["slide_id", "patch_path", "level", "x", "y", "w", "h", "file_size_bytes"])
            w.writeheader()
            w.writerows(manifest_rows)

        try:
            slide.close()
        except Exception:
            pass

    except Exception as e:
        info["err"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc(limit=2)

    return info


def find_sdpc_files(limit: int | None, smallest: bool = False) -> list[Path]:
    files = list(SDPC_DIR.glob("*.sdpc"))
    if smallest:
        files = sorted(files, key=lambda p: p.stat().st_size)
    else:
        files = sorted(files)
    if limit:
        files = files[:limit]
    return files


def merge_manifests(slide_ids: list[str], out_path: Path):
    rows = []
    for sid in slide_ids:
        slide_manifest = OUT_DIR / sid / "_manifest.csv"
        if not slide_manifest.exists():
            continue
        with open(slide_manifest, encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows.extend(r)
    if not rows:
        print(f"[merge] no per-slide manifest found")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[merge] wrote {len(rows)} patch rows -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8, help="process pool size (default 8)")
    ap.add_argument("--limit", type=int, default=0, help="only process N files (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="process 1 file then exit (sanity check)")
    ap.add_argument("--smallest", action="store_true",
                    help="pick the N smallest files first (useful for time/throughput estimation)")
    args = ap.parse_args()

    if args.dry_run:
        args.limit = 1
        args.workers = 1

    files = find_sdpc_files(args.limit, smallest=args.smallest)
    if not files:
        print(f"no .sdpc files under {SDPC_DIR}", file=sys.stderr)
        return 1
    print(f"[start] {len(files)} files, {args.workers} workers, PATCH_LEVEL={PATCH_LEVEL}, "
          f"PATCH_SIZE={PATCH_SIZE}, OVERLAP={PATCH_OVERLAP}, BG_THRESHOLD={BG_THRESHOLD}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    slide_ids = [p.stem for p in files]
    ok_count = 0
    err_count = 0
    total_patches = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(extract_one, str(p)): p for p in files}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                info = fut.result()
            except Exception as e:
                info = {"slide_id": p.stem, "ok": False, "n_patches": 0, "err": f"worker:{e}"}
            tag = "OK" if info["ok"] else "ERR"
            if info["ok"]:
                ok_count += 1
                total_patches += info["n_patches"]
            else:
                err_count += 1
            extra = f"  err={info.get('err','')[:80]}" if not info["ok"] else ""
            print(f"  [{i:>4d}/{len(files)}] {tag}  {info['slide_id']:<24s}  "
                  f"patches={info['n_patches']:>4d}  "
                  f"thumb={'Y' if info.get('thumb_ok') else 'N'}  "
                  f"elapsed={time.time()-t0:.0f}s{extra}")

    # 合并 manifest
    merge_manifests(slide_ids, MANIFEST)

    print(f"\n[done] {ok_count} ok, {err_count} err, total patches={total_patches}, "
          f"elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
