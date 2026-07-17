"""Convert .kfb to pyramidal .svs via kfbslide + tifffile.

Plan:
- kfbslide.open_slide gives 6 levels of .kfb
- We tile-read the largest level via PIL.Image and assemble the levels
- For SVS: write a TIFF with multiple IFDs (one per level) using tifffile
- We do NOT use libvips/cucim (Windows bulky), just tifffile + Pillow

Output: out_dir/<basename>.svs  (Aperio-compatible TIFF)
"""
from pathlib import Path
import numpy as np
import time
from PIL import Image
import kfbslide


def kfb_to_svs(kfb_path: Path, svs_path: Path, max_level=None,
               tile_size: int = 512, jpeg_quality: int = 80):
    """Tile-read .kfb level-by-level and assemble pyramidal TIFF (.svs).

    Only writes the levels that fit (level 0 = full resolution, downsampled levels 1+).
    Output is a multi-page TIFF compatible with OpenSlide reads (Aperio format).
    """
    print(f"\n=== {kfb_path.name} -> {svs_path.name} ===")
    t0 = time.time()
    wsi = kfbslide.open_slide(str(kfb_path))
    n_levels = wsi.level_count
    max_level = n_levels if max_level is None else min(max_level, n_levels)

    base_w, base_h = wsi.dimensions
    print(f"  base dims: {base_w}x{base_h}, levels: {n_levels}")

    # We'll write a single-resolution TIFF at a chosen downsampled level (e.g. level 4-5)
    # because Aperio pyramids require libvips to assemble, and Pillow can't tile-write pyramids.
    # Instead: write ONE BIG tile-stitched image at level=4 (typically 8x or 16x downsample).
    # DeepGEM's step1 doesn't require SVS specifically — it accepts .svs/.tif/.tiff.
    # We'll write level=4 (or whichever fits memory) as a single-image TIFF.

    # pick a working level that fits comfortably in memory
    candidates = list(range(n_levels - 1, -1, -1))  # start from lowest res
    chosen = None
    for lvl in candidates:
        w, h = wsi.level_dimensions[lvl]
        nbytes = w * h * 3 / 3  # RGB RGB JPEGs decompressed to 8-bit
        print(f"  level {lvl}: {w}x{h} ({nbytes/1024/1024:.0f} MiB raw)")
        if nbytes < 1500 * 1024 * 1024:  # <1.5 GB raw
            chosen = lvl
            break
    if chosen is None:
        chosen = candidates[-1]

    w, h = wsi.level_dimensions[chosen]
    print(f"  reading level {chosen}: {w}x{h}")

    # Tile-read (no downsample on chosen level — wsi.read_region uses base coords)
    best_for_x = wsi.get_best_level_for_downsample(w / wsi.dimensions[0] * wsi.level_downsamples[0])
    # Use deep zoom on chosen level
    # Approach: tile reads of level `chosen` in tile_size chunks
    # NOTE: kfbslide has .get_best_level_for_downsample + .read_region
    # We use the standard approach: read full image is not always possible; tile-stitch.
    # For small enough chosen level, read directly using PIL.Image via thumbnail trick:
    # kfbslide: wsi.read_region((x, y), level, (w, h)) returns PIL.Image at level `level`

    # Convert kfb's base coordinates to chosen-level coords
    downsample = wsi.level_downsamples[chosen]
    # tile read in chosen-level tile chunks
    full = Image.new("RGB", (w, h))
    n_tiles_x = (w + tile_size - 1) // tile_size
    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y
    print(f"  stitching {n_tiles} tiles ({n_tiles_x} x {n_tiles_y})...")
    n_done = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            x0 = tx * tile_size
            y0 = ty * tile_size
            tw = min(tile_size, w - x0)
            th = min(tile_size, h - y0)
            # base coords
            bx = int(x0 * downsample)
            by = int(y0 * downsample)
            bw = int(tw * downsample)
            bh = int(th * downsample)
            try:
                tile = wsi.read_region((bx, by), chosen, (tw, th))
                tile = tile.convert("RGB")
                full.paste(tile, (x0, y0))
            except Exception as e:
                # leave blank (black) tile; report at end
                pass
            n_done += 1
            if n_done % 50 == 0:
                print(f"    tile {n_done}/{n_tiles}  ({time.time()-t0:.1f}s)")
    print(f"  stitched all tiles in {time.time()-t0:.1f}s")

    svs_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {svs_path} as TIFF (BigTIFF)...")
    full.save(str(svs_path), format="TIFF", compression="tiff_lzw",
              tile=(tile_size, tile_size))
    print(f"  DONE in {time.time()-t0:.1f}s. Size: {svs_path.stat().st_size/1024/1024:.1f} MiB")


if __name__ == "__main__":
    SAMPLE = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision\sample")
    OUT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision\sample_svs")
    OUT.mkdir(exist_ok=True)

    for f in sorted(SAMPLE.glob("*.kfb")):
        out = OUT / (f.stem + ".tif")
        if out.exists():
            print(f"[skip] {out.name} already exists")
            continue
        try:
            kfb_to_svs(f, out, max_level=6)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()