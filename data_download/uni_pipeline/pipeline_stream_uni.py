#!/usr/bin/env python3
"""pipeline_stream_uni.py
────────────────────────
流式"openslide 读 dcm → 切 patch → 立即抽 UNI → 写 npz → 删 patch/dicom"。

设计目标:
  - 不存中间 patch (jpg/png) - 切完立刻送 UNI
  - 不限 patch 数量 (全切)
  - 多 SM → 选最大 SM (按 TotalPixelMatrix 面积)
  - 每个 SM 内 → 选最大 dcm
  - 单 case 一次性跑完,产物 <case>.npz 在 deepgem_feat/
  - 跑完删 dicom + 任何残留 patch 临时目录
  - GPU 优先 (batch=32),CPU 兜底 (batch=8)

用法:
  # 1) 单 case 验证 (需要先下 dicom)
  python pipeline_stream_uni.py --case TCGA-05-4402

  # 2) 多个 case (从 csv 读)
  python pipeline_stream_uni.py --csv data/MC3/9gene_panel_LUAD.csv

  # 3) 自定义参数
  python pipeline_stream_uni.py --case TCGA-05-4402 --patch-size 256 --stride 256 \
                                --bg-thresh 0.95 --min-tissue 0.15

  # 4) 不删 dicom (debug 用)
  python pipeline_stream_uni.py --case TCGA-05-4402 --keep-dicom
"""
from __future__ import annotations
import argparse
import csv
import functools
import shutil
import sys
import time
from pathlib import Path

# unbuffered print
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
print = functools.partial(print, flush=True)


# ─────────────── 选最大 SM → 选最大 dcm ───────────────
def pick_largest_dcm(case_dir: Path):
    """多 series 选最大,每个 series 内选最大 dcm。
    支持路径模式:
      <case>/SM_<uid>/<dcm>
      <case>/<study>/SM_<uid>/<dcm>
      <case>/<study>/<series>/<dcm>   (idc-index 命名不规范时的兜底)
    返回 Path 或 None
    """
    from openslide import OpenSlide

    # 先按大小排序(跳过 stat 失败的)
    candidates = []
    for dcm in case_dir.rglob("*.dcm"):
        try:
            sz = dcm.stat().st_size
            candidates.append((sz, dcm))
        except (FileNotFoundError, OSError):
            continue
    candidates.sort(reverse=True)

    if not candidates:
        return None

    # 抽样前 10 个最大的 dcm 试 OpenSlide
    best_dcm = None
    best_area = 0
    for sz, dcm in candidates[:10]:
        try:
            s = OpenSlide(str(dcm))
            w, h = s.dimensions
            s.close()
            area = w * h
            if area > best_area:
                best_area = area
                best_dcm = dcm
        except Exception:
            continue
    return best_dcm


# ─────────────── UNI 加载 ───────────────
def load_uni(ckpt_path: str, device: str):
    """加载 UNI (ViT-L/16, 1024 dim)"""
    import timm
    import torch
    print(f"[UNI] loading from {ckpt_path}")
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(state, strict=False)
    print(f"[UNI] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval().to(device)
    return model


# ─────────────── 主流程 ───────────────
def stream_one_case(case_id: str, dicom_root: Path, feat_root: Path,
                    ckpt_path: str, device: str, batch_size: int,
                    patch_size: int, stride: int,
                    bg_thresh: float, min_tissue: float,
                    keep_dicom: bool, keep_patches: bool):
    """流式处理一个 case"""
    import numpy as np
    import torch
    from openslide import OpenSlide
    from PIL import Image
    from torchvision import transforms as T

    feat_path = feat_root / f"{case_id}.npz"
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    # 已完成则跳过
    if feat_path.exists():
        try:
            d = np.load(feat_path, allow_pickle=True)
            n = d["features"].shape[0]
            print(f"[SKIP] {case_id}: already done ({n} patches) -> {feat_path}")
            return True
        except Exception:
            print(f"[WARN] {case_id}: corrupted npz, re-processing")
            feat_path.unlink(missing_ok=True)

    # 找 dicom
    case_dir = dicom_root / case_id
    if not case_dir.exists():
        print(f"[FATAL] {case_id}: dicom dir not found at {case_dir}")
        return False
    dcm = pick_largest_dcm(case_dir)
    if dcm is None:
        print(f"[FATAL] {case_id}: no SM series found")
        return False

    size_mb = dcm.stat().st_size / 1024 / 1024
    print(f"\n========== {case_id} ==========")
    print(f"  dcm = {dcm.parent.name}/{dcm.name} ({size_mb:.1f} MB)")

    s = OpenSlide(str(dcm))
    W, H = s.dimensions
    print(f"  dims = {W} x {H} = {W*H/1e6:.1f} M-pixels")
    print(f"  patch grid = {(W-patch_size)//stride+1} x {(H-patch_size)//stride+1}")

    # UNI transform (官方)
    tx = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_uni(ckpt_path, device)

    # 流式批处理
    feats_buf, ids_buf, paths_buf = [], [], []
    batch_imgs = []
    n_total, n_bg, n_tissue = 0, 0, 0
    t0 = time.time()
    flush_every = batch_size * 4   # 每 N 个 patch 写一次 npz (原子)

    def flush_to_disk():
        if not feats_buf:
            return
        # 追加到 npz
        feats_np = np.stack(feats_buf).astype(np.float32)
        ids_np = np.array(ids_buf, dtype=np.int32)
        paths_np = np.array(paths_buf, dtype=object)
        # 写 npz (如果存在就合并)
        if feat_path.exists():
            old = np.load(feat_path, allow_pickle=True)
            all_feats = np.concatenate([old["features"], feats_np], axis=0)
            all_ids = np.concatenate([old["slice_ids"], ids_np], axis=0)
            all_paths = np.concatenate([old["patch_paths"], paths_np], axis=0)
            uids = old["slice_uids"]
        else:
            all_feats, all_ids, all_paths = feats_np, ids_np, paths_np
            uids = np.array([dcm.stem], dtype=object)
        np.savez_compressed(
            feat_path,
            features=all_feats,
            slice_ids=all_ids,
            slice_uids=uids,
            patch_paths=all_paths,
        )
        feats_buf.clear()
        ids_buf.clear()
        paths_buf.clear()

    @torch.no_grad()
    def process_batch():
        if not batch_imgs:
            return
        ims = torch.stack([tx(img) for img in batch_imgs]).to(device)
        feats = model(ims).cpu().float().numpy()
        feats_buf.extend(list(feats))
        ids_buf.extend([0] * len(batch_imgs))   # 单 SM, slice_id 全 0
        paths_buf.extend([f"{x}_{y}.jpg" for (x, y) in batch_meta])
        batch_imgs.clear()
        batch_meta.clear()

    batch_meta = []

    # row-major scan
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    for ry in range(n_rows):
        for cx in range(n_cols):
            x0 = cx * stride
            y0 = ry * stride
            try:
                img = s.read_region((x0, y0), 0, (patch_size, patch_size)).convert("RGB")
            except Exception as e:
                continue
            arr = np.asarray(img)
            gray = arr.mean(axis=2)
            white_frac = (gray > 220).mean()
            tissue_frac = (gray < 200).mean()
            if white_frac > bg_thresh:
                n_bg += 1
                continue
            if tissue_frac < min_tissue:
                n_tissue += 1
                continue
            n_total += 1
            batch_imgs.append(img)
            batch_meta.append((x0, y0))
            if len(batch_imgs) >= batch_size:
                process_batch()
                if n_total % (batch_size * 10) == 0:
                    elapsed = time.time() - t0
                    rate = n_total / elapsed
                    eta = (n_rows * n_cols - (ry * n_cols + cx)) / max(rate, 0.1)
                    print(f"  [{case_id}] row {ry+1}/{n_rows}  "
                          f"kept={n_total} bg={n_bg} tissue={n_tissue}  "
                          f"rate={rate:.1f} patch/s  eta={eta/60:.1f} min", flush=True)
                if len(feats_buf) >= flush_every:
                    flush_to_disk()
        # 进度报告 (每 10 行)
        if (ry + 1) % 10 == 0:
            print(f"  [{case_id}] progress: row {ry+1}/{n_rows}  kept={n_total}  "
                  f"bg={n_bg} tissue={n_tissue}", flush=True)

    # 处理残留
    process_batch()
    flush_to_disk()

    s.close()
    elapsed = time.time() - t0
    print(f"\n[OK] {case_id}: {n_total} patches  "
          f"({n_bg} bg-skip, {n_tissue} tissue-skip)  "
          f"time={elapsed/60:.1f} min  rate={n_total/max(elapsed,1):.1f} patch/s")
    print(f"  -> {feat_path}")

    # 清理 dicom (除非指定 keep)
    if not keep_dicom:
        try:
            shutil.rmtree(case_dir)
            print(f"  [CLEAN] removed {case_dir}")
        except Exception as e:
            print(f"  [warn] failed to remove {case_dir}: {e}")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="单 case_id")
    ap.add_argument("--csv", help="批量:从 csv 读 case_id (取第一列,跳过表头)")
    ap.add_argument("--dicom-root",
                    default=r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\TCGA-LUAD-WSI\dicom\tcga_luad")
    ap.add_argument("--feat-root",
                    default=r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\deepgem_feat_v2")
    ap.add_argument("--ckpt",
                    default=r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\UNI\assets\ckpts\uni\pytorch_model.bin")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--bg-thresh", type=float, default=0.95)
    ap.add_argument("--min-tissue", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / cuda:0; 默认自动")
    ap.add_argument("--keep-dicom", action="store_true",
                    help="不删 dicom (debug 用)")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="限制批量处理的 case 数 (debug 用)")
    args = ap.parse_args()

    # device
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] device={device} batch={args.batch_size} "
          f"patch={args.patch_size}x{args.patch_size} stride={args.stride}")
    if device == "cpu":
        # CPU 跑自动调小 batch
        if args.batch_size > 8:
            print(f"[warn] CPU 跑, batch 自动降到 8")
            args.batch_size = 8

    dicom_root = Path(args.dicom_root)
    feat_root = Path(args.feat_root)

    if args.case:
        stream_one_case(
            case_id=args.case,
            dicom_root=dicom_root, feat_root=feat_root,
            ckpt_path=args.ckpt, device=device,
            batch_size=args.batch_size,
            patch_size=args.patch_size, stride=args.stride,
            bg_thresh=args.bg_thresh, min_tissue=args.min_tissue,
            keep_dicom=args.keep_dicom, keep_patches=args.keep_patches if hasattr(args, "keep_patches") else False,
        )
        return

    # 批量模式
    if not args.csv:
        ap.error("需要 --case 或 --csv 之一")

    case_ids = []
    with open(args.csv) as f:
        rd = csv.reader(f)
        header = next(rd)
        # 找 case_id 列
        if "case_id" in header:
            idx = header.index("case_id")
        else:
            idx = 0
        for row in rd:
            if row and len(row) > idx:
                case_ids.append(row[idx])

    if args.max_cases:
        case_ids = case_ids[: args.max_cases]

    print(f"[batch] {len(case_ids)} cases from {args.csv}")

    ok, fail = 0, 0
    t_start = time.time()
    for i, cid in enumerate(case_ids, 1):
        print(f"\n[{i}/{len(case_ids)}] {cid}")
        try:
            success = stream_one_case(
                case_id=cid,
                dicom_root=dicom_root, feat_root=feat_root,
                ckpt_path=args.ckpt, device=device,
                batch_size=args.batch_size,
                patch_size=args.patch_size, stride=args.stride,
                bg_thresh=args.bg_thresh, min_tissue=args.min_tissue,
                keep_dicom=args.keep_dicom, keep_patches=False,
            )
            if success:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[ERROR] {cid}: {e}")
            fail += 1
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(case_ids) - i)
        print(f"[batch] ok={ok} fail={fail}  "
              f"elapsed={elapsed/60:.1f} min  eta={eta/60:.1f} min")

    print(f"\n[done] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()