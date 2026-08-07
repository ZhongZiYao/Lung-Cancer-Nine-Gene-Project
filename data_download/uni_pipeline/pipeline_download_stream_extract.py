#!/usr/bin/env python3
"""pipeline_download_stream_extract.py
─────────────────────────────────────
完整流式管线:下载 → 切 patch → 抽 UNI → 删切片, 单 case 一个调用。

用法:
  # 单 case (从 idc-index 下载 → 抽特征 → 删切片)
  python pipeline_download_stream_extract.py --case TCGA-49-4494

  # 多个 case (从 csv 读)
  python pipeline_download_stream_extract.py --csv data/MC3/9gene_panel_LUAD.csv \
        --skip-list data/TCGA-LUAD-WSI/downloaded_41.csv
        --max-cases 5

依赖:
  - idc-index >= 0.12.4
  - s5cmd 在 PATH
  - openslide-python + openslide-bin
  - torch + timm + UNI ckpt
"""
from __future__ import annotations
import argparse
import csv
import functools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force utf-8 on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
print = functools.partial(print, flush=True)


def download_one_case(case_id: str, dicom_root: Path) -> bool:
    """用 idc-index 下载一个 case 的最大 SM series。
    返回是否成功。
    """
    try:
        from idc_index import IDCClient
    except ImportError:
        print(f"  [FATAL] idc-index not installed")
        return False

    c = IDCClient()
    sql = (
        f"SELECT SeriesInstanceUID, series_size_MB, instanceCount "
        f"FROM index "
        f"WHERE collection_id = 'tcga_luad' "
        f"  AND PatientID = '{case_id}' "
        f"  AND Modality = 'SM'"
    )
    df = c.sql_query(sql)
    if df.empty:
        print(f"  [FATAL] no SM series for {case_id}")
        return False
    df = df.sort_values("series_size_MB", ascending=False)
    biggest_uid = df.iloc[0]["SeriesInstanceUID"]
    biggest_mb = df.iloc[0]["series_size_MB"]
    print(f"  [DL] biggest SM = {biggest_uid} ({biggest_mb:.1f} MB)")

    # idc-index 用 dirTemplate=%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID
    # 所以 dicom_root 应为 <...>/tcga_luad 才会生成 <case>/<study>/<series>/ 路径
    # 但目前 dicom_root 是 .../tcga_luad (case_dir 直接在 tcga_luad 下),所以用 adjust
    target = dicom_root
    if target.name.lower() != "tcga_luad":
        # 确保 target/tcga_luad/<case>/<study>/<series>/...
        target = target / "tcga_luad"
    target.mkdir(parents=True, exist_ok=True)
    print(f"  [DL] downloading to: {target}")
    try:
        c.download_dicom_series(
            seriesInstanceUID=biggest_uid,
            downloadDir=str(target),
            show_progress_bar=False,
            use_s5cmd_sync=True,
            dirTemplate="%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID",
        )
    except Exception as e:
        print(f"  [ERR] download failed: {e}")
        return False
    return True


def stream_extract_one_case(case_id: str, dicom_root: Path, feat_root: Path,
                            ckpt: str, device: str, batch_size: int,
                            patch_size: int, stride: int,
                            bg_thresh: float, min_tissue: float,
                            keep_dicom: bool) -> bool:
    """流式切 patch → 抽 UNI → 写 npz。
    与 pipeline_stream_uni.py 的 stream_one_case 等价。
    """
    import numpy as np
    import torch
    from openslide import OpenSlide
    from torchvision import transforms as T

    feat_path = feat_root / f"{case_id}.npz"
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    # 已完成 → 跳过 (断点续传)
    if feat_path.exists():
        try:
            d = np.load(feat_path, allow_pickle=True)
            n = d["features"].shape[0]
            print(f"  [SKIP] {case_id}: already done ({n} patches)")
            # 已 done 也删 dicom
            if not keep_dicom:
                shutil.rmtree(dicom_root / case_id, ignore_errors=True)
            return True
        except Exception:
            feat_path.unlink(missing_ok=True)

    case_dir = dicom_root / case_id
    if not case_dir.exists():
        print(f"  [FATAL] dicom dir not found: {case_dir}")
        return False

    # 选最大 dcm (与 pipeline_stream_uni.py 一致)
    candidates = []
    for dcm in case_dir.rglob("*.dcm"):
        try:
            sz = dcm.stat().st_size
            candidates.append((sz, dcm))
        except (FileNotFoundError, OSError):
            continue
    candidates.sort(reverse=True)
    if not candidates:
        print(f"  [FATAL] no .dcm found in {case_dir}")
        return False
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
    if best_dcm is None:
        print(f"  [FATAL] no openslide-readable dcm in {case_dir}")
        return False

    size_mb = best_dcm.stat().st_size / 1024 / 1024
    print(f"  [DCM] {best_dcm.parent.name}/{best_dcm.name} ({size_mb:.1f} MB)")
    s = OpenSlide(str(best_dcm))
    W, H = s.dimensions
    print(f"  [DCM] dims = {W} x {H} = {W*H/1e6:.1f} M-pixels")

    # 加载 UNI
    import timm
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    state = torch.load(ckpt, map_location="cpu")
    msg = model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"  [UNI] loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})")

    tx = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats_buf, ids_buf, paths_buf = [], [], []
    batch_imgs, batch_meta = [], []
    n_total = n_bg = n_tissue = 0
    t0 = time.time()
    flush_every = batch_size * 4

    def flush_to_disk():
        if not feats_buf:
            return
        feats_np = np.stack(feats_buf).astype(np.float32)
        ids_np = np.array(ids_buf, dtype=np.int32)
        paths_np = np.array(paths_buf, dtype=object)
        if feat_path.exists():
            old = np.load(feat_path, allow_pickle=True)
            all_feats = np.concatenate([old["features"], feats_np], axis=0)
            all_ids = np.concatenate([old["slice_ids"], ids_np], axis=0)
            all_paths = np.concatenate([old["patch_paths"], paths_np], axis=0)
            uids = old["slice_uids"]
        else:
            all_feats, all_ids, all_paths = feats_np, ids_np, paths_np
            uids = np.array([best_dcm.stem], dtype=object)
        np.savez_compressed(feat_path,
                            features=all_feats, slice_ids=all_ids,
                            slice_uids=uids, patch_paths=all_paths)
        feats_buf.clear(); ids_buf.clear(); paths_buf.clear()

    @torch.no_grad()
    def process_batch():
        if not batch_imgs:
            return
        ims = torch.stack([tx(img) for img in batch_imgs]).to(device)
        feats = model(ims).cpu().float().numpy()
        feats_buf.extend(list(feats))
        ids_buf.extend([0] * len(batch_imgs))
        paths_buf.extend([f"{x}_{y}.jpg" for (x, y) in batch_meta])
        batch_imgs.clear(); batch_meta.clear()

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    for ry in range(n_rows):
        for cx in range(n_cols):
            x0 = cx * stride
            y0 = ry * stride
            try:
                img = s.read_region((x0, y0), 0, (patch_size, patch_size)).convert("RGB")
            except Exception:
                continue
            arr = np.asarray(img)
            gray = arr.mean(axis=2)
            white_frac = (gray > 220).mean()
            tissue_frac = (gray < 200).mean()
            if white_frac > bg_thresh:
                n_bg += 1; continue
            if tissue_frac < min_tissue:
                n_tissue += 1; continue
            n_total += 1
            batch_imgs.append(img)
            batch_meta.append((x0, y0))
            if len(batch_imgs) >= batch_size:
                process_batch()
                if len(feats_buf) >= flush_every:
                    flush_to_disk()
        if (ry + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [row {ry+1}/{n_rows}] kept={n_total} bg={n_bg} "
                  f"tissue={n_tissue} rate={n_total/max(elapsed,1):.1f}/s "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    process_batch()
    flush_to_disk()
    s.close()

    elapsed = time.time() - t0
    print(f"  [OK] {n_total} patches  time={elapsed/60:.1f} min  "
          f"rate={n_total/max(elapsed,1):.1f} patch/s  -> {feat_path}")

    # 删 dicom (原子化:特征已落盘才删)
    if not keep_dicom:
        try:
            shutil.rmtree(case_dir, ignore_errors=True)
            print(f"  [CLEAN] removed {case_dir}")
        except Exception as e:
            print(f"  [warn] failed to remove {case_dir}: {e}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="单 case_id")
    ap.add_argument("--csv", help="批量模式:从 csv 读 case_id")
    ap.add_argument("--skip-list", help="跳过这些 case (csv with case_id col)")
    ap.add_argument("--max-cases", type=int, default=None)
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
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-dicom", action="store_true",
                    help="不删 dicom (debug)")
    args = ap.parse_args()

    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.batch_size > 8:
        args.batch_size = 8
    print(f"[config] device={device} batch={args.batch_size} "
          f"patch={args.patch_size}x{args.patch_size} stride={args.stride}")

    dicom_root = Path(args.dicom_root)
    feat_root = Path(args.feat_root)

    # 读 skip list (已下好的 case)
    skip = set()
    if args.skip_list and Path(args.skip_list).exists():
        with open(args.skip_list) as f:
            rd = csv.reader(f)
            next(rd, None)  # skip header
            for row in rd:
                if row:
                    skip.add(row[0])
    print(f"[skip-list] {len(skip)} cases")

    # 决定 case 列表
    cases = []
    if args.case:
        cases = [args.case]
    elif args.csv:
        with open(args.csv) as f:
            rd = csv.DictReader(f)
            for r in rd:
                cases.append(r.get("case_id") or r.get("TCGA_ID") or r[0])
    else:
        ap.error("需要 --case 或 --csv")

    # 过滤 skip list
    if skip:
        before = len(cases)
        cases = [c for c in cases if c not in skip]
        print(f"[filter] skip {before - len(cases)} already-done cases")
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"[plan] {len(cases)} cases to download + extract")

    ok = fail = skip_done = 0
    t_start = time.time()
    for i, cid in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {cid}")
        # 1. 下载 (如果还没下)
        case_dir = dicom_root / cid
        if not case_dir.exists() or not list(case_dir.rglob("*.dcm")):
            print(f"  [step 1/2] download")
            if not download_one_case(cid, dicom_root):
                fail += 1
                continue
        else:
            print(f"  [step 1/2] dicom already present, skip download")

        # 2. 抽特征 + 删 dicom
        print(f"  [step 2/2] stream extract")
        try:
            success = stream_extract_one_case(
                cid, dicom_root, feat_root,
                ckpt=args.ckpt, device=device,
                batch_size=args.batch_size,
                patch_size=args.patch_size, stride=args.stride,
                bg_thresh=args.bg_thresh, min_tissue=args.min_tissue,
                keep_dicom=args.keep_dicom,
            )
            if success:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [ERR] extract failed: {e}")
            fail += 1
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(cases) - i)
        print(f"[batch] ok={ok} fail={fail} skip={skip_done}  "
              f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min")

    print(f"\n[done] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()