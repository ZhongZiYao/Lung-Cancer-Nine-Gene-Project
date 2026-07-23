#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_idc.py — Download TCGA-LUAD SM (slide microscopy) series DICOMs
from IDC public S3 bucket using s5cmd (Rust, super fast).

Layout produced (matching existing 41-case layout):
  data/TCGA-LUAD-WSI/dicom/tcga_luad/<case_id>/<seriesUID>/SM_<seriesUID>/*.dcm

Strategy:
  1. Read --case-file (list of case_id strings)
  2. For each case: query idc-index parquet for SM series only
  3. Build a s5cmd batch file with 'cp -n -s' (no-clobber + multipart) per series
  4. Run s5cmd once with --numworkers 32 --concurrency 32

If parquet is missing, fall back to listing the case prefix directly:
  s3://idc-open-data/<case-uuid>/*   <- using PatientID-like subdir? No.
  Actually IDC layout is bucket/seriesUUID/*.dcm  (no per-case subdir).

So we still need the case -> series UUID mapping. Use idc-index parquet.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


S5CMD_EXE = r"C:\Users\ziyao\.conda\envs\deepgem\Scripts\s5cmd.exe"
DEFAULT_PARQUET = r"C:\Users\ziyao\AppData\Local\Temp\idc-extract\idc_index_data\idc_index.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True,
                    help="JSON list of case_id strings")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--out", default="data/TCGA-LUAD-WSI/dicom/tcga_luad")
    ap.add_argument("--modality", default="SM")
    ap.add_argument("--numworkers", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(S5CMD_EXE).exists():
        sys.exit(f"[FATAL] s5cmd not found at {S5CMD_EXE}")
    print(f"[dl_idc] s5cmd at: {S5CMD_EXE}", flush=True)

    try:
        import duckdb
    except ImportError:
        sys.exit("[FATAL] duckdb not installed: pip install duckdb")

    cases = json.load(open(args.case_file))
    print(f"[dl_idc] cases = {len(cases)}", flush=True)

    parquet_path = args.parquet.replace('\\', '/')
    if not Path(parquet_path).exists():
        sys.exit(
            f"[FATAL] parquet not found: {parquet_path}\n"
            "Re-extract from idc-index-data whl:\n"
            "  pip download idc-index-data --no-deps -d /tmp/idc-data\n"
            "  unzip -o /tmp/idc-data/idc_index_data-*.whl -d /tmp/idc-extract/\n"
            "  then point --parquet at the parquet")

    con = duckdb.connect()
    case_list = '(' + ', '.join(f"'{c}'" for c in cases) + ')'
    df = con.execute(f"""
        SELECT PatientID, SeriesInstanceUID, Modality, series_size_MB, series_aws_url
        FROM read_parquet('{parquet_path}')
        WHERE collection_id='tcga_luad' AND PatientID IN {case_list}
          AND Modality = '{args.modality}'
        ORDER BY PatientID, SeriesInstanceUID
    """).fetchdf()
    print(f"[dl_idc] found {len(df)} {args.modality} series", flush=True)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    total_mb = df['series_size_MB'].sum()
    print(f"[dl_idc] total estimated size = {total_mb:.0f} MB "
          f"({total_mb/1024:.2f} GB)", flush=True)

    # layout: <case>/<seriesUID>/SM_<seriesUID>/*.dcm
    # To match the existing 41-case layout where SM series have 'SM_' prefix,
    # we group downloads per-case into <case_id>/<seriesUID>/SM_<seriesUID>/
    # via 'cp -n -s --raw <src> <dst>'

    batch_lines = []
    for _, row in df.iterrows():
        url = row['series_aws_url']
        if not url.startswith('s3://'):
            continue
        bucket = url.replace('s3://', '').rstrip('/').rstrip('*').rstrip('/')
        # s5cmd cp syntax:
        #   src MUST have wildcard (*) to mean "all keys under prefix"
        #   dst MUST be a directory path WITHOUT trailing /*
        #   BUT on Windows s5cmd requires the dst dir to already exist
        src = f"s3://{bucket}/*"
        series_uid = row['SeriesInstanceUID']
        su_short = series_uid.replace('.', '')[:30]
        dst_dir = out_root / row['PatientID'] / f"SM_{su_short}"
        # pre-create local dir (s5cmd doesn't always do this on Windows)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir.as_posix()  # NO trailing slash, NO *
        batch_lines.append(f"cp --raw {src} {dst}")

    batch_path = Path("data/TCGA-LUAD-WSI/.s5cmd_batch.txt")
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text('\n'.join(batch_lines) + '\n')
    print(f"[dl_idc] batch: {batch_path} ({len(batch_lines)} cp cmds)",
          flush=True)

    if args.dry_run:
        print("[dry-run] first 3:")
        for ln in batch_lines[:3]:
            print(f"  {ln}")
        sys.exit(0)

    cmd = [
        S5CMD_EXE,
        "--no-verify-ssl",
        "--no-sign-request",
        f"--numworkers={args.numworkers}",
        "run",
        str(batch_path),
    ]
    print(f"[dl_idc] running s5cmd: numworkers={args.numworkers}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"\n[dl_idc] s5cmd exit rc={proc.returncode} in {elapsed/60:.1f} min")
    if elapsed > 0:
        speed = total_mb / elapsed
        print(f"[dl_idc] avg speed = {speed:.1f} MB/s")

    # tally
    ok = miss = 0
    for c in df['PatientID'].unique():
        if (out_root / c).exists() and any((out_root / c).rglob("*.dcm")):
            ok += 1
        else:
            miss += 1
    print(f"[dl_idc] cases with files: {ok}, missing: {miss}")


if __name__ == "__main__":
    main()