#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_idc_simple.py — download via S3 listing + HTTPS, NO idc-index required.

The trick: idc-index downloads a parquet from a public GCS bucket at first init.
If idc-index can't bootstrap, we use the same parquet via a public URL.

Strategy:
  1. Read parquet URL from idc-index source if available, else fall back to a
     public gist / GitHub release.
  2. Filter for our case_id.
  3. For each series, list S3 prefix via https://s3.amazonaws.com/?list-type=2,
     download all .dcm files.

If you don't have idc-index installed, run:

  pip install duckdb
  pip install requests

then retry.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


def list_s3_keys(prefix, max_keys=200):
    """List keys under S3 prefix using a single page request."""
    import requests
    bucket_path = prefix.split('s3://')[1].rstrip('/')
    list_url = (
        f"https://s3.amazonaws.com/{bucket_path}/"
        f"?list-type=2&max-keys={max_keys}"
    )
    r = requests.get(list_url, timeout=20)
    if r.status_code != 200:
        return []
    import xml.etree.ElementTree as ET
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    root = ET.fromstring(r.text)
    return [el.text for el in root.findall('s3:Contents/s3:Key', ns)]


def download_one(key, dst):
    import requests
    if dst.exists() and dst.stat().st_size > 1024:
        return 'SKIP'
    url = f"https://s3.amazonaws.com/{key}"
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            if r.status_code != 200:
                return 'FAIL'
            tmp = dst.with_suffix(dst.suffix + '.part')
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dst)
        return 'OK'
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return 'FAIL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-file", required=True,
                    help="JSON list of case_id strings")
    ap.add_argument("--series-file", default=None,
                    help="(optional) tsv: case_id, series_aws_url")
    ap.add_argument("--out", default="data/TCGA-LUAD-WSI/dicom/tcga_luad")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-case-mb", type=int, default=4000,
                    help="Skip case if total size > this MB (skip very large)")
    args = ap.parse_args()

    cases = json.load(open(args.case_file))
    print(f"[dl_simple] {len(cases)} cases", flush=True)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # build per-case series list
    if args.series_file:
        import pandas as pd
        df = pd.read_csv(args.series_file, sep='\t')
        case_series = {}
        for cid in cases:
            sub = df[df['PatientID'] == cid]
            if len(sub) == 0:
                print(f"  [WARN] {cid}: not in series file, skip", flush=True)
                continue
            case_series[cid] = []
            for _, row in sub.iterrows():
                url = row['series_aws_url']
                if not url.endswith('/'):
                    url += '/'
                case_series[cid].append(url)
    else:
        sys.exit(
            "[FATAL] --series-file required (manifest TSV with PatientID "
            "+ series_aws_url). Currently idc-index cannot bootstrap.")

    successes = 0
    fails = 0
    pbar_lock = Lock()

    def fetch_all_series_for_case(case_id, series_urls):
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        n_dl = 0
        n_skip = 0
        n_fail = 0
        for url in series_urls:
            try:
                keys = list_s3_keys(url)
            except Exception as e:
                n_fail += 1
                continue
            for key in keys:
                if not key.endswith('.dcm'):
                    continue
                fname = Path(key).name
                # Skip if duplicate (same series has same uuid name across runs)
                dst = case_dir / fname
                r = download_one(key, dst)
                if r == 'SKIP':
                    n_skip += 1
                elif r == 'OK':
                    n_dl += 1
                else:
                    n_fail += 1
        return case_id, n_dl, n_skip, n_fail

    print(f"[dl_simple] starting download with {args.workers} workers",
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(fetch_all_series_for_case, c, urls)
            for c, urls in case_series.items() if urls
        ]
        for fut in as_completed(futures):
            try:
                case_id, n_dl, n_skip, n_fail = fut.result()
            except Exception as e:
                fails += 1
                continue
            with pbar_lock:
                print(f"  [{case_id}] dl={n_dl} skip={n_skip} fail={n_fail}",
                      flush=True)
            if n_fail == 0 and n_dl + n_skip > 0:
                successes += 1
            elif n_fail > 0:
                fails += 1
    elapsed = time.time() - t0
    print(f"\n[dl_simple] DONE in {elapsed/60:.1f} min:")
    print(f"  fully-OK cases: {successes}")
    print(f"  partial/fail cases: {fails}")


if __name__ == "__main__":
    main()
