"""Download CPTAC-LUAD WSI from Imaging Data Commons (IDC).

No dbGaP required - PDC public access.

Usage:
    1. pip install idc-index
    2. python download_cptac_luad.py
       Files saved under ../data/cptac_luad/
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
DEFAULT_OUT = ROOT / "data" / "cptac_luad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    try:
        from idc_index import IDCClient
    except ImportError:
        sys.exit("[FATAL] idc-index not installed. Run: pip install idc-index")

    args.out.mkdir(parents=True, exist_ok=True)

    client = IDCClient()
    # CPTAC-LUAD collection, only pathology slides (SMI = Slide Microscopy Image)
    # See: https://portal.imaging.datacommons.cancer.gov/explore/
    selection = (
        "collection_id=CPTAC-LUAD"
        "&Modality=SM"  # Slide Microscopy
        "&rows=5000"
    )
    print(f"[INFO] querying IDC for CPTAC-LUAD WSI...")
    df = client.get_index_with_selection(selection)
    print(f"[OK] found {len(df)} series")
    print(df.head())

    print(f"[INFO] downloading to {args.out} ...")
    client.download_from_selection(
        selection,
        download_dir=str(args.out),
        workers=args.workers,
    )
    print(f"[OK] done. Saved to: {args.out}")


if __name__ == "__main__":
    main()