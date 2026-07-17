"""Download TCGA-LUAD / TCGA-LUSC diagnostic WSI from GDC.

REQUIRES dbGaP approval for TCGA controlled-access (phs000178).
Approval timeline: 5-10 working days after submission.

Steps before running:
    1. Apply: https://dbgap.ncbi.nlm.nih.gov/aa/wga.cgi?page=login
       Project: phs000178.v11.p8 (TCGA)
    2. Once approved, install gdc-client:
         https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
    3. On GDC portal:
         - Filter: Project = TCGA-LUAD, Data Type = Slide Image, Experimental Strategy = Tissue Slide
         - Add to cart
         - Cart -> Download -> "Manifest" (gdc_manifest.txt)
    4. Save manifest in this script's directory
    5. Run: python download_tcga_luad.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
DEFAULT_OUT = ROOT / "data" / "tcga_luad"
DEFAULT_MANIFEST = Path(__file__).parent / "gdc_manifest_tcga_luad.txt"
# Update with where you unzipped gdc-client
GDC_CLIENT = Path(r"C:\tools\gdc-client\gdc-client.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gdc-client", type=Path, default=GDC_CLIENT)
    ap.add_argument("--token", type=Path, default=None,
                    help="Optional: path to gdc-user-token (download from GDC portal)")
    args = ap.parse_args()

    if not args.gdc_client.exists():
        sys.exit(
            f"[FATAL] gdc-client not found at {args.gdc_client}\n"
            "Download from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool\n"
            "Unzip to C:\\tools\\gdc-client\\ and retry."
        )
    if not args.manifest.exists():
        sys.exit(
            f"[FATAL] manifest not found: {args.manifest}\n"
            "On https://portal.gdc.cancer.gov/ :\n"
            "  1. Filter: Project=TCGA-LUAD, Data Type=Slide Image\n"
            "  2. Add all to cart -> Download -> Manifest\n"
            "  3. Save to this folder, then retry."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.gdc_client),
        "download",
        "-m", str(args.manifest),
        "-d", str(args.out),
    ]
    if args.token:
        cmd += ["-t", str(args.token)]

    print(f"[INFO] running: {' '.join(cmd)}")
    print("[WARN] TCGA-LUAD download is ~150 GB and may take 6-24 hours.")
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"[FATAL] gdc-client exited with code {rc}")
    print(f"[OK] downloaded to: {args.out}")


if __name__ == "__main__":
    main()