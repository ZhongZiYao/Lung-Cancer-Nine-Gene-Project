"""Download NSCLC-Radiogenomics from TCIA.

Public dataset (no DUC), 211 NSCLC cases with WSI + RNA-seq mutations.
Requires TCIA login: https://www.cancerimagingarchive.net/

Usage:
    1. Register / login on TCIA, go to NSCLC-Radiogenomics collection
    2. Click Download -> "Download manifest" -> save as manifest.tcia
       (in this script's directory, or pass --manifest path)
    3. Run this script:
        python download_nsclc_radiogenomics.py
       It will:
         - Read your .tcia manifest
         - Invoke NBIA Data Retriever CLI (auto-downloaded if missing)
         - Save files under ../data/nsclc_radiogenomics/
"""
import argparse
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

ROOT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
DEFAULT_OUT = ROOT / "data" / "nsclc_radiogenomics"
DEFAULT_MANIFEST = Path(__file__).parent / "manifest_nsclc_radiogenomics.tcia"

# NBIA CLI releases
NBIA_CLI_URL = (
    "https://github.com/GrigoryEvko/NBIA_data_retriever_CLI/releases/latest/download/"
    "nbia-data-retriever-cli-windows-amd64.exe"
)
NBIA_CLI_LOCAL = Path(__file__).parent / "tools" / "nbia-data-retriever-cli.exe"


def ensure_cli():
    if NBIA_CLI_LOCAL.exists():
        return NBIA_CLI_LOCAL
    NBIA_CLI_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading NBIA CLI from {NBIA_CLI_URL}")
    urllib.request.urlretrieve(NBIA_CLI_URL, NBIA_CLI_LOCAL)
    print(f"[OK] saved: {NBIA_CLI_LOCAL}")
    return NBIA_CLI_LOCAL


def run(manifest: Path, out_dir: Path, workers: int = 4, skip_existing: bool = True):
    if not manifest.exists():
        sys.exit(
            f"[FATAL] manifest not found: {manifest}\n"
            "Go to https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/\n"
            "Login -> Download -> save manifest to this folder, then retry."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cli = ensure_cli()

    cmd = [
        str(cli),
        "-i", str(manifest),
        "-o", str(out_dir),
        "-p", str(workers),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    print(f"[INFO] running: {' '.join(cmd)}")
    # Hand off to subprocess so the user can Ctrl-C cleanly
    import subprocess
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"[FATAL] nbia retriever exited with code {rc}")
    print(f"[OK] downloaded to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run(args.manifest, args.out, args.workers)


if __name__ == "__main__":
    main()