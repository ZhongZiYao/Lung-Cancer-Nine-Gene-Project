"""Download Zenodo 45-WSI subset with persistent proxy + retry + path control.
Usage: python download_zenodo.py
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
OUT_DIR = ROOT / "DeepGEM" / "data" / "zenodo45"
PROXY_HOST = "http://127.0.0.1:7892"

# Proxy env vars (zenodo_get uses requests via urllib; HTTP/HTTPS proxy both work)
os.environ["HTTP_PROXY"] = PROXY_HOST
os.environ["HTTPS_PROXY"] = PROXY_HOST
# Lower-level safety: skip SSL verify ONLY if your MITM proxy intercepts TLS;
# for Clash with TLS termination enabled this is usually safe.
# os.environ["CURL_CA_BUNDLE"] = ""  # leave default unless Clash MITM misbehaves

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Confirm proxy reachability + DNS resolve
import urllib.request
print(f"[INFO] Proxy: {PROXY_HOST}")
print(f"[INFO] Output directory: {OUT_DIR}")

# 1) Quick reachability test (not strictly needed but useful for debugging)
try:
    req = urllib.request.Request("https://zenodo.org/api/records/15351001",
                                 headers={"User-Agent": "curl/8.0 zenodo_get-test"})
    with urllib.request.urlopen(req, timeout=15,
                                proxies={"http": PROXY_HOST, "https": PROXY_HOST}) as r:
        print(f"[OK] Zenodo API reachable via proxy (status={r.status}).")
except Exception as e:
    print(f"[WARN] Reachability test failed: {e}")
    print("[WARN] Continuing anyway — zenodo_get will retry per file.")

# 2) Run zenodo_get
ZENODO_ID = "15351001"
LOG_FILE = OUT_DIR.parent / "zenodo_download.log"

print(f"\n[RUN] zenodo_get {ZENODO_ID} -o {OUT_DIR}\n")

cmd = [
    sys.executable, "-m", "zenodo_get",
    ZENODO_ID,
    "-o", str(OUT_DIR),
    "--log-file", str(LOG_FILE),
]

# zenodo_get is a CLI script; if not installed via module, fall back
if subprocess.run([sys.executable, "-c", "import zenodo_get"], capture_output=True).returncode != 0:
    cmd = [sys.executable, "-m", "pip", "install", "zenodo_get"]
    subprocess.check_call(cmd)
    cmd = [sys.executable, "-m", "zenodo_get", ZENODO_ID, "-o", str(OUT_DIR),
           "--log-file", str(LOG_FILE)]

try:
    rc = subprocess.call(cmd, env=os.environ)
except KeyboardInterrupt:
    print("\n[INTERRUPTED] Re-running this script resumes the download.")
    sys.exit(130)

if rc == 0:
    print(f"\n[OK] Download finished → {OUT_DIR}")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT_DIR)}  ({f.stat().st_size/1024/1024:.1f} MB)")
else:
    print(f"\n[FAIL] zenodo_get exited with code {rc}. See log: {LOG_FILE}")