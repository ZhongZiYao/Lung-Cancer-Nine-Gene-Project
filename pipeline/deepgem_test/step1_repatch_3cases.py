"""
deepgem_test/step1_repatch_3cases.py
────────────────────────────────────
为本地 sanity check **重新切 patch** 给 3 个 case:
  TCGA-05-4382 (EGFR+)
  TCGA-05-4249 (EGFR-)
  TCGA-05-4395 (EGFR-)

复用现有 extract_patches_direct.py(已经支持 multi-slice + 256×256 + bg/tissue 过滤)
输出到 pipeline/deepgem_test/patches/

用法(在 deepgem conda env):
    python step1_repatch_3cases.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent.parent
PATCHES_OUT = HERE / "patches"
DICOM = ROOT / "transfer_package" / "data" / "TCGA-LUAD-WSI" / "dicom" / "tcga_luad"
SCRIPT = ROOT / "pipeline" / "extract_patches_direct.py"


def main():
    cases = ["TCGA-05-4382", "TCGA-05-4249", "TCGA-05-4395"]
    cmd = [
        "python",
        str(SCRIPT),
        "--dicom", str(DICOM),
        "--out",   str(PATCHES_OUT),
        "--patch-size", "256",
        "--stride", "256",
    ]
    for c in cases:
        cmd.extend(["--case", c])

    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\n[done] 3 cases patched under", PATCHES_OUT)


if __name__ == "__main__":
    main()
