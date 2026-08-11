"""拉 7 个 TCGA-LUAD EGFR+ 病例 WSI 从 IDC (公开,无需授权)。

跟 data_download/download_cptac_luad.py 同样的 idc-index 模板,
但只下载指定 case_id(不下载整个 collection,省时间 + 空间)。

用法:
    pip install idc-index
    python data_download/download_extra_egfr_pos.py
    # 输出到: ./data/TCGA-LUAD-WSI/dicom/tcga_luad/<case_id>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 7 个 EGFR 阳性 case(从 50-case 列表里有的 EGFR=1)
EGFR_POS = [
    "TCGA-05-4402", "TCGA-05-4410", "TCGA-05-5423",
    "TCGA-38-4627", "TCGA-38-4628", "TCGA-38-4629", "TCGA-38-6178",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("./data/TCGA-LUAD-WSI/dicom/tcga_luad"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cases", nargs="+", default=EGFR_POS,
                    help="覆盖默认列表")
    args = ap.parse_args()

    try:
        from idc_index import IDCClient
    except ImportError:
        sys.exit("[FATAL] idc-index 未装.  pip install idc-index")

    args.out.mkdir(parents=True, exist_ok=True)
    client = IDCClient()

    # IDC collection 名是 tcga_luad(下划线,不是连字符)
    selection = (
        f"collection_id=tcga_luad"
        f"&Modality=SM"                # 只下 slide microscopy (训练用的真切片)
        f"&rows=5000"
    )
    print(f"[INFO] querying IDC for tcga_luad SM series...")
    df = client.get_index_with_selection(selection)
    print(f"[OK] {len(df)} series total in IDC tcga_luad (SM)")

    # 过滤到目标 case
    df_target = df[df["PatientID"].isin(args.cases)]
    print(f"[INFO] {len(df_target)} series for {len(args.cases)} target cases")
    if len(df_target) == 0:
        sys.exit(f"[FATAL] IDC 里找不到这几个 case: {args.cases}")

    # 打印简短确认
    for c in args.cases:
        n = (df_target["PatientID"] == c).sum()
        marker = "✓" if n > 0 else "✗(no SM series)"
        print(f"  {c:18s}  {n} SM series  {marker}")

    if (df_target["PatientID"] != args.cases[0]).all():
        # 有 case 全空,提醒
        empties = [c for c in args.cases if c not in df_target["PatientID"].values]
        if empties:
            print(f"[WARN] these case(s) have no SM DICOM in IDC: {empties}")

    print(f"[INFO] downloading to {args.out}  workers={args.workers}")
    # 这里直接走 selection 字符串, idc-index 会按 selection 过滤后下
    # 我们手动指定 PatientID 列表(避免选到无关 series)
    selection_with_pids = (
        selection
        + "&" + "PatientID=" + ",".join(f'"{c}"' for c in args.cases)
    )
    client.download_from_selection(
        selection_with_pids,
        download_dir=str(args.out),
        workers=args.workers,
    )
    print(f"[OK] done. Next: 跑 pipeline/extract_patches_direct.py 把这些 case 切 patches")


if __name__ == "__main__":
    main()
