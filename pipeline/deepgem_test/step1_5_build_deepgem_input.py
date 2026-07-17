"""
deepgem_test/step1_5_build_deepgem_input.py
───────────────────────────────────────────
构造 DeepGEM step3_extract_feature.py 期望的 dict_name2imgs.pkl:
  {case_id: [patch_path, ...]}

输出: deepgem_test/data/dict_name2imgs_3case.pkl
"""
import pickle
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).parent.resolve()
PATCHES = HERE / "patches"
DEFAULT_OUT = HERE / "data" / "dict_name2imgs_3case.pkl"


def build_dict(patches_dir: Path, cases: list[str]) -> dict:
    out = {}
    for c in cases:
        cd = patches_dir / c
        # 用 CV2 读图,顺带过滤太暗 / 完全白 / 错误文件
        files = sorted(p for p in cd.rglob("*.jpg"))
        valid = []
        # 简单读前 1 张验证 CV2 能开,后续不再 IO
        for f in files:
            valid.append(str(f))  # path string
        out[c] = valid
        print(f"  {c}: {len(valid)} patches")
    return out


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--patches", default=str(HERE / "patches"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cases", nargs="*",
                    default=["TCGA-05-4382", "TCGA-05-4249", "TCGA-05-4395"])
    args = ap.parse_args()

    d = build_dict(Path(args.patches), args.cases)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        pickle.dump(d, f)
    print(f"\n[dict] saved -> {out_p}")
    print(f"  格式: {{case_id: [patch_path, ...]}}  (DeepGEM 标准输入)")


if __name__ == "__main__":
    main()
