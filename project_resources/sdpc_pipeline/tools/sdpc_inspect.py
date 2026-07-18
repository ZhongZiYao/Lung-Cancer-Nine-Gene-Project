"""
sdpc_inspect.py — 深圳生强(SQS/TEKSQRAY) .sdpc 格式头部纯 Python 解析器

不需要 TEKSQRAY 客户端、不需要 DecodeSdpcDll.dll/libDecodeSdpc.so,
仅依赖 Python 3.8+ 标准库,通过 ctypes.Structure 直接 unpack 文件头。

依据:opensdpc(github.com/WonderLandxD/opensdpc) 的 Sdpc_struct.py 结构体定义。
可读出:
  - SqPicHead     文件元信息(magic、版本、金字塔层级、源分辨率、瓦片大小、
                  缩略图尺寸、像素物理尺寸、放大倍数、扩展/tile 偏移)
  - SqPersonInfo  患者信息(病理号、姓名、性别、年龄、科室、医院、临床诊断、
                  病理诊断、报告日期、主治医师、备注)
  - SqExtraInfo   扫描仪信息(扫描仪型号、扫描耗时、扫描时间、相机曝光/增益、
                  序列号、条码、聚焦信息、Z 步距)

像素/缩略图(20x 全分辨率切片、thumbnail jpeg 等)需要 TEKSQRAY 官方解码库
(Windows: DecodeSdpcDll.dll; Linux: libDecodeSdpc.so)——
这部分的限制详见 docs/sdpc_调研报告.md。

用法:
  python tools/sdpc_inspect.py <path/to/file.sdpc>
  python tools/sdpc_inspect.py <dir> --csv sdpc_metadata.csv
  python tools/sdpc_inspect.py data/中日冰冻切片/895983003.sdpc --json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from ctypes import (
    Structure, c_ubyte, c_ushort, c_short, c_uint, c_int, c_int64,
    c_float, c_double, c_char_p,
)
from pathlib import Path
from typing import Any, Iterable


# ---------- 结构体定义(逐字段与 opensdpc/Sdpc_struct.py 一致) ----------

class SqPicHead(Structure):
    _pack_ = 1
    _fields_ = [
        ("flag",            c_ushort),
        ("version",         c_ubyte * 16),
        ("headSize",        c_uint),
        ("fileSize",        c_int64),
        ("macrograph",      c_uint),
        ("personInfor",     c_uint),
        ("hierarchy",       c_uint),
        ("srcWidth",        c_uint),
        ("srcHeight",       c_uint),
        ("sliceWidth",      c_uint),
        ("sliceHeight",     c_uint),
        ("thumbnailWidth",  c_uint),
        ("thumbnailHeight", c_uint),
        ("bpp",             c_ubyte),
        ("quality",         c_ubyte),
        ("colrSpace",       c_ubyte * 4),
        ("scale",           c_float),
        ("ruler",           c_double),
        ("rate",            c_uint),
        ("extraOffset",     c_int64),
        ("tileOffset",      c_int64),
        ("sliceFormat",     c_ubyte),
        ("headSpace",       c_ubyte * 48),
    ]


class SqPersonInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("flag",                c_ushort),
        ("inforSize",           c_uint),
        ("pathologyID",         c_ubyte * 64),
        ("name",                c_ubyte * 64),
        ("sex",                 c_ubyte),
        ("age",                 c_ubyte),
        ("departments",         c_ubyte * 64),
        ("hospital",            c_ubyte * 64),
        ("submittedSamples",    c_ubyte * 1024),
        ("clinicalDiagnosis",   c_ubyte * 2048),
        ("pathologicalDiagnosis", c_ubyte * 2048),
        ("reportDate",          c_ubyte * 64),
        ("attendingDoctor",     c_ubyte * 64),
        ("remark",              c_ubyte * 1024),
        ("nexOffset",           c_int64),
        ("reserved_1",          c_uint),
        ("reserved_2",          c_uint),
        ("reserved",            c_ubyte * 256),
    ]


class SqExtraInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("flag",             c_short),
        ("inforSize",        c_uint),
        ("nextOffset",       c_int64),
        ("model",            c_ubyte * 20),
        ("ccmGamma",         c_float),
        ("ccmRgbRate",       c_float * 3),
        ("ccmHsvRate",       c_float * 3),
        ("ccm",              c_float * 9),
        ("timeConsuming",    c_ubyte * 32),
        ("scanTime",         c_uint),
        ("stepTime",         c_ushort * 10),
        ("serial",           c_ubyte * 32),
        ("fusionLayer",      c_ubyte),
        ("step",             c_float),
        ("focusPoint",       c_ushort),
        ("validFocusPoint",  c_ushort),
        ("barCode",          c_ubyte * 128),
        ("cameraGamma",      c_float),
        ("cameraExposure",   c_float),
        ("cameraGain",       c_float),
        ("reserved",         c_ubyte * 433),
    ]


# ---------- 工具:把定长 c_ubyte* N 字段解成 Python str ----------

def _decode(buf) -> str:
    raw = bytes(buf)
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


# ---------- 主解析 ----------

def parse_sdpc_header(path: str | os.PathLike) -> dict[str, Any]:
    p = Path(path)
    actual_size = p.stat().st_size
    with open(p, "rb") as f:
        head = SqPicHead.from_buffer_copy(f.read(ctypes.sizeof(SqPicHead)))

    out: dict[str, Any] = {
        "file": p.name,
        "file_path": str(p),
        "file_size_bytes": actual_size,
        "sdpc": {
            "magic_flag_hex": f"0x{head.flag:04X}",
            "version": _decode(head.version),
            "headSize": head.headSize,
            "fileSize_declared": head.fileSize,
            "fileSize_match": head.fileSize == actual_size,
            "hierarchy_levels": head.hierarchy,
            "src_WxH": [head.srcWidth, head.srcHeight],
            "slice_WxH": [head.sliceWidth, head.sliceHeight],
            "thumbnail_WxH": [head.thumbnailWidth, head.thumbnailHeight],
            "bpp": head.bpp,
            "quality": head.quality,
            "color_space_hex": bytes(head.colrSpace).hex(),
            "scale": head.scale,
            "ruler_um_per_pixel": head.ruler,
            "rate": head.rate,
            "extra_offset": head.extraOffset,
            "tile_offset": head.tileOffset,
            "slice_format": head.sliceFormat,
        },
    }

    # ---- 读 SqPersonInfo(如果 headSize 之后存在) ----
    pic_head_end = ctypes.sizeof(SqPicHead)
    with open(p, "rb") as f:
        f.seek(pic_head_end)
        try:
            person = SqPersonInfo.from_buffer_copy(f.read(ctypes.sizeof(SqPersonInfo)))
        except Exception as e:
            person = None
            out["person_error"] = str(e)

    if person is not None:
        out["patient"] = {
            "flag": person.flag,
            "inforSize": person.inforSize,
            "pathologyID": _decode(person.pathologyID),
            "name": _decode(person.name),
            "sex": "M" if person.sex == 1 else ("F" if person.sex == 2 else str(person.sex)),
            "age": person.age,
            "departments": _decode(person.departments),
            "hospital": _decode(person.hospital),
            "submittedSamples": _decode(person.submittedSamples),
            "clinicalDiagnosis": _decode(person.clinicalDiagnosis),
            "pathologicalDiagnosis": _decode(person.pathologicalDiagnosis),
            "reportDate": _decode(person.reportDate),
            "attendingDoctor": _decode(person.attendingDoctor),
            "remark": _decode(person.remark),
        }

    # ---- 读 SqExtraInfo(由 head.extraOffset 定位) ----
    if head.extraOffset and head.extraOffset > 0:
        with open(p, "rb") as f:
            f.seek(head.extraOffset)
            try:
                extra = SqExtraInfo.from_buffer_copy(f.read(ctypes.sizeof(SqExtraInfo)))
                out["scanner"] = {
                    "model": _decode(extra.model),
                    "ccmGamma": extra.ccmGamma,
                    "ccmRgbRate": list(extra.ccmRgbRate),
                    "ccmHsvRate": list(extra.ccmHsvRate),
                    "ccm_matrix": list(extra.ccm),
                    "timeConsuming_str": _decode(extra.timeConsuming),
                    "scanTime_unix": extra.scanTime,
                    "stepTime": list(extra.stepTime),
                    "serial": _decode(extra.serial),
                    "fusionLayer": extra.fusionLayer,
                    "step_um": extra.step,
                    "focusPoint": extra.focusPoint,
                    "validFocusPoint": extra.validFocusPoint,
                    "barCode": _decode(extra.barCode),
                    "cameraGamma": extra.cameraGamma,
                    "cameraExposure_ms": extra.cameraExposure,
                    "cameraGain": extra.cameraGain,
                }
            except Exception as e:
                out["scanner_error"] = str(e)

    # ---- 派生:每层金字塔尺寸(沿用 opensdpc 公式) ----
    base = 1.0 / head.scale if head.scale else 1.0
    out["pyramid"] = {
        "level_downsamples": [base ** i for i in range(head.hierarchy)],
        "level_dimensions_est": [
            [int(head.srcWidth * (base ** i)), int(head.srcHeight * (base ** i))]
            for i in range(head.hierarchy)
        ],
    }

    return out


# ---------- 批量入口 ----------

def iter_sdpc_files(target: str | os.PathLike) -> Iterable[Path]:
    p = Path(target)
    if p.is_file():
        yield p
    else:
        for f in sorted(p.rglob("*.sdpc")):
            yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help=".sdpc file or directory containing them")
    ap.add_argument("--json", action="store_true", help="output full JSON per file")
    ap.add_argument("--csv", type=str, help="write flat CSV to this path")
    args = ap.parse_args()

    files = list(iter_sdpc_files(args.target))
    if not files:
        print(f"no .sdpc files under {args.target}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        try:
            info = parse_sdpc_header(f)
        except Exception as e:
            print(f"[ERR] {f.name}: {e}", file=sys.stderr)
            continue

        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
            print("---")
        else:
            sd = info["sdpc"]
            print(f"{info['file']:<24s}  {sd['version']:<14s}  "
                  f"{sd['src_WxH'][0]}x{sd['src_WxH'][1]:<5d}  "
                  f"levels={sd['hierarchy_levels']}  "
                  f"rate={sd['rate']}  scale={sd['scale']}  "
                  f"ruler={sd['ruler_um_per_pixel']:.4f}um/px  "
                  f"size={info['file_size_bytes']/1e6:.1f}MB")
            if "scanner" in info:
                sc = info["scanner"]
                print(f"  scanner model: {sc['model']}   serial: {sc['serial']}   barCode: {sc['barCode']}")
                print(f"  scanTime: {sc['scanTime_unix']}   step: {sc['step_um']}um   focus: {sc['focusPoint']}/{sc['validFocusPoint']}")
            if "patient" in info:
                pt = info["patient"]
                if any(pt.values()):
                    print(f"  patient: ID={pt['pathologyID']}  name={pt['name']}  sex={pt['sex']}  age={pt['age']}  hospital={pt['hospital']}")

        # flatten for CSV
        sd = info["sdpc"]
        sc = info.get("scanner", {})
        pt = info.get("patient", {})
        rows.append({
            "file": info["file"],
            "file_size_bytes": info["file_size_bytes"],
            "version": sd["version"],
            "hierarchy": sd["hierarchy_levels"],
            "src_W": sd["src_WxH"][0],
            "src_H": sd["src_WxH"][1],
            "slice_W": sd["slice_WxH"][0],
            "slice_H": sd["slice_WxH"][1],
            "thumb_W": sd["thumbnail_WxH"][0],
            "thumb_H": sd["thumbnail_WxH"][1],
            "scale": sd["scale"],
            "ruler_um_per_px": sd["ruler_um_per_pixel"],
            "rate": sd["rate"],
            "bpp": sd["bpp"],
            "quality": sd["quality"],
            "slice_format": sd["slice_format"],
            "scanner_model": sc.get("model", ""),
            "scanner_serial": sc.get("serial", ""),
            "barCode": sc.get("barCode", ""),
            "scanTime_unix": sc.get("scanTime_unix", ""),
            "step_um": sc.get("step_um", ""),
            "camera_exposure_ms": sc.get("cameraExposure_ms", ""),
            "pathologyID": pt.get("pathologyID", ""),
            "patient_name": pt.get("name", ""),
            "patient_sex": pt.get("sex", ""),
            "patient_age": pt.get("age", ""),
            "hospital": pt.get("hospital", ""),
            "departments": pt.get("departments", ""),
            "attendingDoctor": pt.get("attendingDoctor", ""),
            "reportDate": pt.get("reportDate", ""),
        })

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[CSV] wrote {len(rows)} rows -> {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
