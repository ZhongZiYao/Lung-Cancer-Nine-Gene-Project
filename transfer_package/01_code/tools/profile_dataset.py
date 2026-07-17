"""
profile_dataset.py — 在不依赖 mojibake 中文(没法 GBK 解)的前提下,
用所有"不乱码"的字段画出这个数据集的全貌

数据源: data/_xlsx_dump/01_原始数据集.csv(500 行) + 04_冰冻结果统计.csv(24 行)
       + 之前生成的 sdpc_metadata.csv(分辨率/扫描仪/大小)
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path("/mnt/d/pan-caner")
XLSX_DUMP = WORKSPACE / "data" / "_xlsx_dump_v2"
SDPC_META = WORKSPACE / "data" / "中日冰冻切片" / "_sdpc_metadata.csv"
OUT = XLSX_DUMP / "_dataset_profile.md"


def load_csv(p: Path) -> tuple[list[str], list[dict]]:
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main():
    headers, rows = load_csv(XLSX_DUMP / "01_原始数据集.csv")
    print(f"[01] {len(rows)} rows, cols: {headers}")
    print(f"[01] sample row keys: {list(rows[0].keys())}")

    # 关键 mojibake 列(已确认是 GBK→latin1): 性别、年龄、住院号、标本名称、临床诊断、冰冻诊断、
    # 恶性2交界1良性0(标签)、石蜡最终诊断、石蜡切片诊断三分类、肉眼所见、不一致标记
    # 用正则和 ascii 字符能拿到的是: 病理号、切片号、年龄里的数字、住院号里的数字

    out: list[str] = ["# 数据集画像报告\n"]

    # --- 病理号唯一性: 一个病人可能有多个切片 ---
    path_ids = [r["病理号"] for r in rows]
    pid_counter = Counter(path_ids)
    n_patients = len(pid_counter)
    pid_counts = Counter(pid_counter.values())

    out += [
        "## 1. 样本规模\n",
        f"- 切片(行)数: **{len(rows)}**",
        f"- 唯一病理号(病人): **{n_patients}**",
        f"- 平均每病人切片数: {len(rows)/n_patients:.2f}",
        "",
        "**每病人切片数分布**",
        "| 切片数/病人 | 病人数 |",
        "|---|---|",
    ]
    for k in sorted(pid_counts):
        out.append(f"| {k} | {pid_counts[k]} |")
    out.append("")

    # --- 年龄分布 ---
    ages = []
    for r in rows:
        m = re.search(r"(\d+)\s*岁", r.get("年龄", ""))
        if m:
            ages.append(int(m.group(1)))
    out += [
        "## 2. 年龄分布\n",
        f"- 有年龄数据: {len(ages)}/{len(rows)}",
        f"- 范围: {min(ages)} - {max(ages)}",
        f"- 均值: {sum(ages)/len(ages):.1f}, 中位: {sorted(ages)[len(ages)//2]}",
        "",
        "**年龄段分布**",
        "| 年龄段 | 切片数 |",
        "|---|---|",
    ]
    bins = [(0, 18, "0-18"), (19, 30, "19-30"), (31, 45, "31-45"),
            (46, 60, "46-60"), (61, 75, "61-75"), (76, 120, "76+")]
    for lo, hi, label in bins:
        out.append(f"| {label} | {sum(1 for a in ages if lo <= a <= hi)} |")
    out.append("")

    # --- 性别: UTF-8 dump 后是"男/女"正常中文 ---
    sex_counter = Counter()
    for r in rows:
        sex_counter[r.get("性别", "")] += 1
    out += [
        "## 3. 性别分布(GBK 解码)\n",
        "| 性别 | 切片数 |",
        "|---|---|",
    ]
    for k, v in sex_counter.most_common():
        out.append(f"| {k!r} | {v} |")
    out.append("")

    # --- 良恶性/三分类 ---
    # 列名是"恶性2、交界1、良性0(冰冻诊断三分类)" -> 值可能是 0/1/2 或乱码
    mal_col = "恶性2、交界1、良性0（冰冻诊断三分类）"
    para_col = "石蜡切片诊断三分类"
    out += [
        "## 4. 良恶性分类(冰冻三分类)\n",
        f"- 冰冻标签列: `{mal_col}`",
    ]
    mal_counter = Counter(r.get(mal_col, "") for r in rows)
    out.append("| 冰冻标签值 | 切片数 |")
    out.append("|---|---|")
    for k, v in mal_counter.most_common():
        out.append(f"| {k!r} | {v} |")
    out.append("")

    para_counter = Counter(r.get(para_col, "") for r in rows)
    out += [
        f"## 5. 石蜡最终三分类(`{para_col}`)\n",
        "| 石蜡标签值 | 切片数 |",
        "|---|---|",
    ]
    for k, v in para_counter.most_common():
        out.append(f"| {k!r} | {v} |")
    out.append("")

    # 冰冻 vs 石蜡一致性
    agree = sum(1 for r in rows if r.get(mal_col, "") == r.get(para_col, "") and r.get(mal_col, ""))
    disagree = sum(1 for r in rows if r.get(mal_col, "") and r.get(para_col, "")
                   and r.get(mal_col, "") != r.get(para_col, ""))
    out += [
        "## 6. 冰冻 vs 石蜡三分类一致性\n",
        f"- 双方都有标签: {mal_counter.total() - mal_counter.get('', 0)} (冰冻), "
        f"{para_counter.total() - para_counter.get('', 0)} (石蜡)",
        f"- 一致(冰冻==石蜡): {agree}",
        f"- 不一致: {disagree}",
        "",
    ]

    # --- 04 测试集分组(器官分类) ---
    _, stat_rows = load_csv(XLSX_DUMP / "04_测试集分组.csv")
    out += [
        "## 7. 04_测试集分组(逐器官分类)\n",
        f"- 共 {len(stat_rows)} 行",
        "| 器官 | 计数 | 占比 |",
        "|---|---|---|",
    ]
    for r in stat_rows:
        out.append(f"| {r.get('器官分类', '')!r} | {r.get('计数', '')} | {r.get('占比', '')} |")
    out.append("")

    # --- 02 sheet 模型判定 ---
    _, t02 = load_csv(XLSX_DUMP / "02_测试集_验证集.csv")
    model_decisions = Counter(r.get("模型判定", "") for r in t02)
    split = Counter(r.get("是否为测试集", "") for r in t02)
    out += [
        "## 8. 02 测试集+验证集 (1339 例全量, 仅 28 例与 .sdpc 重合)\n",
        "**模型判定分布**",
        "| 模型判定 | 例数 |",
        "|---|---|",
    ]
    for k, v in model_decisions.most_common():
        out.append(f"| {k!r} | {v} |")
    out += ["", "**测试/验证集划分**", "| 划分 | 例数 |", "|---|---|"]
    for k, v in split.most_common():
        out.append(f"| {k!r} | {v} |")
    out.append("")

    # --- .sdpc 元信息 ---
    if SDPC_META.exists():
        _, sm = load_csv(SDPC_META)
        sizes = [int(r["file_size_bytes"]) for r in sm]
        out += [
            "## 9. .sdpc 文件元信息(来自 sdpc_inspect.py 跑出的 500 行 metadata)\n",
            f"- 文件数: {len(sm)}",
            f"- 总大小: {sum(sizes)/1e9:.2f} GB",
            f"- 单文件: 最小 {min(sizes)/1e6:.1f} MB / 中位 {sorted(sizes)[len(sizes)//2]/1e6:.1f} MB / 最大 {max(sizes)/1e6:.1f} MB",
            "",
            "**扫描仪/版本/倍率**",
            "| 维度 | 取值 |",
            "|---|---|",
        ]
        for k in ["version", "scanner_model", "scanner_serial", "rate", "slice_format"]:
            vals = Counter(r.get(k, "") for r in sm)
            top = ", ".join(f"{v}({n})" for v, n in vals.most_common(3))
            out.append(f"| {k} | {top} |")
        out.append("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
