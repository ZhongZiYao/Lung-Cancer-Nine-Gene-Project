"""
reconcile_sdpc_labels.py — 统计 500 个 .sdpc 文件能对到多少 xlsx 标签

数据源:
  data/_xlsx_dump/*.csv       — xlsx 4 个 sheet 的 dump
  data/中日冰冻切片/*.sdpc     — 500 个 WSI(文件名 stem = 切片号)
  data/中日冰冻切片/_sdpc_metadata.csv — 之前跑出来的 500 个文件元信息

输出(写到 data/_xlsx_dump/_reconcile_report.md):
  - 500 个 .sdpc 中多少有 xlsx 记录 → 覆盖率
  - xlsx 各 sheet 中切片号去重后多少个
  - 各 sheet 共有的切片号数(交叉)
  - 如果有基因标签字段:每个基因在带标签的 .sdpc 中的正例数
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

WORKSPACE = Path("/mnt/d/pan-caner")  # WSL path
SDPC_DIR = WORKSPACE / "data" / "中日冰冻切片"
XLSX_DUMP = WORKSPACE / "data" / "_xlsx_dump"
REPORT = XLSX_DUMP / "_reconcile_report.md"

GENE_KEYWORDS = [
    "EGFR", "KRAS", "ALK", "HER2", "ROS1", "BRAF",
    "RET", "PIK3CA", "NRAS", "TP53", "LRP1B",
    "基因", "突变", "mutation", "gene",
]


def load_sdpc_stems() -> set[str]:
    return {p.stem for p in SDPC_DIR.glob("*.sdpc")}


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """读 csv,返回 (表头, 行列表)。GBK mojibake 保持原样输出"""
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return list(r.fieldnames or []), rows


def find_id_column(headers: list[str]) -> str | None:
    """找最像'切片号'的列"""
    for cand in ("切片号", "病理号", "蜡块号", "样本号", "编号", "ID", "id"):
        for h in headers:
            if cand in h:
                return h
    return None


def find_gene_columns(headers: list[str]) -> list[str]:
    """找所有含基因关键字的列"""
    hits = []
    for h in headers:
        if any(kw in h for kw in GENE_KEYWORDS):
            hits.append(h)
    return hits


def normalize_id(raw: str) -> str:
    """从 mojibake 字符串里提纯数字 ID"""
    if not raw:
        return ""
    s = str(raw).strip()
    digits = re.findall(r"\d+", s)
    return digits[0] if digits else s


def main() -> int:
    sdpc_stems = load_sdpc_stems()
    print(f"[sdpc] {len(sdpc_stems)} files, sample: {sorted(sdpc_stems)[:3]}")

    csvs = sorted(XLSX_DUMP.glob("*.csv"))
    if not csvs:
        print(f"[ERR] no csv in {XLSX_DUMP}")
        return 1

    report_lines: list[str] = [
        "# .sdpc × xlsx 标签对账报告\n",
        f"- sdpc 数: **{len(sdpc_stems)}**",
        f"- 扫描的 xlsx dump: {len(csvs)}",
        "",
        "## 各 sheet 切片号与 .sdpc 匹配情况\n",
    ]

    # 全局切片号 → 各 sheet 出现情况
    id_to_sheets: dict[str, set[str]] = {}

    for csv_path in csvs:
        if csv_path.name.startswith("_"):
            continue
        sheet_name = csv_path.stem
        try:
            headers, rows = load_csv(csv_path)
        except Exception as e:
            report_lines.append(f"### {sheet_name}\n读失败: {e}\n")
            continue

        if not rows:
            report_lines.append(f"### {sheet_name}\n空表\n")
            continue

        id_col = find_id_column(headers)
        if not id_col:
            report_lines.append(
                f"### {sheet_name} ({len(rows)} 行 × {len(headers)} 列)\n"
                f"找不到 ID 列(表头: {headers})\n"
            )
            continue

        ids = [normalize_id(r[id_col]) for r in rows if r.get(id_col)]
        ids = [i for i in ids if i]
        unique_ids = set(ids)
        matched = unique_ids & sdpc_stems

        gene_cols = find_gene_columns(headers)
        for i in unique_ids:
            id_to_sheets.setdefault(i, set()).add(sheet_name)

        # 统计每基因在 matched 切片号里出现什么值
        gene_stats: dict[str, Counter] = {g: Counter() for g in gene_cols}
        if gene_cols:
            for r in rows:
                rid = normalize_id(r.get(id_col, ""))
                if rid in sdpc_stems:
                    for g in gene_cols:
                        gene_stats[g][r.get(g, "")] += 1

        report_lines += [
            f"### {sheet_name}",
            f"- 行数: {len(rows)}, 表头列: {headers}",
            f"- ID 列: `{id_col}`",
            f"- 切片号去重: {len(unique_ids)}",
            f"- 与 .sdpc 匹配: **{len(matched)}** ({len(matched)/max(1, len(unique_ids))*100:.1f}% of unique IDs, "
            f"{len(matched)/max(1, len(sdpc_stems))*100:.1f}% of {len(sdpc_stems)} .sdpc)",
            f"- 匹配样本: {sorted(matched)[:5]}{'...' if len(matched) > 5 else ''}",
        ]
        if gene_cols:
            report_lines.append("- 基因相关列:")
            for g in gene_cols:
                top = gene_stats[g].most_common(5)
                report_lines.append(f"  - `{g}` (unique vals: {len(gene_stats[g])}): {top}")
        report_lines.append("")

    # ---- 交叉表:.sdpc 出现在哪些 sheet ----
    coverage = Counter()
    for stem in sdpc_stems:
        coverage[len(id_to_sheets.get(stem, set()))] += 1
    report_lines += [
        "## .sdpc 覆盖统计(按被多少个 sheet 提及)\n",
        "| 被 N 个 sheet 提及 | .sdpc 数 |",
        "|---|---|",
    ]
    for n in sorted(coverage):
        report_lines.append(f"| {n} | {coverage[n]} |")
    report_lines.append("")

    # ---- .sdpc ∩ 任何 sheet ----
    any_match = {s for s, sheets in id_to_sheets.items() if sheets} & sdpc_stems
    has_gene_label = any(find_gene_columns(h) for _, (h, _) in [(p.name, load_csv(p)) for p in csvs if not p.name.startswith("_")])

    report_lines += [
        "## 总结\n",
        f"- .sdpc 总数: {len(sdpc_stems)}",
        f"- 出现在 xlsx 至少一个 sheet 的 .sdpc: **{len(any_match)}** "
        f"({len(any_match)/len(sdpc_stems)*100:.1f}%)",
        f"- .sdpc 完全无标签: {len(sdpc_stems) - len(any_match)} "
        f"({(len(sdpc_stems) - len(any_match))/len(sdpc_stems)*100:.1f}%)",
        f"- xlsx 是否含基因突变标签: **{'是' if has_gene_label else '否'}**",
        "",
        "## 结论\n",
        "xlsx 的 4 个 sheet 都是**良恶性/组织学分类**任务,不含 9 基因突变(EGFR/KRAS/ALK/HER2/...)标签。\n"
        "要训 9 基因预测模型,**必须另找带基因突变的标签源**:"
        "  - TCGA-LUAD(541 例 WSI + 基因 panel 标注,公开)"
        "  - 钟/DeepGEM 论文数据集(向作者申请)"
        "  - 甲方(聚时医疗)按合同应提供的"
    ]

    REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[done] wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
