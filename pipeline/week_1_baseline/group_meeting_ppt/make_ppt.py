"""
Week-1 Baseline 组会汇报 PPT
肺癌 9 基因突变预测 — 公开数据集进展、挑战、下一步
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# -------- 路径 --------
BASE = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\pipeline\week_1_baseline\group_meeting_ppt")
IMG = BASE / "imgs"
OUT = BASE / "九基因baseline_组会汇报_2026-07-24.pptx"

# -------- 配色（Sharp + 学术深蓝 + 暖橙强调） --------
PRIMARY  = RGBColor(0x0F, 0x2A, 0x4A)   # 深海军蓝 — 标题/主色
SECONDARY = RGBColor(0x4A, 0x5A, 0x6A)  # 深灰蓝 — 正文
ACCENT   = RGBColor(0xE0, 0x6A, 0x2C)   # 暖橙 — 强调/数字
LIGHT    = RGBColor(0x9A, 0xB5, 0xC9)   # 浅蓝灰 — 副标题
BG       = RGBColor(0xF7, 0xF8, 0xFA)   # 极浅灰 — 背景
RULE     = RGBColor(0xD7, 0xDC, 0xE3)   # 浅灰 — 分隔线
SUCCESS  = RGBColor(0x1F, 0x8A, 0x4D)   # 绿 — 高频组
WARNING  = RGBColor(0xC4, 0x39, 0x2E)   # 红 — 低频组/失败

FONT_EN = "Arial"
FONT_CN = "Microsoft YaHei"

# -------- PPT 16:9 --------
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

blank = prs.slide_layouts[6]


# ============================================================
# 工具函数
# ============================================================
def add_rect(slide, x, y, w, h, color, line=False):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if not line:
        s.line.fill.background()
    else:
        s.line.color.rgb = color
        s.line.width = Pt(0.5)
    s.shadow.inherit = False
    return s


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=SECONDARY,
             align=PP_ALIGN.LEFT, font_cn=FONT_CN, font_en=FONT_EN, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    # 处理多行
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        if i == 0:
            par = p
        else:
            par = tf.add_paragraph()
        par.alignment = align
        run = par.add_run()
        run.text = line
        run.font.name = font_en
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        # 中文字体
        rPr = run._r.get_or_add_rPr()
        eaFont = etree.SubElement(rPr, qn("a:ea"), nsmap={"a": "http://schemas.openxmlformats.org/drawingml/2006/main"})
        # 通过 XML 设置东亚字体
        for child in list(rPr):
            if child.tag == qn("a:ea"):
                rPr.remove(child)
        ea = etree.SubElement(rPr, qn("a:ea"))
        ea.set("typeface", font_cn)
    return box


def add_runs(slide, x, y, w, h, runs, *, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=None):
    """runs: list of (text, dict) 支持富文本"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    for i, (text, opts) in enumerate(runs):
        if i == 0:
            par = tf.paragraphs[0]
        else:
            par = tf.add_paragraph()
        par.alignment = opts.get("align", align)
        if line_spacing is not None:
            par.line_spacing = line_spacing
        # text 可以是 list 表示一次段内多次 run
        if isinstance(text, str):
            segs = [(text, opts)]
        else:
            segs = text
        for seg_text, seg_opts in segs:
            run = par.add_run()
            run.text = seg_text
            run.font.name = seg_opts.get("font_en", FONT_EN)
            run.font.size = Pt(seg_opts.get("size", 14))
            run.font.bold = seg_opts.get("bold", False)
            run.font.color.rgb = seg_opts.get("color", SECONDARY)
            rPr = run._r.get_or_add_rPr()
            for child in list(rPr):
                if child.tag == qn("a:ea"):
                    rPr.remove(child)
            ea = etree.SubElement(rPr, qn("a:ea"))
            ea.set("typeface", seg_opts.get("font_cn", FONT_CN))
    return box


def add_bullet(slide, x, y, w, h, items, *, size=14, color=SECONDARY,
               bullet="•", spacing=0.25, indent_size_diff=-2):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        if i == 0:
            par = tf.paragraphs[0]
        else:
            par = tf.add_paragraph()
        par.alignment = PP_ALIGN.LEFT
        par.space_after = Pt(spacing)
        # bullet
        rb = par.add_run()
        rb.text = f"{bullet}  "
        rb.font.name = FONT_EN
        rb.font.size = Pt(size)
        rb.font.color.rgb = ACCENT
        rb.font.bold = True
        # content
        if isinstance(item, str):
            segs = [(item, {})]
        else:
            segs = item
        for seg_text, seg_opts in segs:
            r = par.add_run()
            r.text = seg_text
            r.font.name = seg_opts.get("font_en", FONT_EN)
            r.font.size = Pt(seg_opts.get("size", size))
            r.font.bold = seg_opts.get("bold", False)
            r.font.color.rgb = seg_opts.get("color", color)
            rPr = r._r.get_or_add_rPr()
            for child in list(rPr):
                if child.tag == qn("a:ea"):
                    rPr.remove(child)
            ea = etree.SubElement(rPr, qn("a:ea"))
            ea.set("typeface", seg_opts.get("font_cn", FONT_CN))
    return box


def add_page_badge(slide, idx, total):
    # 底部右侧页码 (留 SH-0.28, 高度 0.22)
    add_text(slide, SW - Inches(1.0), SH - Inches(0.32),
             Inches(0.9), Inches(0.24),
             f"{idx:02d} / {total:02d}",
             size=9, color=LIGHT, align=PP_ALIGN.RIGHT)
    # 左下角小标识
    add_text(slide, Inches(0.4), SH - Inches(0.32),
             Inches(7), Inches(0.24),
             "9-Gene Lung Adenocarcinoma Mutation Prediction · Week 1",
             size=8, color=LIGHT, align=PP_ALIGN.LEFT)


def add_header(slide, title, kicker=None):
    """统一头部：左侧色块 + 标题"""
    # 左侧色块
    add_rect(slide, Inches(0.4), Inches(0.42), Inches(0.12), Inches(0.6), ACCENT)
    # kicker (小字)
    if kicker:
        add_text(slide, Inches(0.65), Inches(0.4), Inches(8), Inches(0.25),
                 kicker, size=10, color=ACCENT, bold=True)
        add_text(slide, Inches(0.65), Inches(0.65), Inches(11), Inches(0.5),
                 title, size=24, bold=True, color=PRIMARY)
    else:
        add_text(slide, Inches(0.65), Inches(0.5), Inches(11), Inches(0.5),
                 title, size=26, bold=True, color=PRIMARY)
    # 分隔线
    add_rect(slide, Inches(0.4), Inches(1.25), SW - Inches(0.8), Emu(9525), RULE)


def add_kpi_card(slide, x, y, w, h, label, value, sub=None, value_color=ACCENT):
    add_rect(slide, x, y, w, h, BG, line=True)
    # value 大数字
    add_text(slide, x, y + Inches(0.18), w, Inches(0.7),
             value, size=34, bold=True, color=value_color, align=PP_ALIGN.CENTER)
    # label
    add_text(slide, x, y + Inches(0.95), w, Inches(0.3),
             label, size=12, color=SECONDARY, align=PP_ALIGN.CENTER, bold=True)
    if sub:
        add_text(slide, x, y + Inches(1.25), w, Inches(0.3),
                 sub, size=10, color=LIGHT, align=PP_ALIGN.CENTER)


def add_status_chip(slide, x, y, w, h, text, color):
    s = add_rect(slide, x, y, w, h, color)
    add_text(slide, x, y, w, h, text, size=10, bold=True,
             color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)


def set_bg(slide, color=BG):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    # 放到最底层
    spTree = bg._element.getparent()
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)


TOTAL = 8


# ============================================================
# Page 1 — Cover
# ============================================================
def page_cover():
    s = prs.slides.add_slide(blank)
    set_bg(s, PRIMARY)
    # 顶部装饰条
    add_rect(s, 0, 0, SW, Inches(0.18), ACCENT)
    # 装饰大圆
    c1 = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.5), Inches(0.5),
                            Inches(3.0), Inches(3.0))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(0x16, 0x39, 0x5E)
    c1.line.fill.background()
    c2 = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(11.5), Inches(2.5),
                            Inches(2.2), Inches(2.2))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(0x1F, 0x4D, 0x7A)
    c2.line.fill.background()
    # kicker
    add_text(s, Inches(0.8), Inches(2.0), Inches(10), Inches(0.4),
             "WEEK 1  ·  GROUP MEETING  ·  2026-07-24",
             size=12, color=ACCENT, bold=True, font_en=FONT_EN)
    # 主标题
    add_text(s, Inches(0.8), Inches(2.5), Inches(10.5), Inches(1.4),
             "肺癌九基因突变预测",
             size=44, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_text(s, Inches(0.8), Inches(3.5), Inches(10.5), Inches(1.0),
             "公开数据集基线实验 · 算力挑战 · 下一步",
             size=24, color=LIGHT)
    # 底部分隔线
    add_rect(s, Inches(0.8), Inches(5.6), Inches(2.5), Emu(28575), ACCENT)
    # 副信息 (汇报人一行 + 方向一行 + backbone 一行) — 用 add_runs 的 segs 列表
    add_text(s, Inches(0.8), Inches(5.78), Inches(11.5), Inches(0.4),
             "汇报人  Zhong", size=12, color=LIGHT, bold=True)
    add_text(s, Inches(0.8), Inches(6.18), Inches(11.5), Inches(0.4),
             "方向  病理图像 × 弱监督多标签突变预测 (TCGA-LUAD → SDPC)",
             size=12, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_text(s, Inches(0.8), Inches(6.58), Inches(11.5), Inches(0.4),
             "Backbone: UNI (frozen, 1024-dim)   ·   MIL: Gated Attention + DeepGEM Prototype",
             size=11, color=LIGHT)


# ============================================================
# Page 2 — TOC
# ============================================================
def page_toc():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "本次汇报大纲", "OUTLINE")

    items = [
        ("01", "数据规模与处理流程", "83 case · 9 基因 · patch 极度不均"),
        ("02", "V1: EGFR-only 5-fold baseline", "单基因 OOF AUC = 0.584"),
        ("03", "V2: EGFR 固定 70/15/15 划分", "val 0.550 / test 0.318 (崩)"),
        ("04", "V3: 9 基因 5-fold 完整 baseline", "mean AUC = 0.532"),
        ("05", "算力瓶颈与数据噪声", "小样本 + 多中心 + patch 噪声"),
        ("06", "下周计划 (DeepGEM 思路)", "Tissue mask + PartialLoss 标签消歧"),
    ]
    y0 = Inches(1.7)
    for i, (num, title, sub) in enumerate(items):
        yi = y0 + Inches(i * 0.78)
        # 编号方块
        add_rect(s, Inches(0.8), yi, Inches(0.8), Inches(0.6), PRIMARY)
        add_text(s, Inches(0.8), yi, Inches(0.8), Inches(0.6), num,
                 size=20, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # 标题
        add_text(s, Inches(1.85), yi + Inches(0.02), Inches(6.5), Inches(0.36),
                 title, size=18, bold=True, color=PRIMARY)
        add_text(s, Inches(1.85), yi + Inches(0.36), Inches(8), Inches(0.3),
                 sub, size=12, color=SECONDARY)
        # 右侧细线
        if i < len(items) - 1:
            add_rect(s, Inches(1.85), yi + Inches(0.7),
                     SW - Inches(2.5), Emu(6350), RULE)
    add_page_badge(s, 2, TOTAL)


# ============================================================
# Page 3 — 数据规模与处理流程
# ============================================================
def page_data():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "数据规模与处理流程", "01  ·  DATA PIPELINE")

    # 顶部 4 张 KPI 卡片
    y0 = Inches(1.55)
    kw = Inches(2.95)
    gap = Inches(0.15)
    x0 = Inches(0.4)
    kpis = [
        ("总有标签 case", "83", "TCGA-LUAD subset", PRIMARY),
        ("总 patch 数", "407,666", "切自 9 SM × 多切片", PRIMARY),
        ("平均 patch / case", "288", "极度不均 45 ~ 24,222", ACCENT),
        ("9 基因阳性 case 均值", "13 / 83", "范围 2 ~ 56 例", PRIMARY),
    ]
    for i, (label, value, sub, vc) in enumerate(kpis):
        xi = x0 + (kw + gap) * i
        add_kpi_card(s, xi, y0, kw, Inches(1.55), label, value, sub, value_color=vc)

    # 处理流程 (横向箭头图)
    flow_y = Inches(3.45)
    steps = [
        ("DICOM\nWSI", "~200MB/series", PRIMARY),
        ("Patch\n256×256", "Otsu+组织过滤", PRIMARY),
        ("UNI\nFeature", "1024-dim", PRIMARY),
        ("Bag\nper-SM", "Gated Attn", PRIMARY),
        ("5-Fold\nTrain", "9 head BCE", ACCENT),
    ]
    sw = Inches(2.0)
    sgap = Inches(0.35)
    total_w = sw * len(steps) + sgap * (len(steps) - 1)
    start_x = (SW - total_w) / 2
    for i, (t, sub, c) in enumerate(steps):
        xi = start_x + (sw + sgap) * i
        add_rect(s, xi, flow_y, sw, Inches(1.0), c)
        add_text(s, xi, flow_y + Inches(0.15), sw, Inches(0.45), t,
                 size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, xi, flow_y + Inches(0.6), sw, Inches(0.35), sub,
                 size=10, color=LIGHT, align=PP_ALIGN.CENTER)
        # 箭头
        if i < len(steps) - 1:
            ax = xi + sw + Inches(0.05)
            arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                     ax, flow_y + Inches(0.4),
                                     sgap - Inches(0.1), Inches(0.2))
            arr.fill.solid()
            arr.fill.fore_color.rgb = LIGHT
            arr.line.fill.background()

    # 9 基因阳性率分布 (左下) + 处理耗时 (右下)
    panel_y = Inches(4.85)
    panel_h = Inches(2.15)
    # 左
    add_rect(s, Inches(0.4), panel_y, Inches(6.2), panel_h, BG, line=True)
    add_text(s, Inches(0.6), panel_y + Inches(0.08), Inches(6), Inches(0.3),
             "9 基因阳性 case 数 (从高到低)", size=12, bold=True, color=PRIMARY)
    # 每行 1 段, 段内多 run — 缩短文字避免 wrap
    add_runs(s, Inches(0.6), panel_y + Inches(0.45), Inches(5.9), Inches(0.32), [
        ([("TP53 ", {"size": 12, "bold": True, "color": SUCCESS}),
          ("56", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  KRAS ", {"size": 12, "color": SUCCESS}),
          ("22", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  EGFR ", {"size": 12, "color": SUCCESS}),
          ("13", {"size": 12, "bold": True, "color": ACCENT})], {})
    ])
    add_runs(s, Inches(0.6), panel_y + Inches(0.82), Inches(5.9), Inches(0.32), [
        ([("ALK ", {"size": 12, "color": SUCCESS}),
          ("11", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  BRAF ", {"size": 12, "color": SUCCESS}),
          ("9", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  PIK3CA ", {"size": 12, "color": WARNING}),
          ("6", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  ERBB2 ", {"size": 12, "color": WARNING}),
          ("6", {"size": 12, "bold": True, "color": ACCENT})], {})
    ])
    add_runs(s, Inches(0.6), panel_y + Inches(1.19), Inches(5.9), Inches(0.32), [
        ([("ROS1 ", {"size": 12, "color": WARNING}),
          ("5", {"size": 12, "bold": True, "color": ACCENT}),
          ("  ·  NRAS ", {"size": 12, "color": WARNING}),
          ("2", {"size": 12, "bold": True, "color": ACCENT})], {})
    ])
    add_text(s, Inches(0.6), panel_y + Inches(1.55), Inches(5.9), Inches(0.55),
             "高频组 (≥9 例, 5 基因): TP53 / KRAS / EGFR / ALK / BRAF\n"
             "低频组 (<9 例, 4 基因): PIK3CA / ERBB2 / ROS1 / NRAS",
             size=10, color=SECONDARY)

    # 右
    add_rect(s, Inches(6.75), panel_y, Inches(6.2), panel_h, BG, line=True)
    add_text(s, Inches(6.95), panel_y + Inches(0.08), Inches(6), Inches(0.3),
             "处理瓶颈 (本机 RTX 4090 / 24GB)", size=12, bold=True, color=PRIMARY)
    add_bullet(s, Inches(6.95), panel_y + Inches(0.45), Inches(5.8), Inches(1.65), [
        [("切 patch: ", {"size": 10, "bold": True, "color": ACCENT}),
         ("83 case / ~2.5 h  (DICOM 解码 + multi-slice 切片)", {"size": 10, "color": SECONDARY})],
        [("UNI 抽特征: ", {"size": 10, "bold": True, "color": ACCENT}),
         ("288 patch × 1024 / case  ·  83 case  ·  约 4 h", {"size": 10, "color": SECONDARY})],
        [("5-fold 训练: ", {"size": 10, "bold": True, "color": ACCENT}),
         ("9 head × 50 epoch / fold  ·  约 8-10 h", {"size": 10, "color": SECONDARY})],
        [("总耗时: ", {"size": 10, "bold": True, "color": WARNING}),
         ("~17 h 全 pipeline (单机单卡)", {"size": 10, "color": WARNING, "bold": True})],
    ], spacing=3)

    add_page_badge(s, 3, TOTAL)


# ============================================================
# Page 4 — V1: EGFR-only 5-fold
# ============================================================
def page_v1():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "V1 · EGFR-only 5-fold Baseline", "02  ·  EXPERIMENT")

    # 左侧数字
    add_text(s, Inches(0.6), Inches(1.55), Inches(5.5), Inches(0.3),
             "目标", size=11, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(1.85), Inches(5.5), Inches(0.5),
             "单基因 (EGFR) 验证 pipeline 跑通",
             size=18, bold=True, color=PRIMARY)

    add_text(s, Inches(0.6), Inches(2.5), Inches(5.5), Inches(0.3),
             "配置", size=11, color=ACCENT, bold=True)
    add_bullet(s, Inches(0.6), Inches(2.8), Inches(5.5), Inches(2.0), [
        [("数据: ", {"size": 12, "bold": True}), ("83 case · EGFR+ 13 / EGFR- 70", {"size": 12})],
        [("切分: ", {"size": 12, "bold": True}), ("5-fold StratifiedKFold (by any_positive)", {"size": 12})],
        [("模型: ", {"size": 12, "bold": True}), ("UNI(冻结) + EGFROnlyModel, hidden=128, dropout=0.5", {"size": 12})],
        [("Loss: ", {"size": 12, "bold": True}), ("BCE + pos_weight=5.0 (clip 10.0)", {"size": 12})],
        [("训练: ", {"size": 12, "bold": True}), ("AdamW lr=1e-4 · early stop patience=15", {"size": 12})],
    ], spacing=3)

    # 大数字 + 5 fold 小数字
    add_rect(s, Inches(0.6), Inches(5.0), Inches(5.5), Inches(2.0), PRIMARY)
    add_text(s, Inches(0.6), Inches(5.15), Inches(5.5), Inches(0.3),
             "OOF AUC", size=12, color=LIGHT, align=PP_ALIGN.CENTER, bold=True)
    add_text(s, Inches(0.6), Inches(5.45), Inches(5.5), Inches(0.9),
             "0.584", size=56, bold=True, color=ACCENT,
             align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.6), Inches(6.45), Inches(5.5), Inches(0.5),
             "5 fold val:  0.700 / 0.738 / 0.429 / 0.769 / 0.821",
             size=11, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)

    # 右侧 — 5 fold 数字可视化
    add_text(s, Inches(6.6), Inches(1.55), Inches(6), Inches(0.3),
             "5-Fold Val AUC (问题)", size=14, bold=True, color=PRIMARY)
    folds = [
        ("Fold 1", 0.700, SUCCESS),
        ("Fold 2", 0.738, SUCCESS),
        ("Fold 3", 0.429, WARNING),
        ("Fold 4", 0.769, SUCCESS),
        ("Fold 5", 0.821, SUCCESS),
    ]
    y0 = Inches(1.95)
    bw = Inches(6.0)
    bh = Inches(0.5)
    bgap = Inches(0.12)
    for i, (name, val, color) in enumerate(folds):
        yi = y0 + (bh + bgap) * i
        # 标签
        add_text(s, Inches(6.6), yi, Inches(0.9), bh, name,
                 size=12, color=SECONDARY, anchor=MSO_ANCHOR.MIDDLE, bold=True)
        # 进度条背景
        bar_x = Inches(7.55)
        bar_w = Inches(4.2)
        add_rect(s, bar_x, yi + Inches(0.12), bar_w, Inches(0.28), BG, line=True)
        # 进度条
        fill_w = Emu(int(bar_w * val))
        add_rect(s, bar_x, yi + Inches(0.12), fill_w, Inches(0.28), color)
        # 数字
        add_text(s, bar_x + bar_w + Inches(0.1), yi, Inches(0.7), bh,
                 f"{val:.3f}", size=12, color=color, bold=True,
                 anchor=MSO_ANCHOR.MIDDLE)

    # 教训
    add_rect(s, Inches(6.6), Inches(5.55), Inches(6.3), Inches(1.4), BG, line=True)
    add_text(s, Inches(6.75), Inches(5.62), Inches(6), Inches(0.28),
             "Lesson", size=10, color=ACCENT, bold=True)
    add_text(s, Inches(6.75), Inches(5.85), Inches(6.1), Inches(1.05),
             "5 fold AUC 极不稳定 (0.43-0.82), fold3 阳性仅 2 例, 模型对小样本极敏感。\n"
             "→ 固定 70/15/15 划分不可行 → 尝试 LOO → 全部早停, 证明 83 case 上 EGFR 太稀。",
             size=10, color=SECONDARY)

    add_page_badge(s, 4, TOTAL)


# ============================================================
# Page 5 — V2: EGFR 70/15/15
# ============================================================
def page_v2():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "V2 · EGFR 固定 70/15/15 划分", "03  ·  EXPERIMENT")

    # 左 — 配置 + 数字
    add_text(s, Inches(0.6), Inches(1.55), Inches(5.5), Inches(0.3),
             "动机", size=11, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(1.85), Inches(5.5), Inches(0.5),
             "5-fold 波动大 → 改固定 train/val/test", size=16, bold=True, color=PRIMARY)

    add_text(s, Inches(0.6), Inches(2.55), Inches(5.5), Inches(0.3),
             "配置", size=11, color=ACCENT, bold=True)
    add_bullet(s, Inches(0.6), Inches(2.85), Inches(5.5), Inches(2.0), [
        [("切分: ", {"size": 12, "bold": True}), ("train 58 (EGFR+ 9) / val 12 (2) / test 13 (2)", {"size": 12})],
        [("按 ", {"size": 12}), ("any_9gene_positive", {"size": 12, "font_cn": "Consolas"}),
         (" 分层", {"size": 12})],
        [("模型: ", {"size": 12, "bold": True}), ("hidden=64 / dropout=0.5 / wd=1e-4 (更小)", {"size": 12})],
        [("训练: ", {"size": 12, "bold": True}), ("lr=1e-4 · epochs=80 · early stop @ ep26", {"size": 12})],
    ], spacing=3)

    # 结果三卡
    y_res = Inches(5.05)
    cw = Inches(1.7)
    cgap = Inches(0.1)
    cx0 = Inches(0.6)
    cards = [
        ("Train EGFR+", "9", LIGHT, PRIMARY),
        ("Val AUC", "0.550", ACCENT, PRIMARY),
        ("Test AUC", "0.318", WARNING, PRIMARY),
    ]
    for i, (l, v, c, bgc) in enumerate(cards):
        xi = cx0 + (cw + cgap) * i
        add_rect(s, xi, y_res, cw, Inches(1.5), BG, line=True)
        add_text(s, xi, y_res + Inches(0.15), cw, Inches(0.3), l,
                 size=10, color=SECONDARY, align=PP_ALIGN.CENTER, bold=True)
        add_text(s, xi, y_res + Inches(0.55), cw, Inches(0.7), v,
                 size=28, bold=True, color=c, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.6), Inches(6.65), Inches(5.5), Inches(0.3),
             "test 仅 2 例 EGFR+ → AUC 估算极不可信",
             size=10, color=ACCENT, bold=True, align=PP_ALIGN.LEFT)

    # 右 — 图 + 教训
    add_text(s, Inches(6.6), Inches(1.55), Inches(6), Inches(0.3),
             "V2 ROC (val / test)", size=14, bold=True, color=PRIMARY)
    s.shapes.add_picture(str(IMG / "v2_roc.png"),
                         Inches(6.6), Inches(1.9),
                         width=Inches(6.3), height=Inches(2.5))
    add_text(s, Inches(6.6), Inches(4.45), Inches(6), Inches(0.3),
             "V2 Prediction Distribution", size=14, bold=True, color=PRIMARY)
    s.shapes.add_picture(str(IMG / "v2_pred_dist.png"),
                         Inches(6.6), Inches(4.8),
                         width=Inches(6.3), height=Inches(2.1))

    # 底部教训条
    add_rect(s, Inches(0.4), Inches(7.0), SW - Inches(0.8), Inches(0.3), PRIMARY)
    add_text(s, Inches(0.6), Inches(7.0), Inches(12), Inches(0.3),
             "Lesson:  小样本 + 固定 split = 死局  ·  必须 5-fold 才能拿到像样 AUC 估计  ·  推回 5-fold 路线",
             size=11, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True,
             anchor=MSO_ANCHOR.MIDDLE)


# ============================================================
# Page 6 — V3: 9-gene 5-fold
# ============================================================
def page_v3():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "V3 · 9 基因 5-fold 完整 Baseline", "04  ·  EXPERIMENT")

    # 顶部一行 — 架构摘要 + 大数字
    add_text(s, Inches(0.6), Inches(1.5), Inches(7.5), Inches(0.3),
             "架构  (GAMIL + DeepGEM 融合)", size=11, color=ACCENT, bold=True)
    add_bullet(s, Inches(0.6), Inches(1.8), Inches(7.5), Inches(1.0), [
        [("Backbone  ", {"size": 11, "bold": True, "color": SECONDARY}),
         ("UNI (frozen, 1024-dim)  →  Linear(1024→512)  +  LN  +  GELU  +  Dropout(0.5)", {"size": 11})],
        [("Stage-1  ", {"size": 11, "bold": True, "color": SECONDARY}),
         ("per-SM Gated Attention (Ilse 2018)  →  SM vector", {"size": 11})],
        [("Stage-2  ", {"size": 11, "bold": True, "color": SECONDARY}),
         ("case-level Gated Attention  +  9 head BCE  +  DeepGEM prototype 辅助", {"size": 11})],
    ], spacing=2)

    # 右侧大数字
    add_rect(s, Inches(8.4), Inches(1.5), Inches(4.5), Inches(1.6), PRIMARY)
    add_text(s, Inches(8.4), Inches(1.6), Inches(4.5), Inches(0.3),
             "9-Gene Mean AUC", size=11, color=LIGHT, align=PP_ALIGN.CENTER, bold=True)
    add_text(s, Inches(8.4), Inches(1.85), Inches(4.5), Inches(0.85),
             "0.532", size=52, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(s, Inches(8.4), Inches(2.7), Inches(4.5), Inches(0.35),
             "高频组 0.529  |  低频组 0.536",
             size=11, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)

    # 5 fold val AUC 进度条
    add_text(s, Inches(0.6), Inches(3.1), Inches(6), Inches(0.3),
             "5-Fold Best Val AUC (mean over 9 genes)", size=12, bold=True, color=PRIMARY)
    folds = [
        ("Fold 1", 0.5785),
        ("Fold 2", 0.6658),
        ("Fold 3", 0.5368),
        ("Fold 4", 0.6225),
        ("Fold 5", 0.6464),
    ]
    y0 = Inches(3.45)
    bh = Inches(0.32)
    bgap = Inches(0.08)
    bar_x = Inches(1.55)
    bar_w = Inches(5.0)
    for i, (name, val) in enumerate(folds):
        yi = y0 + (bh + bgap) * i
        add_text(s, Inches(0.6), yi, Inches(0.9), bh, name,
                 size=11, color=SECONDARY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        add_rect(s, bar_x, yi + Inches(0.05), bar_w, Inches(0.24), BG, line=True)
        add_rect(s, bar_x, yi + Inches(0.05),
                 Emu(int(bar_w * val)), Inches(0.24), ACCENT)
        add_text(s, bar_x + bar_w + Inches(0.1), yi, Inches(0.7), bh,
                 f"{val:.4f}", size=11, color=PRIMARY, bold=True,
                 anchor=MSO_ANCHOR.MIDDLE)

    # 右侧 — per-gene bar 图
    add_text(s, Inches(8.4), Inches(3.1), Inches(4.5), Inches(0.3),
             "Per-Gene AUC", size=12, bold=True, color=PRIMARY)
    s.shapes.add_picture(str(IMG / "auc_bar.png"),
                         Inches(8.4), Inches(3.4),
                         width=Inches(4.5), height=Inches(1.95))

    # 5 fold 收尾 y = 3.5 + 5*(0.36+0.1) - 0.1 = 3.5+2.2-0.1 = 5.6
    # auc_bar 收尾 y = 3.4 + 1.95 = 5.35

    # 底部 — PR / ROC / pred dist 3 张图 (从 5.65 开始, 高 1.45 → 7.10)
    bottom_y = Inches(5.7)
    th = Inches(1.35)
    tw = Inches(4.0)
    tgap = Inches(0.2)
    add_text(s, Inches(0.6), Inches(5.45), Inches(4), Inches(0.22),
             "ROC (9 子图)", size=10, color=SECONDARY, bold=True)
    s.shapes.add_picture(str(IMG / "roc_curves.png"),
                         Inches(0.6), bottom_y, width=tw, height=th)
    add_text(s, Inches(0.6) + tw + tgap, Inches(5.45), Inches(4), Inches(0.22),
             "PR Curves", size=10, color=SECONDARY, bold=True)
    s.shapes.add_picture(str(IMG / "pr_curves.png"),
                         Inches(0.6) + tw + tgap, bottom_y, width=tw, height=th)
    add_text(s, Inches(0.6) + 2 * (tw + tgap), Inches(5.45), Inches(4), Inches(0.22),
             "Prediction Distribution", size=10, color=SECONDARY, bold=True)
    s.shapes.add_picture(str(IMG / "pred_dist.png"),
                         Inches(0.6) + 2 * (tw + tgap), bottom_y, width=tw, height=th)
    # 注: page badge 在 SH-0.45=7.05 位置, 这里 7.3 不会冲突


# ============================================================
# Page 7 — 算力 / 噪声 / 挑战
# ============================================================
def page_challenges():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "当前瓶颈  ·  算力  ·  数据噪声", "05  ·  CHALLENGES")

    # 三列卡片
    y0 = Inches(1.6)
    ch = Inches(3.6)
    cw = Inches(4.05)
    cgap = Inches(0.2)
    cols = [
        {
            "head": "算力不足",
            "icon": "GPU",
            "color": WARNING,
            "items": [
                "单卡 RTX 4090 (24GB), 全 pipeline 17h+",
                "83 case 全量切 patch + 抽 UNI 特征 ≈ 6.5h",
                "5-fold × 50 epoch × 9 head ≈ 8-10h",
                "难做 hyperparam sweep / 多 seed 验证",
            ],
        },
        {
            "head": "数据缺陷",
            "icon": "DATA",
            "color": ACCENT,
            "items": [
                "TCGA-LUAD 公开数据 421 case 中 83 case 接入",
                "阳性率极不均: NRAS 2 例, TP53 56 例",
                "TCGA 多中心多扫描仪 (Aperio/浜松/北极星)",
                "frozen vs FFPE 染色差异 → 域漂移",
            ],
        },
        {
            "head": "Patch 噪声",
            "icon": "WSI",
            "color": PRIMARY,
            "items": [
                "切出来的 patch 含大量背景/淋巴/血管/坏死",
                "Otsu 只挡空白, 不挡正常组织",
                "83 case 上无 patch-level 标签",
                "缺少 CLAM tissue mask / DeepGEM 消歧",
            ],
        },
    ]
    for i, col in enumerate(cols):
        xi = Inches(0.4) + (cw + cgap) * i
        add_rect(s, xi, y0, cw, ch, BG, line=True)
        # 头部色条
        add_rect(s, xi, y0, cw, Inches(0.55), col["color"])
        add_text(s, xi + Inches(0.25), y0, cw, Inches(0.55),
                 col["head"], size=16, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF),
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, xi + cw - Inches(1.0), y0, Inches(0.7), Inches(0.55),
                 col["icon"], size=12, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
        add_bullet(s, xi + Inches(0.2), y0 + Inches(0.7),
                   cw - Inches(0.4), ch - Inches(0.8),
                   col["items"], size=11, spacing=6)

    # 底部 — 对比论文 + 数字
    add_rect(s, Inches(0.4), Inches(5.25), SW - Inches(0.8), Inches(1.75),
             PRIMARY)
    add_text(s, Inches(0.6), Inches(5.32), Inches(6), Inches(0.28),
             "跟论文的差距", size=11, color=ACCENT, bold=True)
    add_runs(s, Inches(0.6), Inches(5.6), Inches(12.1), Inches(0.32), [
        ([("GAMIL 论文:  ", {"size": 11, "bold": True, "color": RGBColor(0xFF, 0xFF, 0xFF)}),
          ("~2000 case · AUC 0.83-0.99", {"size": 11, "color": LIGHT})], {})
    ])
    add_runs(s, Inches(0.6), Inches(5.95), Inches(12.1), Inches(0.32), [
        ([("本项目 V3:  ", {"size": 11, "bold": True, "color": ACCENT}),
          ("83 case · AUC 0.43-0.71   ", {"size": 11, "color": RGBColor(0xFF, 0xFF, 0xFF)}),
          ("差 25-30× 数据量", {"size": 11, "bold": True, "color": WARNING})], {})
    ])
    add_text(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.32),
             "结论: AUC 受限于样本量 / patch 噪声 / 训练不稳定, 不是模型架构问题。",
             size=10, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_text(s, Inches(0.6), Inches(6.65), Inches(12.1), Inches(0.32),
             "→ 下周: tissue mask 砍噪声 + 标签消歧 (DeepGEM 思路)。",
             size=11, color=ACCENT, bold=True)

    add_page_badge(s, 7, TOTAL)


# ============================================================
# Page 8 — 下周计划
# ============================================================
def page_plan():
    s = prs.slides.add_slide(blank)
    set_bg(s)
    add_header(s, "下周计划  ·  DeepGEM 思路落地", "06  ·  NEXT STEPS")

    # 三阶段优先级条
    y0 = Inches(1.55)
    rows = [
        {
            "tag": "P0",
            "color": SUCCESS,
            "title": "Tissue Mask — 砍非肿瘤 patch",
            "desc": "装 CLAM 跑一遍 tissue classification, 砍掉淋巴/血管/基质/坏死 60-80% 噪声 patch",
            "metric": "预期 patch / case  从 1000-20000 → 200-5000",
            "time": "1-2 天",
        },
        {
            "tag": "P1",
            "color": ACCENT,
            "title": "PartialLoss 标签消歧 (DeepGEM)",
            "desc": "移植 loss.py 改 multi-label: 9 prototype × WSI mask × EMA confidence, 训练中修正 patch 软标签",
            "metric": "目标高频组 AUC 0.55 → 0.65+",
            "time": "3-5 天",
        },
        {
            "tag": "P2",
            "color": PRIMARY,
            "title": "Backbone 升级 (CTransPath 768-dim)",
            "desc": "换更轻 backbone 提速 1.5-2×, 释放算力跑多 seed / 5-fold 严格 CV",
            "metric": "训练耗时 -40%, 显存 -30%",
            "time": "2-3 天",
        },
    ]
    for i, r in enumerate(rows):
        yi = y0 + Inches(i * 1.35)
        # 标签
        add_rect(s, Inches(0.4), yi, Inches(0.9), Inches(1.15), r["color"])
        add_text(s, Inches(0.4), yi, Inches(0.9), Inches(1.15), r["tag"],
                 size=28, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        # 内容区
        add_rect(s, Inches(1.3), yi, SW - Inches(1.7), Inches(1.15),
                 BG, line=True)
        add_text(s, Inches(1.5), yi + Inches(0.1), Inches(8), Inches(0.4),
                 r["title"], size=14, bold=True, color=PRIMARY)
        add_text(s, Inches(1.5), yi + Inches(0.5), Inches(11), Inches(0.4),
                 r["desc"], size=10, color=SECONDARY)
        add_text(s, Inches(1.5), yi + Inches(0.85), Inches(8), Inches(0.28),
                 r["metric"], size=10, color=ACCENT, bold=True)
        add_text(s, SW - Inches(1.4), yi + Inches(0.85), Inches(0.9), Inches(0.28),
                 r["time"], size=10, color=LIGHT, bold=True, align=PP_ALIGN.RIGHT)

    # 底部 — 依赖 / 风险
    add_rect(s, Inches(0.4), Inches(5.55), SW - Inches(0.8), Inches(1.45),
             PRIMARY)
    add_text(s, Inches(0.6), Inches(5.62), Inches(6), Inches(0.28),
             "依赖 / 风险", size=11, color=ACCENT, bold=True)
    add_runs(s, Inches(0.6), Inches(5.92), Inches(12.1), Inches(0.32), [
        ([("·  TCGA-LUAD 50 / 83 case 接入;  ", {"size": 10, "color": RGBColor(0xFF, 0xFF, 0xFF)}),
          ("仍需扩大训练集 (钟老师 2221 NSCLC)", {"size": 10, "color": RGBColor(0xFF, 0xFF, 0xFF), "bold": True})], {})
    ])
    add_runs(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.32), [
        ([("·  中日友好 .sdpc 冰冻切片 ", {"size": 10, "color": RGBColor(0xFF, 0xFF, 0xFF)}),
          ("等接入", {"size": 10, "color": WARNING, "bold": True}),
          (";  生产期 tissue classifier 用 SDPC 自标微调 (与 P0 互补)", {"size": 10, "color": RGBColor(0xFF, 0xFF, 0xFF)})], {})
    ])

    add_page_badge(s, 8, TOTAL)


# ============================================================
# 生成
# ============================================================
page_cover()        # 1
page_toc()          # 2
page_data()         # 3
page_v1()           # 4
page_v2()           # 5
page_v3()           # 6
page_challenges()   # 7
page_plan()         # 8

# 给 page 6 补 page_badge (之前 page 6 没调用 add_page_badge)
def add_v3_badge():
    s = prs.slides[5]  # 0-indexed, page 6
    add_text(s, SW - Inches(1.0), SH - Inches(0.32),
             Inches(0.9), Inches(0.24),
             "06 / 08", size=9, color=LIGHT, align=PP_ALIGN.RIGHT)
    add_text(s, Inches(0.4), SH - Inches(0.32),
             Inches(7), Inches(0.24),
             "9-Gene Lung Adenocarcinoma Mutation Prediction · Week 1",
             size=8, color=LIGHT, align=PP_ALIGN.LEFT)
add_v3_badge()

prs.save(str(OUT))
print(f"OK -> {OUT}")
print(f"slides: {len(prs.slides)}")
