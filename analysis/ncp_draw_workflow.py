from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


TEXT = {
    "en": {
        "column_titles": [
            "1. GWA PROXY DEFINITION",
            "2. BREAKPOINT DETECTION &\nRECOVERY REPORTING",
            "3. VALIDATION, TYPOLOGY\n& UNCERTAINTY",
            "4. COMPREHENSIVE\nSENSITIVITY TESTING",
        ],
        "boxes": {
            "concept": (
                "CONCEPT:\nLAND-WATER-BALANCE",
                "(TWSA ≈ GWSA + SMSA +\nSWEA + CWA + SWA)",
                "#dceaf7",
                "#6a8ba8",
                True,
                False,
            ),
            "input": (
                "INPUT DATA\n(1°×1° Grid, Monthly):",
                "TWSA, SMSA, SWEA, CWA",
                "#dceaf7",
                "#8aa6c9",
                False,
                False,
            ),
            "gwa_def": (
                "GWA DEFINITION (Eq 1):",
                "GWA = TWSA - SMSA -\nSWEA - CWA",
                "#e5d9ee",
                "#b091c2",
                False,
                False,
            ),
            "gwa": (
                "GWA",
                "",
                "#dceaf7",
                "#8aa6c9",
                False,
                False,
            ),
            "gwa_out": (
                "OUTPUT:\nGWA SERIES",
                "(Groundwater-Dominated Proxy,\nMAY contain residual SWA)",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
            "deseason": (
                "DE-SEASONING:",
                "Apply STL(LOESS) to GWA series",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "beast": (
                "BEAST:",
                "Bayesian change-point\n& decomposition\n(cpPr, cpCI)",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "segmented": (
                "SEGMENTED\nREGRESSION:",
                "Piecewise linear model\n(Max 1 breakpoint, BIC rule)",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "rules": (
                "RECOVERY REPORTING RULES",
                "CRITERIA FOR RECOVERY:\n\nBreakpoint > Jan 2015\nPost-break trend positive\nimprovement > 0.5 cm/yr",
                "#fff8c9",
                "#9e8b2f",
                False,
                True,
            ),
            "report_beast": (
                "REPORT BEAST\nRECOVERY",
                "",
                "#fff3b8",
                "#8b7a2d",
                False,
                True,
            ),
            "fallback": (
                "FALLBACK:",
                "Report Segmented\ncandidate (if rules\nmet)",
                "#fff3b8",
                "#8b7a2d",
                False,
                True,
            ),
            "report_out": (
                "OUTPUT:\nReportable Recovery Breakpoints",
                "(late-2010s cluster)",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
            "spatial": (
                "SPATIAL UNCERTAINTY:",
                "2°×2° Spatial block Bootstrap\n(N-S comparisons for\nonset/trend strength)",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "typology": (
                "EXPLORATORY\nTYPOLOGY",
                "ΔW (Eq 2):\nCompare trajectories &\nGWA residualized vs.\nwater-fraction anomalies",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "validation": (
                "PUBLIC WELL\nVALIDATION:",
                "Aggregate wells -> grid -> Region\nPearson/Theil-Sen\n(full & post-2019)\n12-mo Moving-Block Bootstrap",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "sensitivity": (
                "SENSITIVITY\nANALYSES:",
                "Test alternative\nbaselines,\ncoverage filters,\nreporting thresholds,\nspatial block sizes,\nrank-based validation",
                "#e1d4ea",
                "#b091c2",
                False,
                False,
            ),
            "results": (
                "RESULTS:",
                "Assess decision\nstability & robust\nregional conclusions",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
        },
        "priority": "PRIORITY?\nIf BEAST cpPr >=\n0.50",
        "yes": "Yes",
        "no": "No",
    },
    "zh": {
        "column_titles": [
            "1. 地下水代理定义",
            "2. 断点识别与\n恢复报告",
            "3. 验证、分型\n与不确定性",
            "4. 综合\n敏感性检验",
        ],
        "boxes": {
            "concept": (
                "概念：\n水量平衡",
                "(TWSA ≈ GWSA + SMSA +\nSWEA + CWA + SWA)",
                "#dceaf7",
                "#6a8ba8",
                True,
                False,
            ),
            "input": (
                "输入数据\n（1°×1°，月尺度）：",
                "TWSA、SMSA、SWEA、CWA",
                "#dceaf7",
                "#8aa6c9",
                False,
                False,
            ),
            "gwa_def": (
                "GWA 定义（式1）：",
                "GWA = TWSA - SMSA -\nSWEA - CWA",
                "#e5d9ee",
                "#b091c2",
                False,
                False,
            ),
            "gwa": (
                "GWA",
                "",
                "#dceaf7",
                "#8aa6c9",
                False,
                False,
            ),
            "gwa_out": (
                "输出：\nGWA 序列",
                "（地下水主导代理，\n可能含残余地表水）",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
            "deseason": (
                "去季节化：",
                "对 GWA 序列应用 STL(LOESS)",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "beast": (
                "BEAST：",
                "贝叶斯变点\n与分解\n(cpPr, cpCI)",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "segmented": (
                "分段回归：",
                "分段线性模型\n（最多1个断点，BIC 规则）",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "rules": (
                "恢复报告规则",
                "恢复判据：\n\n断点 > 2015-01\n断点后趋势为正\n改善幅度 > 0.5 cm/yr",
                "#fff8c9",
                "#9e8b2f",
                False,
                True,
            ),
            "report_beast": (
                "报告 BEAST\n恢复断点",
                "",
                "#fff3b8",
                "#8b7a2d",
                False,
                True,
            ),
            "fallback": (
                "回退：",
                "若满足规则，\n报告分段回归断点",
                "#fff3b8",
                "#8b7a2d",
                False,
                True,
            ),
            "report_out": (
                "输出：\n可报告恢复断点",
                "（2010年代后期集中期）",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
            "spatial": (
                "空间不确定性：",
                "2°×2° 空间块 Bootstrap\n（南北恢复时间/强度比较）",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "typology": (
                "探索性分型",
                "ΔW（式2）：\n比较轨迹类型与\n残差化后的\n水体异常关系",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "validation": (
                "公开井验证：",
                "井 -> 格网 -> 区域聚合\nPearson/Theil-Sen\n（全期与2019后）\n12月移动块 Bootstrap",
                "#e8deef",
                "#b091c2",
                False,
                False,
            ),
            "sensitivity": (
                "敏感性分析：",
                "测试不同基线、\n覆盖阈值、\n报告阈值、\n空间块大小、\n秩相关验证",
                "#e1d4ea",
                "#b091c2",
                False,
                False,
            ),
            "results": (
                "结果：",
                "评估结论的\n决策稳定性与\n区域稳健性",
                "#dff0da",
                "#8bb083",
                False,
                False,
            ),
        },
        "priority": "优先级？\n若 BEAST cpPr >=\n0.50",
        "yes": "是",
        "no": "否",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the workflow figure for the manuscript.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Output language.")
    return parser.parse_args()


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    rounded: bool = False,
    dashed: bool = False,
    title_size: float = 11.0,
    body_size: float = 8.8,
) -> None:
    patch_cls = FancyBboxPatch if rounded else Rectangle
    if rounded:
        patch = patch_cls(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    else:
        patch = patch_cls(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.74,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#111111",
        zorder=3,
        linespacing=1.1,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#222222",
            zorder=3,
            linespacing=1.2,
        )


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="black",
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
    )


def add_diamond(ax, center_x: float, center_y: float, w: float, h: float, text: str) -> None:
    verts = [
        (center_x, center_y + h / 2),
        (center_x + w / 2, center_y),
        (center_x, center_y - h / 2),
        (center_x - w / 2, center_y),
    ]
    diamond = Polygon(verts, closed=True, facecolor="#fff8c9", edgecolor="#c07a44", linewidth=1.0, zorder=2)
    ax.add_patch(diamond)
    ax.text(center_x, center_y, text, ha="center", va="center", fontsize=8.2, fontweight="bold", linespacing=1.05, zorder=3)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_stack = ["DejaVu Sans", "Arial", "Liberation Sans"] if args.language == "en" else [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "axes.unicode_minus": False,
        }
    )

    text = TEXT[args.language]

    fig, ax = plt.subplots(figsize=(15.8, 6.35))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column backgrounds
    columns = [
        (0.00, 0.18),
        (0.18, 0.52),
        (0.52, 0.77),
        (0.84, 1.00),
    ]
    shades = ["#e6e6e6", "#f0f0f0", "#e6e6e6", "#e6e6e6"]
    for (x0, x1), shade in zip(columns, shades):
        ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.0, facecolor=shade, edgecolor="none", zorder=0))

    # Top titles
    title_x = [0.09, 0.35, 0.645, 0.92]
    for x, title in zip(title_x, text["column_titles"]):
        ax.text(x, 0.965, title, ha="center", va="top", fontsize=13.5, fontweight="bold", linespacing=1.15)

    b = text["boxes"]

    # Column 1
    add_box(ax, 0.005, 0.78, 0.165, 0.11, *b["concept"])
    add_box(ax, 0.005, 0.63, 0.165, 0.10, *b["input"])
    add_box(ax, 0.005, 0.49, 0.165, 0.08, *b["gwa_def"])
    add_box(ax, 0.005, 0.36, 0.165, 0.07, *b["gwa"])
    add_box(ax, 0.005, 0.20, 0.165, 0.11, *b["gwa_out"])

    # Column 2
    add_box(ax, 0.30, 0.79, 0.165, 0.08, *b["deseason"])
    add_box(ax, 0.255, 0.58, 0.12, 0.16, *b["beast"])
    add_box(ax, 0.397, 0.58, 0.12, 0.16, *b["segmented"])
    ax.add_patch(
        Rectangle(
            (0.258, 0.16),
            0.26,
            0.33,
            facecolor="#fff8c9",
            edgecolor="#9e8b2f",
            linewidth=1.1,
            linestyle="--",
            zorder=2,
        )
    )
    ax.text(0.388, 0.43, "RECOVERY REPORTING RULES", ha="center", va="center", fontsize=10.8, fontweight="bold", zorder=3)
    ax.text(0.29, 0.36, "CRITERIA FOR RECOVERY:", ha="left", va="center", fontsize=8.8, fontweight="bold", zorder=3)
    ax.text(
        0.36,
        0.27,
        "Breakpoint > Jan 2015\nPost-break trend positive\nimprovement > 0.5 cm/yr",
        ha="left",
        va="center",
        fontsize=8.5,
        linespacing=1.2,
        zorder=3,
    )
    add_box(ax, 0.435, 0.33, 0.095, 0.08, *b["report_beast"], title_size=9.2, body_size=8.4)
    add_box(ax, 0.435, 0.19, 0.095, 0.11, *b["fallback"], title_size=9.2, body_size=8.2)
    add_diamond(ax, 0.345, 0.24, 0.075, 0.11, text["priority"])
    ax.text(0.385, 0.27, text["yes"], fontsize=8.5, ha="left", va="center")
    ax.text(0.387, 0.215, text["no"], fontsize=8.5, ha="left", va="center")
    add_box(ax, 0.273, 0.01, 0.20, 0.10, *b["report_out"], body_size=8.3)

    # Column 3
    add_box(ax, 0.61, 0.76, 0.16, 0.12, *b["spatial"], title_size=10.6)
    add_box(ax, 0.61, 0.48, 0.16, 0.21, *b["typology"], title_size=10.6)
    add_box(ax, 0.61, 0.16, 0.16, 0.24, *b["validation"], title_size=10.5)

    # Column 4
    add_box(ax, 0.855, 0.48, 0.135, 0.22, *b["sensitivity"], title_size=10.6)
    add_box(ax, 0.865, 0.21, 0.125, 0.10, *b["results"], title_size=10.6)

    # Arrows
    add_arrow(ax, 0.170, 0.83, 0.255, 0.83)   # col1 -> col2 top
    add_arrow(ax, 0.170, 0.685, 0.30, 0.83)   # input -> deseasoning
    add_arrow(ax, 0.465, 0.83, 0.61, 0.83)    # col2 -> col3 top
    add_arrow(ax, 0.77, 0.83, 0.855, 0.83)    # col3 -> col4 top

    add_arrow(ax, 0.088, 0.78, 0.088, 0.73)
    add_arrow(ax, 0.088, 0.63, 0.088, 0.57)
    add_arrow(ax, 0.088, 0.49, 0.088, 0.43)
    add_arrow(ax, 0.088, 0.36, 0.088, 0.31)

    add_arrow(ax, 0.382, 0.79, 0.315, 0.74)
    add_arrow(ax, 0.382, 0.79, 0.457, 0.74)
    add_arrow(ax, 0.315, 0.58, 0.388, 0.49)
    add_arrow(ax, 0.457, 0.58, 0.388, 0.49)
    add_arrow(ax, 0.385, 0.41, 0.385, 0.33)
    add_arrow(ax, 0.385, 0.24, 0.41, 0.37)
    add_arrow(ax, 0.385, 0.24, 0.41, 0.245)
    add_arrow(ax, 0.388, 0.16, 0.388, 0.11)

    add_arrow(ax, 0.518, 0.79, 0.61, 0.82)
    add_arrow(ax, 0.518, 0.58, 0.61, 0.585)
    add_arrow(ax, 0.473, 0.11, 0.61, 0.27)

    add_arrow(ax, 0.77, 0.585, 0.855, 0.585)
    add_arrow(ax, 0.77, 0.28, 0.855, 0.56)
    add_arrow(ax, 0.922, 0.83, 0.922, 0.70)
    add_arrow(ax, 0.922, 0.48, 0.922, 0.31)

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
