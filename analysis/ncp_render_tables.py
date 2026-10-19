from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TABLE_TITLES = {
    "zh": {
        "tab01_data_sources.csv": ("表1 研究所用主要数据源、变量及其用途", 1.15),
        "tab02_trajectory_rules_counts.csv": ("表2 地下水恢复轨迹分型规则及轨迹数量", 1.2),
        "tab03_trajectory_stats.csv": ("表3 不同恢复轨迹类型的趋势与水体变化统计", 1.1),
        "tab04_validation_stats.csv": ("表4 公开观测增强验证结果统计", 1.1),
    },
    "en": {
        "tab01_data_sources.csv": ("Table 1. Major data sources, variables, and their roles in this study", 1.15),
        "tab02_trajectory_rules_counts.csv": ("Table 2. Rules and counts for the exploratory trajectory types", 1.2),
        "tab03_trajectory_stats.csv": ("Table 3. Trend and surface-water statistics for the exploratory trajectory types", 1.1),
        "tab04_validation_stats.csv": ("Table 4. Key statistics of the public observation-based validation", 1.1),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render manuscript tables into PNG and Excel outputs.")
    parser.add_argument("--figure-table-dir", required=True, help="Directory containing table-ready CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for rendered tables.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Output language.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_style(language: str) -> None:
    font_stack = ["DejaVu Sans", "Arial", "Liberation Sans"] if language == "en" else [
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


def format_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def render_table(df: pd.DataFrame, title: str, output_path: Path, height_scale: float) -> None:
    formatted = format_frame(df)
    rows, cols = formatted.shape
    fig_height = max(2.5, 0.40 * rows + 0.7) * height_scale
    fig_width = max(8.3, 1.55 * cols)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=formatted.values,
        colLabels=list(formatted.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.18)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#4d4d4d")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#dceaf7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f7f7")

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.figure_table_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    set_style(args.language)

    table_titles = TABLE_TITLES[args.language]
    workbook_path = out_dir / "manuscript_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for file_name, (title, height_scale) in table_titles.items():
            df = pd.read_csv(in_dir / file_name)
            sheet_name = file_name.replace(".csv", "")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            render_table(df, title, out_dir / file_name.replace(".csv", ".png"), height_scale)


if __name__ == "__main__":
    main()
