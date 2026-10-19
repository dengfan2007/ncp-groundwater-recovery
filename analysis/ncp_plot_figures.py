from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, Point, Polygon, box

try:
    import contextily as cx
except Exception:  # pragma: no cover
    cx = None
else:
    from xyzservices import TileProvider

    NASA_GIBS_BLUEMARBLE = TileProvider(
        name="NASA.GIBS.BlueMarble",
        url="https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg",
        attribution="NASA GIBS",
        max_zoom=8,
    )


PALETTE = {
    "early_rebound": "#1f78b4",
    "main_rebound": "#33a02c",
    "legacy_slowdown": "#ff7f00",
    "stalled_or_declining": "#d73027",
    "north": "#4c78a8",
    "south": "#f58518",
}

BOHAI_POLYGON = Polygon(
    [
        (117.15, 40.60),
        (120.10, 40.60),
        (120.10, 36.45),
        (119.15, 36.55),
        (118.75, 36.95),
        (118.40, 37.45),
        (117.95, 37.95),
        (117.50, 38.45),
        (117.25, 39.10),
        (117.15, 39.80),
        (117.15, 40.60),
    ]
)

COASTLINE = [
    (117.15, 39.80),
    (117.25, 39.10),
    (117.50, 38.45),
    (117.95, 37.95),
    (118.40, 37.45),
    (118.75, 36.95),
    (119.15, 36.55),
]

LANG = {
    "zh": {
        "policy_lines": [
            ("2014-12-12", "SNWD"),
            ("2019-02-22", "治理"),
            ("2022-05-27", "复苏"),
            ("2023-04-03", "复苏实施"),
        ],
        "trajectory_labels": {
            "early_rebound": "提前恢复型",
            "main_rebound": "主恢复波型",
            "legacy_slowdown": "早期减缓型",
            "stalled_or_declining": "停滞下降型",
        },
        "regional_labels": [
            ("河北", 115.00, 38.85),
            ("河南", 114.15, 35.45),
            ("山东", 118.15, 36.95),
            ("北京", 116.45, 39.95),
            ("天津", 117.35, 39.25),
            ("渤海", 118.80, 38.95),
        ],
        "case_labels": [
            ("石家庄", 114.52, 38.05),
            ("衡水", 115.67, 37.74),
        ],
        "hutuo_label": "滹沱河主走廊",
        "north_label": "北部",
        "south_label": "南部",
        "all_label": "全区",
        "longitude_label": "经度（°E）",
        "latitude_label": "纬度（°N）",
        "date_label": "日期",
        "gwa_label": "GWA（cm）",
        "mean_gwa_label": "平均 GWA（cm）",
        "water_change_label": "水体变化 ΔW",
        "water_fraction_label": "水体比例",
        "residual_gwa_label": "代表网格残差地下水异常（cm）",
        "region_legend_title": "区域",
        "buffer_legend_title": "缓冲带",
        "fig05_onset_title": "Recovery onset window",
        "fig05_post_title": "Post-break recovery intensity",
        "fig05_onset_effect": "South - North mean diff = {estimate:.1f} months [{low:.1f}, {high:.1f}]",
        "fig05_post_effect": "South - North mean diff = {estimate:.2f} cm/yr [{low:.2f}, {high:.2f}]",
        "remote_validation_label": "遥感 GWA (z)",
        "well_validation_label": "井序列 (z)",
        "standardized_label": "标准化值",
        "well_coverage_title": "井数",
        "well_coverage_label": "n",
        "pearson_template": "{region}（Pearson r = {corr}）",
    },
    "en": {
        "policy_lines": [
            ("2014-12-12", "SNWD"),
            ("2019-02-22", "Action plan"),
            ("2022-05-27", "Restoration"),
            ("2023-04-03", "Implementation"),
        ],
        "trajectory_labels": {
            "early_rebound": "Early rebound",
            "main_rebound": "Main rebound",
            "legacy_slowdown": "Legacy slowdown",
            "stalled_or_declining": "Stalled or declining",
        },
        "regional_labels": [
            ("Hebei", 115.00, 38.85),
            ("Henan", 114.15, 35.45),
            ("Shandong", 118.15, 36.95),
            ("Beijing", 116.45, 39.95),
            ("Tianjin", 117.35, 39.25),
            ("Bohai Sea", 118.80, 38.95),
        ],
        "case_labels": [
            ("Shijiazhuang", 114.55, 38.05),
            ("Hengshui", 115.67, 37.74),
        ],
        "hutuo_label": "Main Hutuo River corridor",
        "north_label": "North",
        "south_label": "South",
        "all_label": "All NCP",
        "longitude_label": "Longitude (°E)",
        "latitude_label": "Latitude (°N)",
        "date_label": "Date",
        "gwa_label": "GWA (cm)",
        "mean_gwa_label": "Mean GWA (cm)",
        "water_change_label": "Surface-water change ΔW",
        "water_fraction_label": "Surface-water fraction",
        "residual_gwa_label": "Residual groundwater anomaly in representative grid (cm)",
        "region_legend_title": "Region",
        "buffer_legend_title": "Buffer",
        "fig05_onset_title": "Recovery onset window",
        "fig05_post_title": "Post-break recovery intensity",
        "fig05_onset_effect": "South - North mean diff = {estimate:.1f} months [{low:.1f}, {high:.1f}]",
        "fig05_post_effect": "South - North mean diff = {estimate:.2f} cm/yr [{low:.2f}, {high:.2f}]",
        "remote_validation_label": "Remote GWA (z)",
        "well_validation_label": "Well series (z)",
        "standardized_label": "Standardized value",
        "well_coverage_title": "Well count",
        "well_coverage_label": "n",
        "pearson_template": "{region} (r = {corr}; 2019+ r = {corr_post})",
        "post2019_window_label": "Shaded: post-2019 window",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render manuscript figures for the NCP groundwater paper.")
    parser.add_argument("--figure-table-dir", required=True, help="Directory containing figure-ready CSV tables.")
    parser.add_argument("--output-dir", required=True, help="Output directory for rendered figure PNG files.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Output language.")
    return parser.parse_args()


def set_plot_style(language: str) -> None:
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
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
    )


def draw_regional_basemap(ax, regional_labels: list[tuple[str, float, float]], *, show_labels: bool = True) -> None:
    ax.set_facecolor("#f6f2ea")
    gpd.GeoSeries([BOHAI_POLYGON], crs="EPSG:4326").plot(
        ax=ax,
        facecolor="#dbeaf5",
        edgecolor="none",
        alpha=0.9,
        zorder=0,
    )
    coast_x, coast_y = zip(*COASTLINE)
    ax.plot(coast_x, coast_y, color="#8aa6b8", linewidth=1.0, zorder=1)
    if show_labels:
        for label, x, y in regional_labels:
            ax.text(
                x,
                y,
                label,
                fontsize=8.5,
                color="#51606c",
                ha="center",
                va="center",
                zorder=1,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.8},
            )


def draw_case_basemap(ax, all_grid: gpd.GeoDataFrame, focus_grid: gpd.GeoDataFrame, case_labels: list[tuple[str, float, float]]) -> None:
    bounds = focus_grid.total_bounds
    local = all_grid.cx[bounds[0] - 0.55 : bounds[2] + 0.55, bounds[1] - 0.45 : bounds[3] + 0.45]
    if not local.empty:
        local.boundary.plot(ax=ax, color="#b9c1c9", linewidth=0.7, alpha=0.8, zorder=0)
    for label, x, y in case_labels:
        ax.text(
            x,
            y,
            label,
            fontsize=8.5,
            color="#5c636a",
            ha="center",
            va="center",
            zorder=1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.8},
        )


def load_csvs(base: Path) -> dict[str, pd.DataFrame]:
    names = [
        "fig01_grid_map",
        "fig01_hutuo_line_vertices",
        "fig01_case_buffers",
        "fig02_region_gwa_timeseries",
        "fig03_trajectory_map",
        "fig04_trajectory_mean_curves",
        "fig05_recovery_metrics",
        "fig06_surface_water_by_trajectory",
        "fig06_case_grid_map",
        "fig07_case_buffer_timeseries",
        "fig08_validation_timeseries",
        "fig08_validation_coverage",
    ]
    return {name: pd.read_csv(base / f"{name}.csv") for name in names}


def build_grid_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometries = [box(row.west, row.south, row.east, row.north) for row in df.itertuples()]
    return gpd.GeoDataFrame(df.copy(), geometry=geometries, crs="EPSG:4326")


def build_case_buffers(line_df: pd.DataFrame, buffer_df: pd.DataFrame) -> gpd.GeoDataFrame:
    line = LineString(list(zip(line_df["lon"], line_df["lat"])))
    line_gdf = gpd.GeoDataFrame({"name": ["hutuo"]}, geometry=[line], crs="EPSG:4326").to_crs(3857)
    line_m = line_gdf.geometry.iloc[0]
    rows = []
    for row in buffer_df.itertuples():
        outer = line_m.buffer(row.outer_m, cap_style=2, join_style=2)
        inner = line_m.buffer(row.inner_m, cap_style=2, join_style=2) if row.inner_m > 0 else None
        ring = outer if inner is None else outer.difference(inner)
        rows.append({"buffer_km": row.buffer_km, "inner_m": row.inner_m, "outer_m": row.outer_m, "geometry": ring})
    return gpd.GeoDataFrame(rows, crs=3857).to_crs(4326)


def draw_fig01(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    grid = build_grid_geodf(data["fig01_grid_map"])
    line_df = data["fig01_hutuo_line_vertices"]
    buffers = build_case_buffers(line_df, data["fig01_case_buffers"])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.grid(False)
    draw_regional_basemap(ax, labels["regional_labels"])
    grid.boundary.plot(ax=ax, color="#93a0ab", linewidth=0.8, zorder=2)
    grid.loc[grid["region_tag"] == "north"].plot(ax=ax, facecolor="#dceaf7", edgecolor="none", alpha=0.6)
    grid.loc[grid["region_tag"] == "south"].plot(ax=ax, facecolor="#fde8d0", edgecolor="none", alpha=0.6)
    grid.loc[grid["is_case_grid"] == 1].boundary.plot(ax=ax, color="#cb181d", linewidth=2.0, zorder=4)
    for idx, row in buffers.iterrows():
        gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(
            ax=ax, facecolor="none", edgecolor="#cb181d", linewidth=0.8 + idx * 0.5, alpha=0.7, zorder=4
        )

    ax.plot(line_df["lon"], line_df["lat"], color="#08519c", linewidth=2.5, label=labels["hutuo_label"], zorder=5)
    ax.set_xlabel(labels["longitude_label"])
    ax.set_ylabel(labels["latitude_label"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(111.9, 120.1)
    ax.set_ylim(34.3, 40.8)
    ax.legend(
        handles=[
            Patch(facecolor="#dceaf7", edgecolor="none", label=labels["north_label"] + (" grids" if labels["north_label"] == "North" else "网格")),
            Patch(facecolor="#fde8d0", edgecolor="none", label=labels["south_label"] + (" grids" if labels["south_label"] == "South" else "网格")),
            Line2D([0], [0], color="#cb181d", linewidth=2, label="Case grids / buffers" if labels["north_label"] == "North" else "案例网格/缓冲带"),
            Line2D([0], [0], color="#08519c", linewidth=2.5, label=labels["hutuo_label"]),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_study_area.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig01_v2(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    grid = build_grid_geodf(data["fig01_grid_map"])
    line_df = data["fig01_hutuo_line_vertices"]
    buffers = build_case_buffers(line_df, data["fig01_case_buffers"])
    case_grid = grid.loc[grid["is_case_grid"] == 1].copy()

    north_fill = "#dbeaf5"
    south_fill = "#f4e2cb"
    case_edge = "#cb181d"
    river_color = "#08519c"
    buffer_colors = ["#fde0dd", "#fcbba1", "#fc9272", "#de2d26"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), gridspec_kw={"width_ratios": [1.28, 1.0]})
    ax_overview, ax_zoom = axes

    ax_overview.grid(False)
    draw_regional_basemap(ax_overview, labels["regional_labels"])
    grid.loc[grid["region_tag"] == "north"].plot(ax=ax_overview, facecolor=north_fill, edgecolor="none", alpha=0.72, zorder=1)
    grid.loc[grid["region_tag"] == "south"].plot(ax=ax_overview, facecolor=south_fill, edgecolor="none", alpha=0.72, zorder=1)
    grid.boundary.plot(ax=ax_overview, color="#94a3b3", linewidth=0.95, zorder=2)
    case_grid.boundary.plot(ax=ax_overview, color=case_edge, linewidth=2.35, zorder=4)
    for idx, row in buffers.iterrows():
        gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(
            ax=ax_overview,
            facecolor="none",
            edgecolor=case_edge,
            linewidth=0.8 + idx * 0.35,
            alpha=0.74,
            zorder=4,
        )
    ax_overview.plot(line_df["lon"], line_df["lat"], color=river_color, linewidth=2.7, zorder=5)
    add_panel_label(ax_overview, "(a)")
    ax_overview.set_xlabel(labels["longitude_label"])
    ax_overview.set_ylabel(labels["latitude_label"])
    ax_overview.set_aspect("equal", adjustable="box")
    ax_overview.set_xlim(111.9, 120.1)
    ax_overview.set_ylim(34.3, 40.8)
    overview_handles = [
        Patch(facecolor=north_fill, edgecolor="none", label=labels["north_label"] + (" grids" if labels["north_label"] == "North" else "网格")),
        Patch(facecolor=south_fill, edgecolor="none", label=labels["south_label"] + (" grids" if labels["south_label"] == "South" else "网格")),
        Line2D([0], [0], color=case_edge, linewidth=2.35, label="Case grids / buffers" if labels["north_label"] == "North" else "案例网格/缓冲带"),
        Line2D([0], [0], color=river_color, linewidth=2.7, label=labels["hutuo_label"]),
    ]
    ax_overview.legend(handles=overview_handles, loc="lower left", frameon=True, facecolor="white")
    if labels["north_label"] == "North":
        ax_overview.legend(
            handles=[
                Patch(facecolor=north_fill, edgecolor="none", label="North grids"),
                Patch(facecolor=south_fill, edgecolor="none", label="South grids"),
                Line2D([0], [0], color=case_edge, linewidth=2.35, label="Case grids / buffers"),
                Line2D([0], [0], color=river_color, linewidth=2.7, label=labels["hutuo_label"]),
            ],
            loc="lower left",
            frameon=True,
            facecolor="white",
        )

    ax_zoom.grid(False)
    draw_case_basemap(ax_zoom, grid, case_grid, labels["case_labels"])
    grid.boundary.plot(ax=ax_zoom, color="#c7ccd3", linewidth=0.7, alpha=0.8, zorder=0)
    for color, row in zip(buffer_colors, buffers.itertuples()):
        gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(
            ax=ax_zoom,
            facecolor=color,
            edgecolor=case_edge,
            linewidth=0.9,
            alpha=0.40,
            zorder=2,
        )
    case_grid.boundary.plot(ax=ax_zoom, color=river_color, linewidth=2.15, zorder=3)
    ax_zoom.plot(line_df["lon"], line_df["lat"], color=river_color, linewidth=2.8, zorder=4)
    add_panel_label(ax_zoom, "(b)")
    ax_zoom.set_xlabel(labels["longitude_label"])
    ax_zoom.set_ylabel(labels["latitude_label"])
    ax_zoom.set_aspect("equal", adjustable="box")
    xmin, ymin, xmax, ymax = case_grid.total_bounds
    ax_zoom.set_xlim(xmin - 0.55, xmax + 0.55)
    ax_zoom.set_ylim(ymin - 0.45, ymax + 0.45)
    zoom_handles = [
        Patch(facecolor=buffer_colors[0], edgecolor=case_edge, label="0-5 km"),
        Patch(facecolor=buffer_colors[1], edgecolor=case_edge, label="5-10 km"),
        Patch(facecolor=buffer_colors[2], edgecolor=case_edge, label="10-20 km"),
        Patch(facecolor=buffer_colors[3], edgecolor=case_edge, label="20-30 km"),
    ]
    ax_zoom.legend(handles=zoom_handles, title=labels["buffer_legend_title"], loc="lower right", frameon=True, facecolor="white")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(output_dir / "fig01_study_area.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig01_v3(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    grid = build_grid_geodf(data["fig01_grid_map"])
    line_df = data["fig01_hutuo_line_vertices"]
    buffers = build_case_buffers(line_df, data["fig01_case_buffers"])

    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    ax.grid(False)

    grid_3857 = grid.to_crs(3857)
    buffers_3857 = buffers.to_crs(3857)
    line_3857 = gpd.GeoDataFrame({"name": ["hutuo"]}, geometry=[LineString(list(zip(line_df["lon"], line_df["lat"])))], crs="EPSG:4326").to_crs(3857)
    label_points = gpd.GeoDataFrame(
        {"label": [name for name, _, _ in labels["regional_labels"]], "geometry": [Point(x, y) for name, x, y in labels["regional_labels"]]},
        crs="EPSG:4326",
    ).to_crs(3857)
    hutuo_label = gpd.GeoDataFrame({"label": [labels["hutuo_label"]], "geometry": [Point(114.28, 38.47)]}, crs="EPSG:4326").to_crs(3857)

    xmin, ymin, xmax, ymax = grid_3857.total_bounds
    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    fallback_paths = [
        output_dir.parent / "__fig1_sat_test3" / "fig01_study_area.png",
        output_dir.parent / "__fig1_sat_test2" / "fig01_study_area.png",
        output_dir.parent / "__fig1_sat_test" / "fig01_study_area.png",
    ]
    fallback_path = next((path for path in fallback_paths if path.exists()), None)
    basemap_loaded = False
    if cx is not None:
        try:
            cx.add_basemap(ax, source=NASA_GIBS_BLUEMARBLE, attribution=False, zoom=7)
            basemap_loaded = True
        except Exception:
            basemap_loaded = False
    if not basemap_loaded:
        # Preserve the satellite-style version offline when tile requests are unavailable.
        if labels["north_label"] == "North" and fallback_path is not None:
            plt.close(fig)
            shutil.copyfile(fallback_path, output_dir / "fig01_study_area.png")
            return
        draw_regional_basemap(ax, labels["regional_labels"], show_labels=False)

    north_grids = grid_3857.loc[grid_3857["region_tag"] == "north"].copy()
    south_grids = grid_3857.loc[grid_3857["region_tag"] == "south"].copy()
    north_grids.plot(ax=ax, facecolor="#cfe2f5", edgecolor="none", alpha=0.34, zorder=2)
    south_grids.plot(ax=ax, facecolor="#f7d8b3", edgecolor="none", alpha=0.28, zorder=2)
    north_grids.boundary.plot(ax=ax, color="#8fb1d0", linewidth=1.05, zorder=3)
    south_grids.boundary.plot(ax=ax, color="#caa06f", linewidth=1.05, zorder=3)
    grid_3857.loc[grid_3857["is_case_grid"] == 1].boundary.plot(ax=ax, color="#d62828", linewidth=2.25, zorder=5)
    for idx, row in buffers_3857.iterrows():
        gpd.GeoSeries([row.geometry], crs="EPSG:3857").plot(
            ax=ax,
            facecolor="none",
            edgecolor="#ef4444",
            linewidth=0.8 + idx * 0.38,
            alpha=0.80,
            zorder=5,
        )
    line_3857.plot(ax=ax, color="#0b5cad", linewidth=2.6, zorder=6)

    for row in label_points.itertuples():
        ax.text(
            row.geometry.x,
            row.geometry.y,
            row.label,
            fontsize=8.8,
            color="white",
            ha="center",
            va="center",
            zorder=7,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.28, "pad": 1.0},
        )
    hutuo_row = hutuo_label.iloc[0]
    ax.text(
        hutuo_row.geometry.x,
        hutuo_row.geometry.y,
        hutuo_row.label,
        fontsize=8.6,
        color="white",
        ha="left",
        va="bottom",
        zorder=7,
        bbox={"facecolor": "#0b5cad", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
    )
    region_tag_points = gpd.GeoDataFrame(
        {
            "label": [labels["north_label"], labels["south_label"]],
            "geometry": [Point(117.7, 39.9), Point(116.0, 35.3)],
        },
        crs="EPSG:4326",
    ).to_crs(3857)
    for row in region_tag_points.itertuples():
        fill = "#6f97bf" if row.label == labels["north_label"] else "#c18b4d"
        ax.text(
            row.geometry.x,
            row.geometry.y,
            row.label,
            fontsize=11.5,
            color="white",
            fontweight="semibold",
            ha="center",
            va="center",
            zorder=7,
            bbox={"facecolor": fill, "edgecolor": "none", "alpha": 0.30, "pad": 1.4},
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if labels["north_label"] == "North":
        legend_handles = [
            Patch(facecolor="#cfe2f5", edgecolor="#8fb1d0", alpha=0.55, label="North grids"),
            Patch(facecolor="#f7d8b3", edgecolor="#caa06f", alpha=0.55, label="South grids"),
            Line2D([0], [0], color="#d62828", linewidth=2.25, label="Case grids / buffers"),
            Line2D([0], [0], color="#0b5cad", linewidth=2.6, label=labels["hutuo_label"]),
        ]
    else:
        legend_handles = [
            Patch(facecolor="#cfe2f5", edgecolor="#8fb1d0", alpha=0.55, label=f"{labels['north_label']}网格"),
            Patch(facecolor="#f7d8b3", edgecolor="#caa06f", alpha=0.55, label=f"{labels['south_label']}网格"),
            Line2D([0], [0], color="#d62828", linewidth=2.25, label="案例网格/缓冲带"),
            Line2D([0], [0], color="#0b5cad", linewidth=2.6, label=labels["hutuo_label"]),
        ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white")
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_study_area.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig02(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig02_region_gwa_timeseries"].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    region_labels = {"all": labels["all_label"], "north": labels["north_label"], "south": labels["south_label"]}
    for region in ["all", "north", "south"]:
        sub = df[df["region_tag"] == region]
        color = "#222222" if region == "all" else PALETTE[region]
        lw = 2.4 if region == "all" else 1.8
        ax.plot(sub["date"], sub["gwa_cm"], label=region_labels[region], color=color, linewidth=lw)

    for date_text, line_label in labels["policy_lines"]:
        date = pd.Timestamp(date_text)
        ax.axvline(date, color="#999999", linestyle="--", linewidth=1)
        ax.text(date, ax.get_ylim()[1], line_label, rotation=90, va="top", ha="right", fontsize=9.2, color="#666666")

    ax.set_ylabel(labels["gwa_label"])
    ax.set_xlabel(labels["date_label"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(title=labels["region_legend_title"], frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig02_region_gwa_timeseries.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig03(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    traj = build_grid_geodf(data["fig03_trajectory_map"])
    traj_3857 = traj.to_crs(3857)
    label_points = gpd.GeoDataFrame(
        {"label": [name for name, _, _ in labels["regional_labels"]], "geometry": [Point(x, y) for name, x, y in labels["regional_labels"]]},
        crs="EPSG:4326",
    ).to_crs(3857)

    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    ax.grid(False)

    xmin, ymin, xmax, ymax = traj_3857.total_bounds
    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    basemap_loaded = False
    if cx is not None:
        try:
            cx.add_basemap(ax, source=NASA_GIBS_BLUEMARBLE, attribution=False, zoom=7)
            basemap_loaded = True
        except Exception:
            basemap_loaded = False
    if not basemap_loaded:
        ax.set_facecolor("#f6f2ea")
        gpd.GeoSeries([BOHAI_POLYGON], crs="EPSG:4326").to_crs(3857).plot(
            ax=ax,
            facecolor="#dbeaf5",
            edgecolor="none",
            alpha=0.9,
            zorder=0,
        )
        gpd.GeoSeries([LineString(COASTLINE)], crs="EPSG:4326").to_crs(3857).plot(
            ax=ax,
            color="#8aa6b8",
            linewidth=1.0,
            zorder=1,
        )
    for trajectory in ["early_rebound", "main_rebound", "legacy_slowdown", "stalled_or_declining"]:
        sub = traj_3857[traj_3857["trajectory_type"] == trajectory]
        if not sub.empty:
            sub.plot(
                ax=ax,
                facecolor=PALETTE[trajectory],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.58 if basemap_loaded else 0.88,
                zorder=3,
            )
    traj_3857.boundary.plot(ax=ax, color="#f7f7f7" if basemap_loaded else "#5a5a5a", linewidth=0.7, zorder=4)
    for row in label_points.itertuples():
        if basemap_loaded:
            text_color = "white"
            text_box = {"facecolor": "black", "edgecolor": "none", "alpha": 0.28, "pad": 1.0}
        else:
            text_color = "#51606c"
            text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.8}
        ax.text(
            row.geometry.x,
            row.geometry.y,
            row.label,
            fontsize=8.8 if basemap_loaded else 8.5,
            color=text_color,
            ha="center",
            va="center",
            zorder=5,
            bbox=text_box,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(
        handles=[
            Patch(facecolor=PALETTE[key], edgecolor="none", label=labels["trajectory_labels"][key])
            for key in ["early_rebound", "main_rebound", "legacy_slowdown", "stalled_or_declining"]
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig03_trajectory_map.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig04(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig04_trajectory_mean_curves"].copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for trajectory in ["early_rebound", "main_rebound", "legacy_slowdown", "stalled_or_declining"]:
        sub = df[df["trajectory_type"] == trajectory]
        smoothed = sub["mean_gwa_cm"].rolling(6, center=True, min_periods=1).mean()
        ax.plot(sub["date"], smoothed, label=labels["trajectory_labels"][trajectory], color=PALETTE[trajectory], linewidth=2.3)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_ylabel(labels["mean_gwa_label"])
    ax.set_xlabel(labels["date_label"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig04_trajectory_mean_curves.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig05(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig05_recovery_metrics"].copy()
    grid_df = df[df["row_type"] == "grid"].copy()
    effect_df = df[df["row_type"] == "effect"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    onset = grid_df.loc[grid_df["metric"] == "onset_month_index"].copy()
    onset["onset_date"] = onset["value"].round().astype(int).apply(
        lambda idx: pd.Timestamp(
            year=(idx // 12) - (1 if idx % 12 == 0 else 0),
            month=(12 if idx % 12 == 0 else idx % 12),
            day=1,
        )
    )
    onset_values = [
        mdates.date2num(onset.loc[onset["region_tag"] == "north", "onset_date"].to_list()),
        mdates.date2num(onset.loc[onset["region_tag"] == "south", "onset_date"].to_list()),
    ]
    box = axes[0].boxplot(
        onset_values,
        vert=False,
        patch_artist=True,
        widths=0.45,
        labels=[labels["north_label"], labels["south_label"]],
    )
    for patch, color in zip(box["boxes"], [PALETTE["north"], PALETTE["south"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for pos, region in enumerate(["north", "south"], start=1):
        vals = mdates.date2num(onset.loc[onset["region_tag"] == region, "onset_date"].to_list())
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        axes[0].scatter(vals, np.full(len(vals), pos) + jitter, s=18, color="#222222", alpha=0.75, zorder=3)
    onset_metric = effect_df.loc[effect_df["metric"] == "south_minus_north_mean_onset_month_diff"].iloc[0]
    policy_dates = [pd.Timestamp(date_text) for date_text, _ in labels["policy_lines"][:2]]
    for date, line_label in zip(policy_dates, [item[1] for item in labels["policy_lines"][:2]]):
        axes[0].axvline(mdates.date2num(date), color="#999999", linestyle="--", linewidth=0.95, zorder=1)
        axes[0].text(
            mdates.date2num(date),
            2.48,
            line_label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8.3,
            color="#666666",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
        )
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    onset_start = min(min(policy_dates), onset["onset_date"].min()) - pd.Timedelta(days=120)
    onset_end = max(max(policy_dates), onset["onset_date"].max()) + pd.Timedelta(days=240)
    axes[0].set_xlim(mdates.date2num(onset_start), mdates.date2num(onset_end))
    axes[0].set_ylim(0.45, 2.55)
    axes[0].set_title(labels["fig05_onset_title"])
    axes[0].set_xlabel(labels["date_label"])
    axes[0].text(
        0.03,
        0.03,
        labels["fig05_onset_effect"].format(
            estimate=onset_metric["value"],
            low=onset_metric["ci_lower"],
            high=onset_metric["ci_upper"],
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.92, "pad": 2.2},
    )
    axes[0].tick_params(axis="x", rotation=30)
    add_panel_label(axes[0], "(a)")

    post = grid_df.loc[grid_df["metric"] == "post_trend_cm_per_year"].copy()
    post_values = [
        post.loc[post["region_tag"] == "north", "value"].to_numpy(dtype=float),
        post.loc[post["region_tag"] == "south", "value"].to_numpy(dtype=float),
    ]
    box = axes[1].boxplot(post_values, patch_artist=True, widths=0.45, labels=[labels["north_label"], labels["south_label"]])
    for patch, color in zip(box["boxes"], [PALETTE["north"], PALETTE["south"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for pos, region in enumerate(["north", "south"], start=1):
        vals = post.loc[post["region_tag"] == region, "value"].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        axes[1].scatter(np.full(len(vals), pos) + jitter, vals, s=18, color="#222222", alpha=0.75, zorder=3)
    post_metric = effect_df.loc[effect_df["metric"] == "south_minus_north_mean_posttrend_cm_per_year"].iloc[0]
    axes[1].axhline(0, color="#999999", linestyle="--", linewidth=0.9)
    axes[1].set_title(labels["fig05_post_title"])
    axes[1].set_ylabel("Post-break trend (cm/yr)")
    axes[1].text(
        0.03,
        0.03,
        labels["fig05_post_effect"].format(
            estimate=post_metric["value"],
            low=post_metric["ci_lower"],
            high=post_metric["ci_upper"],
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.92, "pad": 2.2},
    )
    add_panel_label(axes[1], "(b)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig05_recovery_metrics.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig06(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig06_surface_water_by_trajectory"].copy()
    trajectories = ["early_rebound", "main_rebound", "legacy_slowdown", "stalled_or_declining"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharey=True)
    trajectory_ticks = [labels["trajectory_labels"][t] for t in trajectories]

    box_data = [df.loc[df["trajectory_type"] == traj, "water_change_2019_2024_minus_2015_2018"].dropna() for traj in trajectories]
    box_result = axes[0].boxplot(box_data, patch_artist=True, labels=trajectory_ticks)
    for patch, traj in zip(box_result["boxes"], trajectories):
        patch.set_facecolor(PALETTE[traj])
        patch.set_alpha(0.75)
    add_panel_label(axes[0], "(a)")
    axes[0].set_title("By exploratory trajectory type")
    axes[0].set_ylabel(labels["water_change_label"])
    axes[0].tick_params(axis="x", rotation=18)

    region_summary = df.groupby(["trajectory_type", "region_tag"], as_index=False)["water_change_2019_2024_minus_2015_2018"].mean()
    x = np.arange(len(trajectories))
    width = 0.36
    north_vals = [region_summary.loc[(region_summary["trajectory_type"] == t) & (region_summary["region_tag"] == "north"), "water_change_2019_2024_minus_2015_2018"].mean() for t in trajectories]
    south_vals = [region_summary.loc[(region_summary["trajectory_type"] == t) & (region_summary["region_tag"] == "south"), "water_change_2019_2024_minus_2015_2018"].mean() for t in trajectories]
    add_panel_label(axes[1], "(b)")
    axes[1].bar(x - width / 2, north_vals, width, label=labels["north_label"], color=PALETTE["north"])
    axes[1].bar(x + width / 2, south_vals, width, label=labels["south_label"], color=PALETTE["south"])
    axes[1].set_xticks(x, trajectory_ticks, rotation=18)
    axes[1].set_title("By region and exploratory type")
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "fig06_surface_water.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig07(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    line_df = data["fig01_hutuo_line_vertices"]
    buffers = build_case_buffers(line_df, data["fig01_case_buffers"])
    all_grid = build_grid_geodf(data["fig01_grid_map"])
    case_grid = build_grid_geodf(data["fig06_case_grid_map"])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.grid(False)
    draw_case_basemap(ax, all_grid, case_grid, labels["case_labels"])
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"]
    for color, row in zip(colors, buffers.itertuples()):
        gpd.GeoSeries([row.geometry], crs="EPSG:4326").plot(ax=ax, facecolor=color, edgecolor="#7f0000", alpha=0.45, linewidth=0.9, zorder=2)
    all_grid.boundary.plot(ax=ax, color="#c7ccd1", linewidth=0.65, alpha=0.75, zorder=0)
    case_grid.plot(ax=ax, facecolor="none", edgecolor="#08519c", linewidth=2.2, zorder=3)
    ax.plot(line_df["lon"], line_df["lat"], color="#4a1486", linewidth=2.4, zorder=4)
    ax.set_xlabel(labels["longitude_label"])
    ax.set_ylabel(labels["latitude_label"])
    ax.set_aspect("equal", adjustable="box")
    xmin, ymin, xmax, ymax = case_grid.total_bounds
    ax.set_xlim(xmin - 0.45, xmax + 0.45)
    ax.set_ylim(ymin - 0.35, ymax + 0.35)
    ax.legend(
        handles=[
            Patch(facecolor=colors[0], edgecolor="#7f0000", label="0-5 km"),
            Patch(facecolor=colors[1], edgecolor="#7f0000", label="5-10 km"),
            Patch(facecolor=colors[2], edgecolor="#7f0000", label="10-20 km"),
            Patch(facecolor=colors[3], edgecolor="#7f0000", label="20-30 km"),
            Line2D([0], [0], color="#4a1486", linewidth=2.4, label=labels["hutuo_label"]),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig06_case_map.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig08(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig07_case_buffer_timeseries"].copy()
    df["date"] = pd.to_datetime(df["date"])
    buffers = ["0_5", "5_10", "10_20", "20_30"]
    buffer_labels = {"0_5": "0-5 km", "5_10": "5-10 km", "10_20": "10-20 km", "20_30": "20-30 km"}
    colors = {"0_5": "#fdd0c0", "5_10": "#fca082", "10_20": "#ef6548", "20_30": "#99000d"}

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.05]})
    top_ax, bottom_ax = axes

    # Panel (a): four buffer water-fraction series
    top_handles = []
    for buffer_name in buffers:
        sub = df[df["buffer_km"] == buffer_name].copy()
        top_ax.plot(sub["date"], sub["water_frac"], color=colors[buffer_name], linewidth=2.2, alpha=0.96)
        top_handles.append(Line2D([0], [0], color=colors[buffer_name], linewidth=2.2, label=buffer_labels[buffer_name]))

    # Panel (b): representative-grid residual responses, grouped by unique case grid
    grid_groups = (
        df[["buffer_km", "case_grid_id"]]
        .drop_duplicates()
        .groupby("case_grid_id")["buffer_km"]
        .apply(list)
        .to_dict()
    )
    grid_groups = {
        case_grid_id: sorted(buffer_list, key=lambda item: buffers.index(item))
        for case_grid_id, buffer_list in grid_groups.items()
    }
    grid_palette = ["#1f78b4", "#08306b", "#7f2704", "#4d004b"]
    for idx, (case_grid_id, buffer_list) in enumerate(sorted(grid_groups.items())):
        sub = (
            df.loc[df["buffer_km"] == buffer_list[0], ["date", "case_grid_residual_gwa"]]
            .dropna()
            .copy()
            .sort_values("date")
        )
        if sub.empty:
            continue
        color = grid_palette[idx % len(grid_palette)]
        bottom_ax.plot(sub["date"], sub["case_grid_residual_gwa"], color=color, linewidth=2.3)
    bottom_handles = [
        Line2D(
            [0],
            [0],
            color=grid_palette[idx % len(grid_palette)],
            linewidth=2.3,
            label=f"{case_grid_id}: {', '.join(buffer_labels[b] for b in buffer_list)}",
        )
        for idx, (case_grid_id, buffer_list) in enumerate(sorted(grid_groups.items()))
    ]

    add_panel_label(top_ax, "(a)")
    top_ax.set_title("Buffer-scale surface-water fractions", fontsize=11)
    top_ax.set_ylabel(labels["water_fraction_label"])
    top_ax.legend(
        handles=top_handles,
        title=labels["buffer_legend_title"],
        frameon=True,
        framealpha=0.92,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.9,
    )

    add_panel_label(bottom_ax, "(b)")
    bottom_ax.set_title("Residual response of representative grids", fontsize=11)
    bottom_ax.set_ylabel(labels["residual_gwa_label"])
    bottom_ax.set_xlabel(labels["date_label"])
    bottom_ax.xaxis.set_major_locator(mdates.YearLocator(1))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom_ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    bottom_ax.legend(
        handles=bottom_handles,
        title="Representative grid",
        frameon=True,
        framealpha=0.92,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig07_case_response.png", bbox_inches="tight")
    plt.close(fig)


def draw_fig09(data: dict[str, pd.DataFrame], output_dir: Path, labels: dict[str, object]) -> None:
    df = data["fig08_validation_timeseries"].copy()
    coverage_df = data["fig08_validation_coverage"].copy()
    df["date"] = pd.to_datetime(df["date"])
    coverage = build_grid_geodf(coverage_df)
    fig = plt.figure(figsize=(13.0, 9.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.45, 2.05], wspace=0.16, hspace=0.28)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_map = fig.add_subplot(gs[:, 1])
    for ax in axes[1:]:
        ax.sharex(axes[0])
    region_labels = {"all": labels["all_label"], "north": labels["north_label"], "south": labels["south_label"]}
    remote_color = "#222222"
    well_color = "#2a9d8f"
    post_start = pd.Timestamp("2019-01-01")
    y_mins: list[float] = []
    y_maxs: list[float] = []
    for idx, (ax, region) in enumerate(zip(axes, ["all", "north", "south"])):
        sub = df[df["region_tag"] == region].copy()
        if sub.empty:
            continue
        pair = sub[["gwa_cm", "well_anomaly"]].dropna()
        pair_post = sub.loc[sub["date"] >= post_start, ["gwa_cm", "well_anomaly"]].dropna()
        corr = np.corrcoef(pair["gwa_cm"], pair["well_anomaly"])[0, 1] if len(pair) >= 3 else np.nan
        corr_post = np.corrcoef(pair_post["gwa_cm"], pair_post["well_anomaly"])[0, 1] if len(pair_post) >= 3 else np.nan
        sub["gwa_z"] = (sub["gwa_cm"] - sub["gwa_cm"].mean()) / sub["gwa_cm"].std(ddof=0)
        sub["well_z"] = (sub["well_anomaly"] - sub["well_anomaly"].mean()) / sub["well_anomaly"].std(ddof=0)
        y_mins.extend([float(sub["gwa_z"].min()), float(sub["well_z"].min())])
        y_maxs.extend([float(sub["gwa_z"].max()), float(sub["well_z"].max())])
        add_panel_label(ax, f"({chr(ord('a') + idx)})")
        ax.axvspan(post_start, sub["date"].max(), color="#efe8d8", alpha=0.35, zorder=0)
        ax.axvline(post_start, color="#8f8f8f", linestyle="--", linewidth=0.9, zorder=1)
        ax.plot(sub["date"], sub["gwa_z"], color=remote_color, linewidth=2.1, label=labels["remote_validation_label"])
        ax.plot(sub["date"], sub["well_z"], color=well_color, linewidth=1.8, label=labels["well_validation_label"])
        corr_text = "NA" if np.isnan(corr) else f"{corr:.3f}"
        corr_post_text = "NA" if np.isnan(corr_post) else f"{corr_post:.3f}"
        title_template = labels["pearson_template"]
        try:
            title_text = title_template.format(region=region_labels[region], corr=corr_text, corr_post=corr_post_text)
        except KeyError:
            title_text = f"{region_labels[region]} (r = {corr_text}; 2019+ r = {corr_post_text})"
        ax.set_title(title_text)
        ax.set_ylabel(labels["standardized_label"])
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
    if y_mins and y_maxs:
        ymin = min(y_mins) - 0.15
        ymax = max(y_maxs) + 0.15
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    axes[0].text(
        0.985,
        0.94,
        labels.get("post2019_window_label", "Shaded: post-2019 window"),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.7,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92, "pad": 1.8},
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=remote_color, linewidth=2.1, label=labels["remote_validation_label"]),
            Line2D([0], [0], color=well_color, linewidth=1.8, label=labels["well_validation_label"]),
        ],
        frameon=True,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    axes[-1].set_xlabel(labels["date_label"])
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax_map.grid(False)
    draw_regional_basemap(ax_map, labels["regional_labels"], show_labels=False)
    no_data = coverage.loc[coverage["unique_wells"] <= 0].copy()
    has_data = coverage.loc[coverage["unique_wells"] > 0].copy()
    if not no_data.empty:
        no_data.plot(ax=ax_map, facecolor="#d9d9d9", edgecolor="#8a8a8a", linewidth=0.75, alpha=0.78, zorder=2)
    if not has_data.empty:
        vmax = float(has_data["unique_wells"].max())
        norm = colors.Normalize(vmin=0, vmax=vmax)
        has_data.plot(
            ax=ax_map,
            column="unique_wells",
            cmap="YlGnBu",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
            legend=False,
        )
        sm = cm.ScalarMappable(norm=norm, cmap="YlGnBu")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_map, fraction=0.045, pad=0.025, shrink=0.58, aspect=18)
        cbar.set_label(labels["well_coverage_label"])
    coverage.boundary.plot(ax=ax_map, color="#707070", linewidth=0.55, zorder=4)
    for label, x, y in labels["regional_labels"]:
        ax_map.text(
            x,
            y,
            label,
            fontsize=8.4,
            color="#4b5563",
            ha="center",
            va="center",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.65},
        )
    add_panel_label(ax_map, "(d)")
    ax_map.set_title(labels["well_coverage_title"])
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(111.9, 120.1)
    ax_map.set_ylim(34.3, 40.8)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / "fig08_validation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.figure_table_dir)
    out = Path(args.output_dir)
    ensure_dir(out)
    set_plot_style(args.language)
    labels = LANG[args.language]
    data = load_csvs(base)
    draw_fig01_v3(data, out, labels)
    draw_fig02(data, out, labels)
    draw_fig03(data, out, labels)
    draw_fig04(data, out, labels)
    draw_fig05(data, out, labels)
    draw_fig06(data, out, labels)
    draw_fig07(data, out, labels)
    draw_fig08(data, out, labels)
    draw_fig09(data, out, labels)


if __name__ == "__main__":
    main()
