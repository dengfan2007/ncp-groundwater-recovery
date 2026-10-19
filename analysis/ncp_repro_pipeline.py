from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
STANDARD_OUTPUT_DIRS = [
    "qc",
    "main",
    "robustness",
    "postprocess",
    "figure_tables",
    "figure_tables_en",
    "figures",
    "figures_en",
    "tables",
    "tables_en",
    "manuscript",
    "manuscript_en",
    "revision",
]
MANIFEST_PACKAGES = [
    "pandas",
    "numpy",
    "statsmodels",
    "scipy",
    "Rbeast",
    "earthengine-api",
    "matplotlib",
    "geopandas",
    "shapely",
    "openpyxl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local end-to-end reproducibility pipeline for the NCP paper.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing local raw CSV inputs.")
    parser.add_argument("--external-dir", default="data/external", help="Directory containing external validation inputs.")
    parser.add_argument("--output-dir", default="outputs", help="Root output directory.")
    parser.add_argument(
        "--manuscript",
        default="docs/ncp_full_manuscript_draft.md",
        help="Path to the markdown-like manuscript source.",
    )
    parser.add_argument("--well-data", help="Optional override path for the public well-proxy CSV.")
    parser.add_argument(
        "--water-overlap",
        help="Optional path to the JRC-Landsat overlap CSV for surface-water continuity diagnostics.",
    )
    parser.add_argument("--compile-pdf", action="store_true", help="Compile the generated TeX into PDF.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Language for rendered figures, tables, and manuscript outputs.")
    parser.add_argument(
        "--archive-standard-outputs",
        action="store_true",
        help="Move existing standard output directories into outputs/archive before rerunning.",
    )
    return parser.parse_args()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in MANIFEST_PACKAGES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def archive_standard_outputs(output_root: Path) -> str | None:
    existing_paths = [output_root / name for name in STANDARD_OUTPUT_DIRS if (output_root / name).exists()]
    if not existing_paths:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = output_root / "archive" / f"{timestamp}_pre_beast_required"
    archive_root.mkdir(parents=True, exist_ok=False)
    for source in existing_paths:
        shutil.move(str(source), archive_root / source.name)
    return str(archive_root)


def run_step(name: str, command: list[str], *, steps: list[dict[str, object]]) -> None:
    started = time.time()
    print(f"[repro] {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    steps.append(
        {
            "name": name,
            "command": command,
            "duration_seconds": round(time.time() - started, 3),
        }
    )


def resolve_well_inputs(args: argparse.Namespace, external_dir: Path, output_root: Path, steps: list[dict[str, object]]) -> tuple[Path, str | None]:
    if args.well_data:
        return Path(args.well_data).resolve(), None

    workbook_path = external_dir / "GroundwaterNCP_open" / "GroundwaterDepth.xlsx"
    if workbook_path.exists():
        normalized_csv = external_dir / "ncp_long2025_well_2005_2024.csv"
        run_step(
            "public_wells",
            [
                sys.executable,
                "analysis/ncp_long2025_to_wellcsv.py",
                "--excel",
                str(workbook_path),
                "--output",
                str(normalized_csv),
            ],
            steps=steps,
        )
        return normalized_csv, "aquifer_group == 'unconfined'"

    return external_dir / "ncp_obsgrid_wellproxy_2005_2018.csv", None


def main() -> None:
    args = parse_args()
    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    external_dir = (REPO_ROOT / args.external_dir).resolve()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    manuscript_path = (REPO_ROOT / args.manuscript).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, object]] = []
    well_path, well_query = resolve_well_inputs(args, external_dir, output_root, steps)
    if not well_path.exists():
        raise FileNotFoundError(f"Required well-proxy file was not found: {well_path}")

    archived_path = archive_standard_outputs(output_root) if args.archive_standard_outputs else None
    python_exe = sys.executable
    suffix = "_en" if args.language == "en" else ""
    figure_tables_dir = output_root / f"figure_tables{suffix}"
    figures_dir = output_root / f"figures{suffix}"
    tables_dir = output_root / f"tables{suffix}"
    manuscript_dir = output_root / f"manuscript{suffix}"

    run_step(
        "qc",
        [
            python_exe,
            "analysis/ncp_data_qc.py",
            "--panel",
            str(raw_dir / "grid_monthly_panel.csv"),
            "--grid",
            str(raw_dir / "ncp_grid_definition.csv"),
            "--case",
            str(raw_dir / "hutuo_case_buffers_raw.csv"),
            "--output-dir",
            str(output_root / "qc"),
        ],
        steps=steps,
    )
    run_step(
        "main",
        [
            python_exe,
            "analysis/ncp_recovery_analysis.py",
            "--panel",
            str(raw_dir / "grid_monthly_panel.csv"),
            "--case",
            str(raw_dir / "hutuo_case_buffers_raw.csv"),
            "--output-dir",
            str(output_root / "main"),
        ],
        steps=steps,
    )
    run_step(
        "robustness",
        [
            python_exe,
            "analysis/ncp_experiment_suite.py",
            "--panel",
            str(raw_dir / "grid_monthly_panel.csv"),
            "--case",
            str(raw_dir / "hutuo_case_buffers_raw.csv"),
            "--output-dir",
            str(output_root / "robustness"),
        ],
        steps=steps,
    )
    run_step(
        "postprocess",
        [
            python_exe,
            "analysis/ncp_trajectory_postprocess.py",
            "--panel",
            str(raw_dir / "grid_monthly_panel.csv"),
            "--counterfactual",
            str(output_root / "main" / "grid_counterfactual.csv"),
            "--breakpoints",
            str(output_root / "main" / "grid_breakpoints.csv"),
            "--case",
            str(output_root / "main" / "hutuo_case_buffers.csv"),
            "--well-data",
            str(well_path),
            *(
                [
                    "--well-query",
                    well_query,
                ]
                if well_query
                else []
            ),
            "--output-dir",
            str(output_root / "postprocess"),
        ],
        steps=steps,
    )
    run_step(
        "revision",
        [
            python_exe,
            "analysis/ncp_revision_analyses.py",
            "--postprocess-dir",
            str(output_root / "postprocess"),
            "--main-dir",
            str(output_root / "main"),
            "--panel",
            str(raw_dir / "grid_monthly_panel.csv"),
            "--well-data",
            str(well_path),
            *(
                [
                    "--well-query",
                    well_query,
                ]
                if well_query
                else []
            ),
            *(
                [
                    "--water-overlap",
                    str((REPO_ROOT / args.water_overlap).resolve()),
                ]
                if args.water_overlap
                else []
            ),
            "--output-dir",
            str(output_root / "revision"),
        ],
        steps=steps,
    )
    run_step(
        "figure_tables",
        [
            python_exe,
            "analysis/ncp_prepare_figure_tables.py",
            "--raw-dir",
            str(raw_dir),
            "--main-dir",
            str(output_root / "main"),
            "--postprocess-dir",
            str(output_root / "postprocess"),
            "--revision-dir",
            str(output_root / "revision"),
            "--external-dir",
            str(external_dir),
            "--well-data",
            str(well_path),
            *(
                [
                    "--well-query",
                    well_query,
                ]
                if well_query
                else []
            ),
            "--language",
            args.language,
            "--output-dir",
            str(figure_tables_dir),
        ],
        steps=steps,
    )
    run_step(
        "figures",
        [
            python_exe,
            "analysis/ncp_plot_figures.py",
            "--figure-table-dir",
            str(figure_tables_dir),
            "--language",
            args.language,
            "--output-dir",
            str(figures_dir),
        ],
        steps=steps,
    )
    run_step(
        "workflow_figure",
        [
            python_exe,
            "analysis/ncp_draw_workflow.py",
            "--output",
            str(figures_dir / "fig09_technical_route.png"),
            "--language",
            args.language,
        ],
        steps=steps,
    )
    run_step(
        "tables",
        [
            python_exe,
            "analysis/ncp_render_tables.py",
            "--figure-table-dir",
            str(figure_tables_dir),
            "--language",
            args.language,
            "--output-dir",
            str(tables_dir),
        ],
        steps=steps,
    )

    manuscript_command = [
        python_exe,
        "analysis/ncp_build_tex.py",
        "--manuscript",
        str(manuscript_path),
        "--figures-dir",
        str(figures_dir),
        "--tables-dir",
        str(tables_dir),
        "--output-dir",
        str(manuscript_dir),
    ]
    if args.compile_pdf:
        manuscript_command.append("--compile-pdf")
    run_step("manuscript", manuscript_command, steps=steps)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "archived_previous_outputs": archived_path,
        "compile_pdf": args.compile_pdf,
        "inputs": {
            "raw_dir": str(raw_dir),
            "external_dir": str(external_dir),
            "well_data": str(well_path),
            "water_overlap": str((REPO_ROOT / args.water_overlap).resolve()) if args.water_overlap else None,
            "manuscript": str(manuscript_path),
            "language": args.language,
        },
        "package_versions": package_versions(),
        "steps": steps,
    }
    (output_root / "repro_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
