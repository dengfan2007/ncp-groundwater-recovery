from __future__ import annotations

import argparse
from pathlib import Path

from ee_hutuo_case_local_export import main as case_main
from ee_ncp_grid_local_export import main as grid_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all core Earth Engine tables directly to local CSV files.")
    parser.add_argument("--project", required=True, help="Earth Engine Cloud project.")
    parser.add_argument("--auth-mode", help="Optional Earth Engine auth mode, e.g. localhost.")
    parser.add_argument("--force-auth", action="store_true", help="Force re-authentication.")
    parser.add_argument("--output-dir", default="data/raw", help="Local output directory.")
    parser.add_argument("--describe", action="store_true", help="Describe collections only; do not download.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload all monthly chunks even if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = ["--project", args.project, "--output-dir", str(output_dir)]
    if args.auth_mode:
        base_args.extend(["--auth-mode", args.auth_mode])
    if args.force_auth:
        base_args.append("--force-auth")
    if args.describe:
        base_args.append("--describe")
    if args.overwrite:
        base_args.append("--overwrite")

    import sys

    original_argv = list(sys.argv)
    try:
        sys.argv = ["ee_ncp_grid_local_export.py", *base_args]
        grid_main()
        sys.argv = ["ee_hutuo_case_local_export.py", *base_args]
        case_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
