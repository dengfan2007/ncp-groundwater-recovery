from __future__ import annotations

import argparse
import json

from ee_local import ee_smoke_summary, initialize_ee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Authenticate and smoke-test local Earth Engine Python access.")
    parser.add_argument("--project", help="Google Cloud project for Earth Engine initialization.")
    parser.add_argument(
        "--auth-mode",
        help="Optional ee.Authenticate(auth_mode=...) override, e.g. localhost or gcloud.",
    )
    parser.add_argument(
        "--force-auth",
        action="store_true",
        help="Force re-authentication even if cached credentials already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = initialize_ee(
        project=args.project,
        auth_mode=args.auth_mode,
        force_auth=args.force_auth,
    )
    summary = {"project": project}
    summary.update(ee_smoke_summary())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
