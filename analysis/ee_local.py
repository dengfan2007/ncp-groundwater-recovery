from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ee


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def initialize_ee(
    project: str | None = None,
    auth_mode: str | None = None,
    force_auth: bool = False,
) -> str:
    """Initialize Earth Engine for local Python use.

    Supported modes:
    - Interactive OAuth: provide `project`, optionally `auth_mode`.
    - Service account: set `EE_SERVICE_ACCOUNT` and `EE_PRIVATE_KEY_FILE`.
    """
    project = project or _env("EE_PROJECT")
    if not project:
        raise ValueError("Missing Earth Engine project. Set EE_PROJECT or pass project=...")

    service_account = _env("EE_SERVICE_ACCOUNT")
    private_key_file = _env("EE_PRIVATE_KEY_FILE")

    if service_account or private_key_file:
        if not service_account or not private_key_file:
            raise ValueError(
                "Service account auth requires both EE_SERVICE_ACCOUNT and EE_PRIVATE_KEY_FILE."
            )
        key_path = Path(private_key_file)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file does not exist: {key_path}")
        credentials = ee.ServiceAccountCredentials(service_account, str(key_path))
        ee.Initialize(credentials=credentials, project=project)
        return project

    ee.Authenticate(auth_mode=auth_mode, force=force_auth)
    ee.Initialize(project=project)
    return project


def ee_smoke_summary() -> dict[str, Any]:
    """Run a small Earth Engine request to verify connectivity."""
    point = ee.Geometry.Point([116.3913, 39.9075])
    grace_count = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI").limit(5).size().getInfo()
    sample_value = ee.Number(1).add(2).getInfo()
    point_lonlat = point.coordinates().getInfo()
    return {
        "number_test": sample_value,
        "grace_sample_count": grace_count,
        "point_lonlat": point_lonlat,
    }
