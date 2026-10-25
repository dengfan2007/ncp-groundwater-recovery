# Reproducing Results

## 1. Obtain and Verify

Clone https://github.com/dengfan2007/ncp-groundwater-recovery and check out `v1.0.0` (or `main`). Download `ncp-groundwater-recovery-evidence-v1.0.0.zip` from the `v1.0.0` release. Extract its contents at the repository root to populate the `data/` and `outputs/` directories.

Run verification from the repository root:

```text
python scripts/verify_resources.py
```

The verifier checks every published resource's size and SHA-256 using only the Python standard library.

## 2. Environment Setup

Analyses were performed with Python 3.11.7. Key dependencies include:

- `numpy` 1.26.4
- `scipy` 1.11.4
- `pandas` 2.3.3 (or 2.1.4)
- `statsmodels` 0.14.0
- `Rbeast` 0.1.25
- `matplotlib` 3.8.0

Signal decomposition routines additionally require `EMD-signal` 1.10.0 (`pip install EMD-signal==1.10.0`). See `docs/ENVIRONMENT.md` for full details.

## 3. Recreate Figures from Frozen Evidence

Run the dedicated rendering scripts using Python:

```text
python scripts/render_hydrogeology_figure3.py --output outputs/runs/reproduce-figure3
python scripts/render_hydrogeology_figure4.py --output outputs/runs/reproduce-figure4
python scripts/render_hydrogeology_figure5.py --output outputs/runs/reproduce-figure5
```

These scripts verify candidate selection, slopes, conditional contrasts, and paired held-out false-report fractions before drawing.

The general renderer `analysis/ncp_monitoring_figures.py` generates the remaining figures. Map acquisition and drawing code is provided in `scripts/build_hydrogeology_maps.py`.

## 4. Scientific Execution and Model Fits

Existing test suites can be executed via:

```text
python analysis/test_monitoring_revision.py
python analysis/test_calendar_time.py
python analysis/test_review_revision.py
python analysis/test_signal_reconstruction.py
```

To execute model runs or candidate searches, specify isolated output directories:

```powershell
$resourceRoot = (Get-Location).Path
$env:NCP_MONITORING_EVIDENCE = Join-Path $resourceRoot 'outputs/monitoring_revision/reproduction-01'
$env:NCP_MONITORING_RUN_DIR = Join-Path $resourceRoot 'outputs/runs/reproduction-01'
python analysis/ncp_monitoring_revision.py --help
```

Multi-seed refits, native-mascon inference, and local slope-rule evaluations are provided in `analysis/ncp_r9_seed_checks.py`, `ee_r9_mascon_membership.py`, `ncp_r9_mascon_inference.py`, and `ncp_r9_local_rule.py`, with precomputed outputs in `outputs/r9_checks/2026-10-19-r9-r1/`.
