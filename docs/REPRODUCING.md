# Reproducing the deposited results

The repository is currently private. The following download instructions require authorized repository access; contact the corresponding author about access. Full well-level reruns also require the separate source-data permissions described in DATA_ACCESS.md.

## 1. Obtain and verify

Clone https://github.com/dengfan2007/ncp-groundwater-recovery and check out `v1.0.0`. Download `ncp-groundwater-recovery-evidence-v1.0.0.zip` from that release. Extract its contents at the repository root; it supplies `data/` and `outputs/` at the paths expected by the code.

For the frozen `v1.0.0` source archive, also download the release attachment `resources_manifest.json` and place it at the extracted code root before verification. This corrects the byte size/hash of `excluded_resources.json` after LF newline normalization on GitHub; it changes no research payload. The main branch already carries the corrected checksums and current private-access notes. The tag remains unchanged.

Run from the repository root:

```text
python scripts/verify_resources.py
```

The verifier checks every published resource's size and SHA-256 without importing scientific packages or making network requests. Manifests exclude themselves. Checksums establish file integrity, not scientific correctness.

## 2. Environment

The recorded analysis used Python 3.11.7, numpy 1.26.4, scipy 1.11.4, pandas imported version 2.3.3, statsmodels 0.14.0, Rbeast 0.1.25 and matplotlib 3.8.0. The historical environment reported inconsistent pandas distribution metadata (2.1.4); see `docs/ENVIRONMENT.md`. These are recorded versions, not a claim of a newly tested portable environment lock. Plotting requires an installed matplotlib backend and fonts; Arial was used for the archived figures.

Full reconstruction additionally requires EMD-signal 1.10.0. Install it into `outputs/runs/2026-10-16-story-runtime-r1/` using your package manager (for example `python -m pip install --no-deps --target outputs/runs/2026-10-16-story-runtime-r1 EMD-signal==1.10.0`). The third-party runtime is not vendored in this deposit. Earth Engine is needed only for new extractions. No existing environment needs to be upgraded to inspect CSV/JSON evidence.

## 3. Recreate the revised figures from frozen evidence

Use a scientific environment with pandas, numpy and matplotlib. Each output directory must be new:

```text
python scripts/render_hydrogeology_figure3.py --output outputs/runs/reproduce-figure3
python scripts/render_hydrogeology_figure4.py --output outputs/runs/reproduce-figure4
python scripts/render_hydrogeology_figure5.py --output outputs/runs/reproduce-figure5
```

These scripts recheck counts, conditional contrasts, H/E candidate selection and slopes, or paired held-out false-report fractions before drawing. They do not fit BEAST, rerun simulations or tune thresholds. Figure 5 uses the same held-out trials for both fixed rules. Rendered PDFs may differ in metadata/font handling across platforms.

The general renderer `analysis/ncp_monitoring_figures.py` retains its historical figure versions. Use the three scripts above for the revised Figures 3-5. The original map and point-level well figure cannot be fully regenerated from this public deposit: point coordinates and mixed-source GIS layers are intentionally omitted. Map acquisition and drawing code is retained as an access route, with source requirements in DATA_ACCESS.md. Do not interpret the public package as a self-contained well-level or complete cartographic rerun.

## 4. Scientific reruns

Existing tests are in `analysis/test_monitoring_revision.py`, `test_calendar_time.py`, `test_review_revision.py` and `test_signal_reconstruction.py`. Tests with well-level dependencies require separately obtained source records.

Formal model runs are computationally expensive (29,200 BEAST fits, excluding pilot/technical checks). Preserve the deposited evidence. In PowerShell, point new runs at new directories before starting Python:

```powershell
$resourceRoot = (Get-Location).Path
$env:NCP_MONITORING_EVIDENCE = Join-Path $resourceRoot 'outputs/monitoring_revision/reproduction-01'
$env:NCP_MONITORING_RUN_DIR = Join-Path $resourceRoot 'outputs/runs/reproduction-01'
python analysis/ncp_monitoring_revision.py --help
```

Consult the stage runner and retained source snapshots before invoking baseline, repeatability, diagnostic, pilot, calibration, two-stage and inference stages. Repeatability/seed gates precede simulations. Preserve calibration/validation separation. Extraction requires the reader's own access and project; frozen extraction CSVs are supplied for inspection. Well stages require the original authorized records and expected local conversion, and the 2018 common-window sensitivity must be distinguished from the unsuccessful 2005-start coverage check.

No new fitting was performed when preparing this release. Cross-platform bitwise repeatability and a full well-data rerun have not been established by this publication workflow. Some rank correlations have recorded floating-point tie portability limitations; the frozen evidence is retained.
