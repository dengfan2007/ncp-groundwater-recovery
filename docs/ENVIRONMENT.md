# Computation Environment

## Python Runtime

Analyses were executed using Python 3.11.7 with the following package versions:

- `numpy`: 1.26.4
- `scipy`: 1.11.4
- `pandas`: 2.3.3 (or 2.1.4)
- `statsmodels`: 0.14.0
- `Rbeast`: 0.1.25
- `matplotlib`: 3.8.0
- `geopandas`: 1.1.3
- `shapely`: 2.1.2

Signal decomposition analysis (`analysis/ncp_story_analysis.py`) additionally utilizes `EMD-signal` 1.10.0.

## Dependency Installation

Dependencies are listed in `analysis/requirements.txt` and can be installed via:

```bash
pip install -r analysis/requirements.txt
```

To install `EMD-signal` for signal reconstruction:

```bash
pip install EMD-signal==1.10.0
```

## Computational Notes

- Bayesian change-point analyses (BEAST fits) across 29,200 grid/realization trials are computationally intensive. Precomputed trial-level results and threshold grids are provided in the evidence archive.
- Scripts use deterministic random seeds to ensure reproducibility across runs.
