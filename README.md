# North China Plain Groundwater Recovery: Analysis Resources

Code and evidence archive supporting **Identifying groundwater recovery in the North China Plain: regional trends and the stability of inferred timing**, by Hui Li, Fan Deng, Yanxin Chen, Yong Wang, Longwei Xiang, and Ran Jing.

- Repository: https://github.com/dengfan2007/ncp-groundwater-recovery
- Version: `v1.0.0`

## Access

Repository and release downloads require authorized access. Inquiries regarding access to deposited materials should be directed to corresponding author Fan Deng (dengfan@yangtzeu.edu.cn), subject to third-party data conditions described in [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

## Contents

- `analysis/`: Core data processing, candidate selection, synthetic simulations, endpoint stability experiments, multi-seed repeatability checks, native mascon inference, local slope-rule evaluations, diagnostics, and test suites.
- `scripts/`: Figure rendering scripts and resource verification utilities. The scripts recreate the endpoint, candidate selection, and matched-validation figures directly from frozen evidence.
- `gee/`: Satellite and model extraction scripts for Google Earth Engine.
- `docs/`: Reproduction instructions, evidence map, data access terms, map source descriptions, and computation environment details.
- `resources_manifest.json`: Relative paths, byte sizes, and SHA-256 checksums of code and deposited evidence.
- Release asset `ncp-groundwater-recovery-evidence-v1.0.0.zip`: Precomputed gridded satellite/model inputs, real-data candidate pools, trial-level simulation truth and selection outcomes, calibration and local-rule evaluations, multi-seed and native mascon evidence, endpoint histories, diagnostic results, and aggregate figure source tables. Extract at the repository root, preserving directory structure.

To begin, consult the [reproduction instructions](docs/REPRODUCING.md) and the [evidence map](docs/RESOURCE_MAP.md). Run `python scripts/verify_resources.py` after extracting the release asset to verify file integrity.

## Scientific Interpretation Boundaries

The residual groundwater-storage proxy retains surface-water storage. Units of cm/yr represent equivalent water thickness, whereas well water levels are measured in metres of hydraulic head. Inferred turnaround dates are exploratory candidates: no tested threshold grid achieved the 5% false-alarm rate across all null scenarios. North-south differences depend on components, endpoints, and spatial aggregation; the larger-block confidence interval includes zero. Aggregate well validation is weaker in the northern plain. These data do not support exact monthly timing or strong causal attribution.

## Data Access and Reuse

Groundwater monitoring records are curated by Long et al. (2025; https://github.com/YuanchengXu2718/GroundwaterNCP_open). Individual-well coordinates and raw time series are subject to source conditions and are not redistributed in this repository; aggregate cell-level summaries are provided. Full well-level analyses require access to original records as detailed in [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

For licensing and terms of reuse, see [RIGHTS.md](RIGHTS.md).
