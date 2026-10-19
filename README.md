# North China Plain groundwater recovery: analysis resources

Resources supporting **Identifying groundwater recovery in the North China Plain: regional trends and the stability of inferred timing**, by Hui Li, Fan Deng, Yanxin Chen, Yong Wang, Longwei Xiang and Ran Jing.

Repository: https://github.com/dengfan2007/ncp-groundwater-recovery

Resource version: `v1.0.0` (2026-10-19). This is a research resource release; it does not indicate journal acceptance or submission.

## Contents

- `analysis/`: processing, candidate selection, known-truth simulations, endpoint experiments, diagnostics and existing tests. Historical dependencies are retained where needed; manuscript generators are excluded.
- `scripts/`: figure renderers and resource verification. The three revised figure renderers recreate the endpoint, example-candidate and matched-validation figures directly from frozen evidence without fitting models.
- `gee/`: satellite/model extraction scripts, requiring the reader's own authorized Earth Engine project.
- `docs/`: reproduction instructions, evidence map, data access and rights information, and the recorded computation environment.
- `resources_manifest.json`: relative paths, byte sizes and SHA-256 of code and frozen inputs/results.
- Release asset `ncp-groundwater-recovery-evidence-v1.0.0.zip`: frozen gridded satellite/model inputs, complete real-data and simulation candidate pools, trial truth and outcomes, endpoint histories, diagnostic results and aggregate figure source tables. Extract at the repository root, preserving paths. Individual-well records and locations are excluded.

Start with [reproduction instructions](docs/REPRODUCING.md) and the [evidence map](docs/RESOURCE_MAP.md). Run `python scripts/verify_resources.py` after extracting the asset.

## Interpretation

The residual groundwater-storage proxy retains surface-water storage. Its cm/yr units describe equivalent water thickness, whereas well changes are in metres of hydraulic head. Dates are exploratory candidates: no tested threshold grid achieved the 5% false-report criterion across the null scenarios. North-south contrasts depend on components, endpoints and spatial blocks; the larger-block interval includes zero. Aggregated well evidence is weaker in the north. These resources do not support precise recovery months or strong causal attribution.

## Access and reuse

The source well repository currently publishes 2005-2017 records and directs requests for 2018-2024 records to the original corresponding author. Individual-well data, identifiers, coordinates and cohort membership are not redistributed here. Aggregate well comparisons are included; full well-level reruns require authorized access to the source records. See [data access and rights](docs/DATA_ACCESS.md). No archival DOI has been assigned. Use the tagged version and commit to identify this deposit.

No blanket licence is asserted for mixed-source materials. See [RIGHTS.md](RIGHTS.md). Credentials, personal configuration, internal review files, historical manuscript versions, journal instructions and submission documents are excluded.
