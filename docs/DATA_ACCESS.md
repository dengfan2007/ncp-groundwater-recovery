# Data access, provenance and exclusions

Checked 2026-10-19.

The repository is temporarily private at the authors' request. Access enquiries go to the corresponding author; the curation exclusions below also apply to authorized copies. A repository URL does not currently provide anonymous public access.

| Resource | Access and deposited scope |
| --- | --- |
| JPL GRACE mascons | NASA; extraction code and frozen gridded study panel are supplied. Product IDs, units, missing-month masks and source hashes remain in extraction manifests. https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON_CRI |
| GLDAS-2.1 Noah | NASA; frozen study extractions and code are supplied. https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H |
| ERA5-Land | Copernicus/ECMWF; frozen gridded study extractions and component diagnostics are supplied, with their provenance. https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR |
| Groundwater monitoring | Long et al. (2025), https://doi.org/10.1038/s41467-025-62719-5 and https://github.com/YuanchengXu2718/GroundwaterNCP_open. The current upstream README offers 2005-2017 publicly and states that 2018-2024 are available from the corresponding author. No repository-level licence was visible at the upstream root. Obtain those records and reuse permission directly from the source. This deposit excludes raw/converted well records, per-well identifiers, locations, coverage and cohort membership; it retains aggregate grid/region results. |
| Surface-water occurrence | Pekel et al. (2016), https://doi.org/10.1038/nature20584; gridded diagnostic summaries only. Product information: https://global-surface-water.appspot.com/ |
| Natural Earth | Public-domain source per https://www.naturalearthdata.com/about/terms-of-use/. Acquisition/drawing code is supplied. |
| WHYMAP, lowland mask and UN reference map | Mixed-source cartographic inputs are not redistributed in this deposit. Sources/access routes are recorded in `docs/map_sources.md` and the map acquisition script. The UN reference map was used only for visual comparison; it is not a deposited analysis input. |

`resources_manifest.json` identifies every deposited file and source path. `excluded_resources.json` records excluded candidate evidence paths, reasons and SHA-256 without exposing their contents. The public evidence archive is an intentionally bounded research deposit, not the complete local project or a substitute for source-data permissions. Frozen gridded products and aggregate outputs retain their original source acknowledgements and conditions.

For the deposited project code/derived results, consult RIGHTS.md and contact the corresponding author for reuse terms. No permissions for third-party materials are granted by this repository.
