# Data Access and Provenance

Access inquiries should be directed to the corresponding author (dengfan@yangtzeu.edu.cn).

| Resource | Source and Access Route | Deposited Scope |
| --- | --- | --- |
| JPL GRACE mascons | NASA (https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON_CRI) | Extraction scripts and frozen gridded study panel. |
| GLDAS-2.1 Noah | NASA (https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H) | Extraction scripts and frozen gridded soil moisture and snow storage series. |
| ERA5-Land | Copernicus/ECMWF (https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR) | Extraction scripts and frozen gridded precipitation and evaporation diagnostics. |
| Groundwater monitoring | Long et al. (2025; https://doi.org/10.1038/s41467-025-62719-5; https://github.com/YuanchengXu2718/GroundwaterNCP_open) | Aggregated grid and regional summaries are deposited. Raw/individual well records, IDs, coordinates, and cohort lists are available from the upstream repository. |
| Surface-water occurrence | Pekel et al. (2016; https://doi.org/10.1038/nature20584; https://global-surface-water.appspot.com/) | Gridded diagnostic summaries. |
| Natural Earth | Natural Earth (https://www.naturalearthdata.com/about/terms-of-use/) | Acquisition and rendering scripts for public-domain cartographic boundary data. |
| Regional hydrogeology & base layers | BGR/UNESCO WHYMAP, SRTM/GAUL | Cartographic acquisition and drawing code is provided in `scripts/build_hydrogeology_maps.py`. See `docs/map_sources.md`. |

The file `resources_manifest.json` provides checksums (SHA-256) and relative paths for all deposited files. `excluded_resources.json` documents the provenance and hashes of upstream individual-well and raw simulation files that require source access or can be regenerated deterministically.

For reuse conditions of project-authored code and derived summaries, consult [RIGHTS.md](../RIGHTS.md).
