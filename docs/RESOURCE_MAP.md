# Evidence map

All paths below are relative to the repository root after extracting the release asset.

| Paper component | Deposited files/code |
| --- | --- |
| Original gridded satellite/model inputs | `data/raw/grid_monthly_panel.csv`, `data/raw/ncp_grid_definition.csv`; `gee/ncp_grid_monthly_panel.js` |
| Calendar/coverage checks | `outputs/calendar_revision/2026-10-12-r4/`; individual-well coverage excluded |
| Trend and component contrasts | `outputs/review_revision/2026-10-15-r1/`, `outputs/validation_revision/2026-10-17-r2/`, `outputs/monitoring_revision/2026-10-18-r1/diagnostics/` |
| Figure 1 grid counts/context | `outputs/figure_revision/2026-10-17-final/tables/fig01_analysis_cells.csv`; point wells and mixed-source map layers require source access |
| Figure 2 trends | `outputs/monitoring_revision/2026-10-18-r1/figure_tables/`, `outputs/monitoring_revision/2026-10-19-figure-tables/`; `analysis/ncp_monitoring_figures.py` |
| Figure 3 endpoint history | `outputs/monitoring_revision/2026-10-18-r1/inference/real_selections.csv`, `endpoint_date_contrasts.csv`; `scripts/render_hydrogeology_figure3.py` |
| Figure 4 examples and supplement S24 | `outputs/monitoring_revision/2026-10-18-r1/processing/`; `scripts/render_hydrogeology_figure4.py`; `outputs/current_figure_checks/figure4_evidence/` |
| Figure 5 matched validation | `outputs/monitoring_revision/2026-10-18-r1/calibration/`, `two_stage/`, `inference/`; `scripts/render_hydrogeology_figure5.py`; `outputs/current_figure_checks/figure5_evidence/` |
| Figure 6 and well comparisons | Aggregate series/contrasts in diagnostics and `outputs/monitoring_revision/2026-10-19-cell-trends/`; raw-well reruns require source access |
| Trial-level truth, full candidate pools and selection rules | `outputs/monitoring_revision/2026-10-18-r1/calibration/`, `two_stage/`, `inference/rules.json`; `analysis/ncp_monitoring_core.py`, `ncp_monitoring_revision.py`, `ncp_monitoring_inference.py` |
| Formal-run code provenance | `outputs/monitoring_revision/2026-10-18-r1/source_snapshots/` (Python/JSON scientific snapshots only) |
| Earlier controlled comparisons | `outputs/story_revision/2026-10-17-r1/`, `outputs/validation_revision/2026-10-17-r2/` |

Earlier evidence is retained as dependencies and provenance, not as an instruction to replace the latest figure scripts. The 2026-10-19 directory labels arrived in an author-returned package received locally on 2026-10-18; labels are preserved rather than reinterpreted as execution timestamps.
