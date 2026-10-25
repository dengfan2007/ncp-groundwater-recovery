# Evidence Map

All paths below are relative to the repository root after extracting the release asset.

| Paper Component | Deposited Files and Code |
| --- | --- |
| Original gridded satellite/model inputs | `data/raw/grid_monthly_panel.csv`, `data/raw/ncp_grid_definition.csv`; `gee/ncp_grid_monthly_panel.js` |
| Component closure and uniform-support validation (Table S2) | `outputs/validation_revision/2026-10-17-r2/extraction_comparison.csv`, `extraction/uniform_support_panel.csv`, `uniform/uniform_calendar_panel.csv`; `analysis/ee_ncp_closure_reextract.py` |
| Controlled routes and bounded-setting sensitivity (Table 1; Tables S3-S6) | `outputs/story_revision/2026-10-17-r1/actual_summary.csv`, `settings/timing_contrasts.csv`, `settings/settings_selections.csv`, `outputs/validation_revision/2026-10-17-r2/ablation_step_summary.csv`; `analysis/ncp_story_analysis.py`, `ncp_story_sensitivity.py` |
| Trend, estimator, boundary and precipitation sensitivity (Tables S7-S9) | `outputs/monitoring_revision/2026-10-18-r1/diagnostics/`; `analysis/ncp_monitoring_diagnostics.py` |
| Native JPL mascon footprints and contrasts (Table S10) | `outputs/r9_checks/2026-10-19-r9-r1/mascon_r2/`; `analysis/ee_r9_mascon_membership.py`, `ncp_r9_mascon_inference.py` |
| Calibration, held-out validation and local slope rule (Tables S11-S12; S17-S18) | `outputs/monitoring_revision/2026-10-18-r1/calibration/`, `two_stage/`, `inference/`; `outputs/r9_checks/2026-10-19-r9-r1/local_rule/`; `analysis/ncp_monitoring_inference.py`, `ncp_r9_local_rule.py` |
| Multi-seed repeatability, endpoint revisions and land-model alternatives (Tables S13-S16) | `outputs/r9_checks/2026-10-19-r9-r1/seeds/`, `seeds_local_rule/`; `outputs/monitoring_revision/2026-10-18-r1/inference/`, `land_models/`; `analysis/ncp_r9_seed_checks.py`, `ncp_monitoring_models.py` |
| Fixed wells, coverage and storage-well correspondence (Tables S19-S22; Fig. 6) | `outputs/monitoring_revision/2026-10-18-r1/well_dates/`, `well_dates_2018/`, `diagnostics/`; `outputs/monitoring_revision/2026-10-19-cell-trends/`; `analysis/ncp_monitoring_observations.py`, `ncp_cell_well_trends.py` |
| Figure 1 grid counts and context | `outputs/figure_revision/2026-10-17-final/tables/fig01_analysis_cells.csv`; point wells and mixed-source map layers require source access |
| Figure 2 trends | `outputs/monitoring_revision/2026-10-18-r1/figure_tables/`, `outputs/monitoring_revision/2026-10-19-figure-tables/`; `analysis/ncp_monitoring_figures.py` |
| Figure 3 endpoint history | `outputs/monitoring_revision/2026-10-18-r1/inference/real_selections.csv`, `endpoint_date_contrasts.csv`; `scripts/render_hydrogeology_figure3.py`; `outputs/current_figure_checks/figure3_evidence/` |
| Figure 4 examples and candidate audit (Table S23) | `outputs/monitoring_revision/2026-10-18-r1/processing/`; `scripts/render_hydrogeology_figure4.py`; `outputs/current_figure_checks/figure4_evidence/` |
| Figure 5 matched validation | `outputs/monitoring_revision/2026-10-18-r1/calibration/truth.csv`, `selections.csv`, `two_stage/truth.csv`, `selections.csv`, `inference/`; `scripts/render_hydrogeology_figure5.py`; `outputs/current_figure_checks/figure5_evidence/` |
| Supplementary figures (Figs. S1-S11) | `outputs/monitoring_revision/2026-10-18-r1/`, `outputs/validation_revision/2026-10-17-r2/uniform/well_lags_0_12.csv`, `well_multiscale_correlations.csv`; `analysis/ncp_monitoring_figures.py` |

Deposited simulation evidence includes trial-level truth and selection outcomes (`truth.csv`, `selections.csv`, `two_stage_trials.csv`) and complete threshold-grid evaluations (`calibration_search.csv`, `local_rule_grid.csv`). Full raw candidate pools can be regenerated deterministically from fixed seeds using `analysis/ncp_monitoring_revision.py`.
