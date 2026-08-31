# Architecture

The repository has one implementation path from raw tracks to the Streamlit app:

```text
IBTrACS CSV
    │
    ▼
scripts/prepare_data.py ──► processed tracks
    │
    ▼
scripts/run_models.py ──► split-aware persistence and Kalman results
    │
    ▼
scripts/build_dashboard_data.py ──► schema-v1 tables and manifest
    │
    ▼
app/data_loader.py ──► Streamlit pages
```

`scripts/run_pipeline.py` owns the stage order. It reads `configs/pipeline.toml`, shows the exact commands in dry-run mode, and executes them only when `--execute` is provided.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `storm_tracks.data` | Load and normalize IBTrACS tracks, unwrap longitude, and split complete storms |
| `storm_tracks.features` | Select agency positions and create motion and track features |
| `storm_tracks.models` | Persistence and Kalman forecasts plus covariance estimation |
| `storm_tracks.evaluation` | Sliding origins, matched errors, metrics, and result metadata |
| `storm_tracks.dashboard_data` | Validate paired results and create app-facing tables |
| `scripts/` | File I/O and command-line entry points around package functions |
| `notebooks/` | Narrative and inspection around the same package functions |
| `app/` | Read the exported contract and render it without recomputing models |

Model or metric behavior belongs in `src/storm_tracks/`. Scripts should translate command-line arguments into package calls. Notebooks should stay short and avoid carrying private copies of the implementation.

## Data ownership

| Location | Meaning | Version control |
| --- | --- | --- |
| `data/` | Raw input, processed tracks, and current model outputs | Ignored |
| `app/data/` | Current generated Streamlit contract | Generated; may be tracked for deployment |
| `artifacts/historical_dashboard/` | Preserved pre-refactor tables with incomplete provenance | Tracked, historical only |
| `artifacts/figures/` | Preserved analysis figures | Tracked |
| `docs/reports/` | Reports and report source | Tracked |
| `docs/references/` | Local reference papers | Tracked |

The app never falls back to the historical tables. It requires a `dashboard_manifest.json` with schema version `1.0.0`, declared files, row counts, split policy, and metric definitions.

## Evaluation flow

The configured seed determines a storm-level split. Covariances are estimated from training storms, while one named validation or test split is evaluated. Each model receives the same storms, forecast origins, and horizons. The dashboard builder rejects duplicate keys, unmatched model cases, invalid errors, unsupported horizons, and storms shared across splits.

## Extending the project

When adding a model, keep its forecast output compatible with the evaluation columns: `split`, `storm_id`, `origin_idx`, `lead_time_hours`, and `error_km`. Add the model to package-level evaluation first, test matched cases and metric semantics, then extend the dashboard contract. Do not make the Streamlit app read model-specific pickles directly.
