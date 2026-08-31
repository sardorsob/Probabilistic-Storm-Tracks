# Regenerate dashboard data

The Streamlit app reads a generated, versioned contract. It does not read model pickles or historical CSV files directly.

## Build the complete workflow

Place the raw IBTrACS CSV at the path configured in `configs/pipeline.toml`. Preview the work first:

```bash
python scripts/run_pipeline.py --dry-run
```

The full command prepares all tracks, evaluates both models, and builds the app data. It can take a long time:

```bash
python scripts/run_pipeline.py --execute
```

## Rebuild from existing model results

If the processed tracks and paired model outputs already exist, run only the export step:

```bash
python scripts/build_dashboard_data.py \
  --persistence-results data/test_persistence_results.pkl \
  --kalman-results data/test_kalman_results.pkl \
  --tracks data/hurricane_paths_processed.pkl \
  --output-dir app/data \
  --headline-split test
```

`--tracks` is optional. Without it, the overview and error pages work, but the storm map is unavailable.

The builder validates finite non-negative errors, supported horizons, unique forecast keys, matched model cases, and one split per storm. It writes the manifest last so an interrupted export cannot look complete.

## Output contract

Required files:

- `dashboard_manifest.json`
- `model_comparison_summary.csv`
- `error_distributions.csv`
- `storm_performance.csv`
- `model_comparison_detailed.csv`

Optional track files:

- `storm_metadata.csv`
- `storm_tracks.csv.gz`

The manifest records schema version, source artifact names, generation time, split policy, metric definitions, file names, and row counts. The app rejects missing, stale, or malformed contracts with a readable message.

## Validate and open the app

```bash
python -m pytest tests/test_dashboard_data.py tests/test_app.py -q
streamlit run app/app.py
```

To load a contract outside `app/data/`:

```bash
STORM_TRACKS_DASHBOARD_DATA_DIR=/absolute/path/to/dashboard-data \
  streamlit run app/app.py
```

Do not copy the files in `artifacts/historical_dashboard/` into `app/data/`. They predate schema version 1 and do not carry the split provenance required by the current app.
