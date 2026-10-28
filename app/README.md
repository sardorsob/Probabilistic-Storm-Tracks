# Streamlit app

The app is a read-only view of dashboard schema `1.0.0`. Forecasting and aggregation run before Streamlit starts, so the pages display exported results rather than recomputing model output.

## Run locally

Install the app dependencies:

```bash
python -m pip install -e ".[app]"
```

Build `app/data/` from paired model results:

```bash
python scripts/build_dashboard_data.py \
  --persistence-results data/test_persistence_results.pkl \
  --kalman-results data/test_kalman_results.pkl \
  --tracks data/hurricane_paths_processed.pkl \
  --output-dir app/data \
  --headline-split test
```

Start the app from the repository root:

```bash
streamlit run app/app.py
```

Set `STORM_TRACKS_DASHBOARD_DATA_DIR` to load a contract from another directory.

No current full-data contract is tracked in `app/data/`. A fresh clone therefore needs a generated contract, either at that path or at the directory named by the environment variable.

## Pages

- **Overview** shows the headline split, case counts, and RMSE by horizon.
- **Error analysis** compares error distributions and paired storm-level RMSE differences.
- **Storm explorer** shows observed tracks and per-storm error when track tables are present.
- **Methodology** explains the model and metric contract and links to the report.

The app does not load `artifacts/historical_dashboard/`. Those legacy tables use a different schema and do not contain the provenance required by the current contract.

## Data failures

`app/data_loader.py` checks the manifest schema, declared file list, row counts, required columns, and UTC timestamps before rendering a page. If the contract is missing or invalid, the app shows a build instruction instead of partial results.

See [the deployment guide](../keep_alive/README.md) for Streamlit Community Cloud settings.
