# Streamlit deployment

## Prepare the repository

The deployed app needs the schema 1.0.0 files in `app/data/`. Build them from current paired results before deployment:

```bash
python scripts/build_dashboard_data.py \
  --persistence-results data/test_persistence_results.pkl \
  --kalman-results data/test_kalman_results.pkl \
  --tracks data/hurricane_paths_processed.pkl \
  --output-dir app/data \
  --headline-split test
```

Run the app locally and inspect every page:

```bash
streamlit run app/app.py
```

If Streamlit Community Cloud will build directly from GitHub, include the generated `app/data/` contract in the deployment commit. Raw tracks and model pickles remain ignored.

## Community Cloud settings

Create the app at [share.streamlit.io](https://share.streamlit.io) with these repository settings:

- Branch: `main`
- Main file path: `app/app.py`

The repository uses a setuptools `pyproject.toml`. Streamlit Community Cloud currently documents `pyproject.toml` as its Poetry dependency route, so dependency installation for this repository is not confirmed. Before publishing, provide a [Community Cloud-supported dependency file](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) that installs the app dependencies, then verify the build logs.

The app validates the manifest at startup and displays a build instruction if its data is missing or incompatible.

## Optional availability ping

`.github/workflows/keep-streamlit-awake.yml` and `keep_alive/ping.py` target `https://hurricanepaths.streamlit.app/`. Change both locations if the deployment URL changes.

Run the local check from the repository root:

```bash
python keep_alive/ping.py
```

The Python command exits with an error when the response is not HTTP 200. The scheduled workflow makes the same request every 30 minutes, but it only prints the response code; a non-200 response does not currently fail that workflow. It can also be started manually from the GitHub Actions page.
