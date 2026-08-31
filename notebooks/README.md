# Notebook guide

The notebooks are a readable layer over the Python package. They keep configuration and exploration visible without maintaining a second copy of the model code.

Install the project before opening Jupyter:

```bash
python -m pip install -e ".[notebooks,dev]"
```

## Run in order

`00_run_all.ipynb` previews the complete pipeline and includes an execution guard for an intentional full run. The remaining notebooks are useful when inspecting one stage:

1. `01_data_preparation.ipynb` loads, normalizes, filters, and sorts tracks.
2. `02_feature_engineering.ipynb` selects positions and builds model features.
3. `03_null_model_baseline.ipynb` evaluates the persistence baseline.
4. `04_kalman_filter.ipynb` estimates covariance and evaluates the Kalman filter.
5. `05_evaluation_and_visualization.ipynb` compares split-aware results.

The full IBTrACS preparation and forecast evaluation can take a long time. The notebooks were checked for structure and Python conversion during the refactor, but they were not executed on the complete dataset.

## Historical notebooks

The original output-heavy notebooks are in `artifacts/notebooks/legacy/`. They preserve earlier analysis output but predate the current split and metric rules. Use the numbered notebooks for new work.

## Static verification

These checks do not download data or run notebook cells:

```bash
python -m pytest tests/test_notebook_structure.py -q
jupyter nbconvert --to script notebooks/*.ipynb \
  --output-dir /tmp/storm_tracks_notebook_scripts
```
