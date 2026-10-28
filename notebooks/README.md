# Notebook guide

The six numbered notebooks are the supported Jupyter workflow. They are a readable layer over the Python package and do not maintain a second copy of the model code.

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

The full IBTrACS preparation and forecast evaluation can take a long time. Repository tests validate notebook structure, compile their code cells, and ensure the model implementation remains in the package. They do not execute cells or establish a full-data result.

## Static verification

These checks do not download data or run notebook cells:

```bash
python -m pytest tests/test_notebook_structure.py -q
jupyter nbconvert --to script notebooks/*.ipynb \
  --output-dir /tmp/storm_tracks_notebook_scripts
```
