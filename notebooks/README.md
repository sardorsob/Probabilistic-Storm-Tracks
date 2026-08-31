# Notebook Guide

> The full IBTrACS preparation and forecast evaluation can take a long time.
> The notebooks added here have not been rerun against the full dataset during
> this refactor, and the tracked historical dashboard results were not
> regenerated with the corrected scientific semantics.

Install the project before opening a notebook:

```bash
python -m pip install -e ".[notebooks,dev]"
```

## Orchestration Mode

[`00_run_all.ipynb`](00_run_all.ipynb) is the reviewer entry point. It delegates
to the canonical stage runner rather than maintaining its own workflow list.
Preview the complete plan safely with:

```bash
python scripts/run_pipeline.py --dry-run
```

Run the expensive workflow only when the raw IBTrACS file is available and a
full regeneration is intended:

```bash
python scripts/run_pipeline.py --execute
```

## Granular Exploration Mode

Use the numbered notebooks in order when inspecting one part of the workflow:

1. `01_data_preparation.ipynb` loads, normalizes, filters, and sorts tracks.
2. `02_feature_engineering.ipynb` selects positions and builds model features.
3. `03_null_model_baseline.ipynb` evaluates the persistence baseline.
4. `04_kalman_filter.ipynb` estimates covariance and evaluates the Kalman model.
5. `05_evaluation_and_visualization.ipynb` compares split-aware results.

The Python package is the source of truth. These notebooks contain narrative,
configuration, package calls, checks, and links—not duplicate implementations.
The original root notebooks remain available for historical reference until the
later cleanup task.

## Verification Status

This refactor validates notebook structure and Python syntax without running
data downloads, model evaluation, or notebook cells:

```bash
python -m pytest tests/test_notebook_structure.py -q
jupyter nbconvert --to script notebooks/*.ipynb \
  --output-dir /tmp/storm_tracks_notebook_scripts
```
