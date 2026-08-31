from pathlib import Path

import nbformat

EXPECTED_NOTEBOOKS = {
    "00_run_all.ipynb",
    "01_data_preparation.ipynb",
    "02_feature_engineering.ipynb",
    "03_null_model_baseline.ipynb",
    "04_kalman_filter.ipynb",
    "05_evaluation_and_visualization.ipynb",
}
FORBIDDEN_DEFINITIONS = (
    "def load_ibtracs",
    "def estimate_noise_covariances",
    "def persistence_forecast",
    "class HurricaneKalmanFilter",
)


def test_numbered_notebooks_are_thin_and_valid(repository_root: Path) -> None:
    notebook_dir = repository_root / "notebooks"
    paths = sorted(notebook_dir.glob("*.ipynb"))

    assert {path.name for path in paths} == EXPECTED_NOTEBOOKS
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        source = "\n".join(cell.source for cell in code_cells)
        assert len(code_cells) <= 6
        assert not any(token in source for token in FORBIDDEN_DEFINITIONS)
        assert "in globals()" not in source
        compile(source, str(path), "exec")
        assert all(cell.execution_count is None for cell in code_cells)
        assert all(not cell.outputs for cell in code_cells)


def test_orchestration_notebook_delegates_to_shared_runner(
    repository_root: Path,
) -> None:
    path = repository_root / "notebooks" / "00_run_all.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "scripts/run_pipeline.py" in source
    assert "prepare_data" not in source
    assert "run_models" not in source
    assert "build_dashboard_data" not in source
