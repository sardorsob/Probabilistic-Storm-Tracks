from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ALLOWED_LEAD_TIMES = {6, 12, 24, 48, 72}


def load_table(repository_root: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(repository_root / "streamboard" / "data" / filename)


def test_legacy_dashboard_contract_matches_shipped_files(
    repository_root: Path,
    legacy_contract: dict[str, dict[str, object]],
) -> None:
    for filename, expected in legacy_contract.items():
        frame = load_table(repository_root, filename)
        assert list(frame.columns) == expected["columns"]
        assert frame.dtypes.astype(str).to_dict() == expected["dtypes"]
        assert len(frame) == expected["rows"]


def test_legacy_forecasts_use_supported_lead_times(repository_root: Path) -> None:
    filenames = (
        "error_distributions.csv",
        "model_comparison_detailed.csv",
        "model_comparison_summary.csv",
        "storm_performance.csv",
    )
    for filename in filenames:
        frame = load_table(repository_root, filename)
        assert set(frame["lead_time_hours"].unique()) <= ALLOWED_LEAD_TIMES


def test_legacy_error_columns_are_finite_and_nonnegative(repository_root: Path) -> None:
    columns_by_file = {
        "error_distributions.csv": ("error_km", "error_squared"),
        "model_comparison_detailed.csv": ("kf_error_km", "null_error_km"),
        "model_comparison_summary.csv": (
            "kf_mean_error",
            "kf_median_error",
            "kf_rmse",
            "kf_std_error",
            "null_mean_error",
            "null_median_error",
            "null_rmse",
            "null_std_error",
        ),
        "storm_performance.csv": (
            "kf_mean_error",
            "kf_std_error",
            "null_mean_error",
            "null_std_error",
        ),
    }
    for filename, columns in columns_by_file.items():
        values = load_table(repository_root, filename).loc[:, columns].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0).all()


@pytest.mark.xfail(
    strict=True,
    reason="legacy comparison rows do not preserve train/validation/test split",
)
def test_legacy_comparisons_preserve_split(repository_root: Path) -> None:
    frame = load_table(repository_root, "model_comparison_detailed.csv")
    assert "split" in frame.columns
