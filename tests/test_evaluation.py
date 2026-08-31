import numpy as np
import pandas as pd
import pytest

from storm_tracks.evaluation import (
    evaluate_kalman,
    evaluate_persistence,
    forecast_distances,
    relative_rmse_improvement,
    results_metadata,
    rmse,
    summarize_lead_time_errors,
)
from storm_tracks.models import HurricaneKalmanFilter


def metric_track(length: int = 8) -> pd.DataFrame:
    steps = np.arange(length, dtype=float)
    return pd.DataFrame(
        {
            "sid": "A",
            "iso_time": pd.date_range("2020-01-01", periods=length, freq="6h"),
            "x_km": steps * 10,
            "y_km": steps * 5,
            "vx_km": np.full(length, 10.0),
            "vy_km": np.full(length, 5.0),
        }
    )


def test_forecast_distances_uses_euclidean_kilometers() -> None:
    forecasts = np.array([[3.0, 4.0], [6.0, 8.0]])
    observations = np.zeros((2, 2))

    assert forecast_distances(forecasts, observations).tolist() == [5.0, 10.0]


def test_forecast_distances_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        forecast_distances(np.zeros((2, 2)), np.zeros((1, 2)))


def test_summarize_lead_time_errors_calculates_rmse() -> None:
    results = pd.DataFrame(
        {
            "split": ["test", "test", "validation", "validation"],
            "lead_time_hours": [6, 6, 6, 6],
            "error_km": [3.0, 4.0, 30.0, 40.0],
        }
    )

    summary = summarize_lead_time_errors(results, split="test")

    assert summary.loc[6, "mean"] == pytest.approx(3.5)
    assert summary.loc[6, "rmse"] == pytest.approx(3.54)
    assert summary.loc[6, "count"] == 2
    assert set(summary["split"]) == {"test"}


def test_rmse_improvement_uses_root_mean_square_errors() -> None:
    assert rmse(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    assert rmse(np.array([3.0, 4.0])) != pytest.approx(3.5)
    assert relative_rmse_improvement(
        baseline_errors=np.array([6.0, 8.0]),
        candidate_errors=np.array([3.0, 4.0]),
    ) == pytest.approx(50.0)


def test_summary_requires_split_provenance() -> None:
    results = pd.DataFrame({"lead_time_hours": [6], "error_km": [3.0]})

    with pytest.raises(ValueError, match="split"):
        summarize_lead_time_errors(results, split="test")


def test_results_metadata_warns_that_legacy_artifacts_were_not_regenerated() -> None:
    metadata = results_metadata(split="test")

    assert metadata["split"] == "test"
    assert metadata["metric"] == "rmse"
    assert "not regenerated" in metadata["historical_results_warning"].lower()


def test_sliding_origin_evaluators_return_requested_horizons() -> None:
    frame = metric_track(length=10)

    persistence = evaluate_persistence(
        frame,
        storm_ids=["A"],
        lead_times_hours=(6, 12),
        min_origin_index=2,
        max_origins_per_storm=2,
        split="test",
    )
    kalman = evaluate_kalman(
        frame,
        storm_ids=["A"],
        model=HurricaneKalmanFilter(adaptive=False),
        lead_times_hours=(6, 12),
        min_origin_index=2,
        max_origins_per_storm=2,
        use_features=False,
        split="test",
    )

    assert set(persistence["lead_time_hours"]) == {6, 12}
    assert set(kalman["lead_time_hours"]) == {6, 12}
    assert persistence["error_km"].max() == pytest.approx(0.0)
    assert kalman["error_km"].max() == pytest.approx(0.0, abs=1e-8)
    assert set(persistence["split"]) == {"test"}
    assert set(kalman["split"]) == {"test"}
