from dataclasses import dataclass

import numpy as np
import pandas as pd

from storm_tracks.evaluation import rmse

ALLOWED_FORECAST_HORIZONS = frozenset({6, 12, 24, 48, 72})
FORECAST_KEY_COLUMNS = [
    "storm_id",
    "origin_idx",
    "forecast_horizon_hours",
    "split",
]


@dataclass(frozen=True)
class DashboardData:
    """Tables consumed by the Streamlit dashboard."""

    model_comparison_summary: pd.DataFrame
    error_distributions: pd.DataFrame
    storm_performance: pd.DataFrame
    model_comparison_detailed: pd.DataFrame
    storm_metadata: pd.DataFrame | None
    storm_tracks: pd.DataFrame | None


def _normalize_results(results: pd.DataFrame, model: str) -> pd.DataFrame:
    required = {"storm_id", "origin_idx", "lead_time_hours", "error_km", "split"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"missing required {model} result columns: {sorted(missing)}")
    normalized = results.rename(
        columns={"lead_time_hours": "forecast_horizon_hours"}
    ).copy()
    normalized["model"] = model
    normalized = normalized[
        [
            "storm_id",
            "origin_idx",
            "forecast_horizon_hours",
            "error_km",
            "split",
            "model",
        ]
    ]
    normalized["error_km"] = pd.to_numeric(normalized["error_km"], errors="coerce")
    valid_errors = np.isfinite(normalized["error_km"]) & (normalized["error_km"] >= 0)
    if not valid_errors.all():
        raise ValueError(f"{model} errors must be finite and non-negative")
    horizons = pd.to_numeric(normalized["forecast_horizon_hours"], errors="coerce")
    invalid_horizons = normalized.loc[
        ~horizons.isin(ALLOWED_FORECAST_HORIZONS), "forecast_horizon_hours"
    ].unique()
    if len(invalid_horizons):
        raise ValueError(
            f"unsupported forecast horizons in {model} results: "
            f"{sorted(map(str, invalid_horizons))}"
        )
    normalized["forecast_horizon_hours"] = horizons.astype(int)
    if normalized[["storm_id", "origin_idx", "split"]].isna().any().any():
        raise ValueError(f"{model} forecast keys cannot contain missing values")
    if normalized.duplicated(FORECAST_KEY_COLUMNS).any():
        raise ValueError(f"duplicate {model} forecast keys")
    return normalized


def _validate_paired_results(
    persistence: pd.DataFrame,
    kalman: pd.DataFrame,
) -> None:
    persistence_keys = pd.MultiIndex.from_frame(persistence[FORECAST_KEY_COLUMNS])
    kalman_keys = pd.MultiIndex.from_frame(kalman[FORECAST_KEY_COLUMNS])
    if len(persistence_keys.difference(kalman_keys)) or len(
        kalman_keys.difference(persistence_keys)
    ):
        raise ValueError("persistence and kalman results must have matching forecast keys")


def _validate_split_provenance(errors: pd.DataFrame) -> None:
    split_counts = errors.groupby("storm_id", observed=True)["split"].nunique()
    shared = sorted(map(str, split_counts.index[split_counts > 1]))
    if shared:
        raise ValueError(f"storms appear in more than one split: {shared}")


def _track_tables(
    tracks: pd.DataFrame,
    storm_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sid", "iso_time", "lat", "lon"}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"missing required track columns: {sorted(missing)}")
    source = tracks.loc[:, list(required)].copy()
    source["sid"] = source["sid"].astype("string").str.strip()
    expected_storms = set(storm_splits["storm_id"])
    missing_storms = sorted(expected_storms.difference(source["sid"].dropna()))
    if missing_storms:
        raise ValueError(f"missing track points for evaluated storms: {missing_storms}")
    points = source.loc[source["sid"].isin(expected_storms)].copy()
    points = points.rename(
        columns={
            "sid": "storm_id",
            "iso_time": "time_utc",
            "lat": "latitude",
            "lon": "longitude",
        }
    )
    points["time_utc"] = pd.to_datetime(points["time_utc"], errors="coerce", utc=True)
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    invalid_coordinates = (
        ~np.isfinite(points["latitude"])
        | ~np.isfinite(points["longitude"])
        | ~points["latitude"].between(-90, 90)
        | ~points["longitude"].between(-180, 180)
    )
    if points["time_utc"].isna().any() or invalid_coordinates.any():
        raise ValueError("track times and coordinates must be valid")
    if points.duplicated(["storm_id", "time_utc"]).any():
        raise ValueError("track data contains duplicate storm timestamps")
    points = points.merge(storm_splits, on="storm_id", validate="many_to_one")
    points = points[
        ["storm_id", "split", "time_utc", "latitude", "longitude"]
    ].sort_values(["storm_id", "time_utc"], kind="stable", ignore_index=True)
    metadata = (
        points.groupby(["storm_id", "split"], sort=True, observed=True)["time_utc"]
        .agg(
            start_time_utc="min",
            end_time_utc="max",
            observation_count="size",
        )
        .reset_index()
    )
    return metadata, points


def _storm_performance(errors: pd.DataFrame) -> pd.DataFrame:
    return (
        errors.groupby(
            ["storm_id", "model", "split", "forecast_horizon_hours"],
            sort=True,
            observed=True,
        )["error_km"]
        .agg(
            forecast_count="count",
            mean_error_km="mean",
            rmse_km=rmse,
            median_error_km="median",
        )
        .reset_index()
    )


def _detailed_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    keys = ["storm_id", "split", "forecast_horizon_hours"]
    metric_columns = ["forecast_count", "rmse_km"]
    persistence = performance.loc[performance["model"] == "persistence", keys + metric_columns]
    kalman = performance.loc[performance["model"] == "kalman", keys + metric_columns]
    comparison = persistence.merge(
        kalman,
        on=keys,
        validate="one_to_one",
        suffixes=("_persistence", "_kalman"),
    )
    comparison = comparison.rename(
        columns={
            "forecast_count_persistence": "forecast_count",
            "rmse_km_persistence": "persistence_rmse_km",
            "rmse_km_kalman": "kalman_rmse_km",
        }
    ).drop(columns="forecast_count_kalman")
    comparison["rmse_improvement_km"] = (
        comparison["persistence_rmse_km"] - comparison["kalman_rmse_km"]
    )
    comparison["rmse_improvement_pct"] = np.where(
        comparison["persistence_rmse_km"] > 0,
        comparison["rmse_improvement_km"]
        / comparison["persistence_rmse_km"]
        * 100,
        np.nan,
    )
    comparison["better_model"] = np.select(
        [
            comparison["rmse_improvement_km"] > 0,
            comparison["rmse_improvement_km"] < 0,
        ],
        ["kalman", "persistence"],
        default="tie",
    )
    return comparison


def build_dashboard_data(
    persistence_results: pd.DataFrame,
    kalman_results: pd.DataFrame,
    *,
    tracks: pd.DataFrame | None = None,
    headline_split: str = "test",
) -> DashboardData:
    """Build the dashboard tables from paired model results."""
    persistence = _normalize_results(persistence_results, "persistence")
    kalman = _normalize_results(kalman_results, "kalman")
    _validate_paired_results(persistence, kalman)
    errors = pd.concat(
        [persistence, kalman],
        ignore_index=True,
    )
    _validate_split_provenance(errors)
    selected = errors.loc[errors["split"] == headline_split]
    if selected.empty:
        raise ValueError(f"no forecast results found for split {headline_split!r}")
    summary = (
        selected.groupby(
            ["model", "split", "forecast_horizon_hours"],
            sort=True,
            observed=True,
        )["error_km"]
        .agg(
            forecast_count="count",
            mean_error_km="mean",
            rmse_km=rmse,
            median_error_km="median",
            error_p25_km=lambda values: values.quantile(0.25),
            error_p75_km=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    errors = errors.sort_values(
        ["split", "storm_id", "origin_idx", "forecast_horizon_hours", "model"],
        kind="stable",
    ).reset_index(drop=True)
    performance = _storm_performance(errors)
    metadata = None
    track_points = None
    if tracks is not None:
        storm_splits = errors[["storm_id", "split"]].drop_duplicates(ignore_index=True)
        metadata, track_points = _track_tables(tracks, storm_splits)
    return DashboardData(
        model_comparison_summary=summary,
        error_distributions=errors,
        storm_performance=performance,
        model_comparison_detailed=_detailed_comparison(performance),
        storm_metadata=metadata,
        storm_tracks=track_points,
    )
