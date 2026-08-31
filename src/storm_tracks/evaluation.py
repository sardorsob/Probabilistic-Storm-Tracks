from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from storm_tracks.models import HurricaneKalmanFilter, kalman_forecast, persistence_forecast


def forecast_distances(forecasts: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Return Euclidean position errors in kilometers."""
    forecast_array = np.asarray(forecasts, dtype=float)
    observation_array = np.asarray(observations, dtype=float)
    if forecast_array.shape != observation_array.shape:
        raise ValueError("forecasts and observations must have matching shapes")
    if forecast_array.ndim != 2 or forecast_array.shape[1] != 2:
        raise ValueError("forecasts and observations must have shape (n, 2)")
    return np.linalg.norm(forecast_array - observation_array, axis=1)


def summarize_lead_time_errors(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast errors by lead time."""
    return (
        results.groupby("lead_time_hours")["error_km"]
        .agg(
            mean="mean",
            median="median",
            rmse=lambda values: np.sqrt(np.mean(values**2)),
            count="count",
            std="std",
        )
        .round(2)
    )


def _evaluate_sliding_origins(
    frame: pd.DataFrame,
    storm_ids: Iterable[str],
    forecaster: Callable[[pd.DataFrame, int, int], dict[str, np.ndarray] | None],
    *,
    lead_times_hours: Sequence[int],
    min_origin_index: int,
    max_origins_per_storm: int | None,
) -> pd.DataFrame:
    rows = []
    max_steps = max(lead_times_hours) // 6
    for storm_id in storm_ids:
        storm = (
            frame.loc[frame["sid"] == storm_id]
            .sort_values("iso_time", kind="stable")
            .reset_index(drop=True)
        )
        if len(storm) < min_origin_index + max_steps + 1:
            continue
        origins = list(range(min_origin_index, len(storm) - max_steps))
        if max_origins_per_storm is not None:
            step = max(1, len(origins) // max_origins_per_storm)
            origins = origins[::step]
        for origin_index in origins:
            result = forecaster(storm, origin_index, max_steps)
            if result is None:
                continue
            for lead_time in lead_times_hours:
                lead_step = lead_time // 6
                if lead_step <= len(result["distances_km"]):
                    rows.append(
                        {
                            "storm_id": storm_id,
                            "origin_idx": origin_index,
                            "lead_time_hours": lead_time,
                            "error_km": result["distances_km"][lead_step - 1],
                        }
                    )
    return pd.DataFrame(rows, columns=["storm_id", "origin_idx", "lead_time_hours", "error_km"])


def evaluate_persistence(
    frame: pd.DataFrame,
    storm_ids: Iterable[str],
    *,
    lead_times_hours: Sequence[int] = (6, 12, 24, 48, 72),
    min_origin_index: int = 5,
    max_origins_per_storm: int | None = 5,
) -> pd.DataFrame:
    """Evaluate persistence forecasts from multiple origins."""
    return _evaluate_sliding_origins(
        frame,
        storm_ids,
        persistence_forecast,
        lead_times_hours=lead_times_hours,
        min_origin_index=min_origin_index,
        max_origins_per_storm=max_origins_per_storm,
    )


def evaluate_kalman(
    frame: pd.DataFrame,
    storm_ids: Iterable[str],
    model: HurricaneKalmanFilter,
    *,
    lead_times_hours: Sequence[int] = (6, 12, 24, 48, 72),
    min_origin_index: int = 5,
    max_origins_per_storm: int | None = 5,
    use_features: bool = True,
) -> pd.DataFrame:
    """Evaluate open-loop Kalman forecasts from multiple origins."""

    def forecast(storm: pd.DataFrame, origin_index: int, steps: int) -> dict[str, np.ndarray]:
        return kalman_forecast(
            storm,
            model,
            origin_index,
            steps,
            use_features=use_features,
        )

    return _evaluate_sliding_origins(
        frame,
        storm_ids,
        forecast,
        lead_times_hours=lead_times_hours,
        min_origin_index=min_origin_index,
        max_origins_per_storm=max_origins_per_storm,
    )
