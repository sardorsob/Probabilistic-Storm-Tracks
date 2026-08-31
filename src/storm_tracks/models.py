from collections.abc import Mapping

import numpy as np
import pandas as pd

STATE_TRANSITION = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
OBSERVATION_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
FEATURE_COLUMNS = (
    "track_curvature",
    "is_approaching_land",
    "motion_regime",
    "latitude_regime",
    "land_gradient",
    "storm_stage",
)


def _scalar(features: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    if hasattr(value, "iloc"):
        return float(value.iloc[0]) if len(value) else default
    if isinstance(value, list | np.ndarray):
        return float(value[0]) if len(value) else default
    return float(value)


class HurricaneKalmanFilter:
    """Constant-velocity Kalman filter in kilometers per six-hour step."""

    def __init__(
        self,
        q_base: np.ndarray | None = None,
        r_base: np.ndarray | None = None,
        *,
        adaptive: bool = True,
    ) -> None:
        self.A = STATE_TRANSITION.copy()
        self.H = OBSERVATION_MATRIX.copy()
        self.Q_base = (
            np.diag([1.0, 1.0, 10.0, 10.0]) if q_base is None else np.array(q_base, copy=True)
        )
        self.R_base = np.eye(2) * 0.25 if r_base is None else np.array(r_base, copy=True)
        self.adaptive = adaptive
        self.Q = self.Q_base.copy()
        self.R = self.R_base.copy()
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.x_pred: np.ndarray | None = None
        self.P_pred: np.ndarray | None = None

    def adapt_parameters(self, features: Mapping[str, object]) -> None:
        """Scale process noise using the notebook's feature rules."""
        if not self.adaptive:
            return
        scale = 1.0
        if "track_curvature" in features:
            curvature = _scalar(features, "track_curvature", np.nan)
            if not np.isnan(curvature):
                scale *= 1 + min(abs(curvature) * 0.5, 2.0)
        if "is_approaching_land" in features and _scalar(features, "is_approaching_land") == 1:
            scale *= 1.5
        if "motion_regime" in features:
            motion_regime = _scalar(features, "motion_regime")
            if motion_regime == 2:
                scale *= 1.3
            elif motion_regime == 0:
                scale *= 1.1
        if "latitude_regime" in features and _scalar(features, "latitude_regime") == 2:
            scale *= 1.2
        self.Q = self.Q_base * scale
        self.R = self.R_base.copy()

    def initialize(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        """Initialize the filter state and covariance."""
        self.x = np.array(initial_state, dtype=float, copy=True)
        self.P = (
            np.eye(4) * 0.1
            if initial_covariance is None
            else np.array(initial_covariance, dtype=float, copy=True)
        )

    def predict(self, process_noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Advance the filter by one six-hour step."""
        if self.x is None or self.P is None:
            raise RuntimeError("initialize the filter before prediction")
        noise = self.Q if process_noise is None else process_noise
        self.x_pred = self.A @ self.x
        self.P_pred = self.A @ self.P @ self.A.T + noise
        return self.x_pred.copy(), self.P_pred.copy()

    def update(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update the predicted state with an observed position."""
        if self.x_pred is None or self.P_pred is None:
            raise RuntimeError("predict before updating the filter")
        innovation = np.asarray(observation) - self.H @ self.x_pred
        innovation_covariance = self.H @ self.P_pred @ self.H.T + self.R
        kalman_gain = self.P_pred @ self.H.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x_pred + kalman_gain @ innovation
        self.P = (np.eye(4) - kalman_gain @ self.H) @ self.P_pred
        return self.x.copy(), self.P.copy()

    def forecast(self, steps: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return open-loop state forecasts without mutating the filter."""
        if self.x is None or self.P is None:
            raise RuntimeError("initialize the filter before forecasting")
        state = self.x.copy()
        covariance = self.P.copy()
        forecasts = []
        for _ in range(steps):
            state = self.A @ state
            covariance = self.A @ covariance @ self.A.T + self.Q
            forecasts.append((state.copy(), covariance.copy()))
        return forecasts


def _forecast_result(
    forecasts: list[np.ndarray],
    observations: list[np.ndarray],
) -> dict[str, np.ndarray]:
    forecast_array = np.asarray(forecasts, dtype=float).reshape(-1, 2)
    observation_array = np.asarray(observations, dtype=float).reshape(-1, 2)
    errors = forecast_array - observation_array
    return {
        "forecasts": forecast_array,
        "true_observations": observation_array,
        "errors": errors,
        "distances_km": np.linalg.norm(errors, axis=1),
        "lead_times_hours": np.arange(1, len(forecast_array) + 1) * 6,
    }


def persistence_forecast(
    storm: pd.DataFrame,
    origin_index: int,
    lead_steps: int,
) -> dict[str, np.ndarray] | None:
    """Forecast by persisting the most recent observed displacement."""
    ordered = storm.sort_values("iso_time", kind="stable").reset_index(drop=True)
    if origin_index < 1 or origin_index >= len(ordered):
        return None
    velocity = ordered.loc[origin_index, ["x_km", "y_km"]].to_numpy(dtype=float) - ordered.loc[
        origin_index - 1, ["x_km", "y_km"]
    ].to_numpy(dtype=float)
    position = ordered.loc[origin_index, ["x_km", "y_km"]].to_numpy(dtype=float)
    forecasts = []
    observations = []
    for step in range(1, lead_steps + 1):
        target = origin_index + step
        if target >= len(ordered):
            break
        position = position + velocity
        forecasts.append(position.copy())
        observations.append(ordered.loc[target, ["x_km", "y_km"]].to_numpy(dtype=float))
    return _forecast_result(forecasts, observations)


def _initialize_from_observations(
    storm: pd.DataFrame,
    model: HurricaneKalmanFilter,
    origin_index: int,
    *,
    use_features: bool,
) -> None:
    initial = np.array(
        [
            storm.loc[0, "x_km"],
            storm.loc[0, "y_km"],
            0.0 if pd.isna(storm.loc[0, "vx_km"]) else storm.loc[0, "vx_km"],
            0.0 if pd.isna(storm.loc[0, "vy_km"]) else storm.loc[0, "vy_km"],
        ]
    )
    model.initialize(initial)
    has_features = use_features and set(FEATURE_COLUMNS).issubset(storm.columns)
    for index in range(1, min(origin_index + 1, len(storm))):
        if has_features:
            model.adapt_parameters({column: storm.loc[index, column] for column in FEATURE_COLUMNS})
        model.predict()
        model.update(storm.loc[index, ["x_km", "y_km"]].to_numpy(dtype=float))


def kalman_forecast(
    storm: pd.DataFrame,
    model: HurricaneKalmanFilter,
    origin_index: int,
    lead_steps: int,
    *,
    use_features: bool = True,
) -> dict[str, np.ndarray]:
    """Run an open-loop Kalman forecast from one track origin."""
    ordered = storm.sort_values("iso_time", kind="stable").reset_index(drop=True)
    if not 0 <= origin_index < len(ordered):
        raise ValueError("origin_index is outside the storm track")
    forecast_model = HurricaneKalmanFilter(
        q_base=model.Q_base,
        r_base=model.R_base,
        adaptive=model.adaptive,
    )
    _initialize_from_observations(
        ordered,
        forecast_model,
        origin_index,
        use_features=use_features,
    )
    assert forecast_model.x is not None
    assert forecast_model.P is not None
    state = forecast_model.x.copy()
    covariance = forecast_model.P.copy()
    forecasts = []
    observations = []
    has_features = use_features and set(FEATURE_COLUMNS).issubset(ordered.columns)
    for step in range(1, lead_steps + 1):
        target = origin_index + step
        if target >= len(ordered):
            break
        if has_features:
            feature_index = min(target - 1, len(ordered) - 1)
            forecast_model.adapt_parameters(
                {column: ordered.loc[feature_index, column] for column in FEATURE_COLUMNS}
            )
        state = forecast_model.A @ state
        covariance = forecast_model.A @ covariance @ forecast_model.A.T + forecast_model.Q
        forecasts.append(state[:2].copy())
        observations.append(ordered.loc[target, ["x_km", "y_km"]].to_numpy(dtype=float))
    return _forecast_result(forecasts, observations)


def estimate_noise_covariances(
    training_tracks: pd.DataFrame,
    *,
    min_storm_length: int = 5,
    sample_size: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate notebook-compatible process and observation covariance matrices."""
    counts = training_tracks.groupby("sid")["sid"].transform("size")
    storm_ids = training_tracks.loc[counts >= min_storm_length, "sid"].unique()
    if sample_size is not None:
        storm_ids = storm_ids[:sample_size]
    state_changes = []
    for storm_id in storm_ids:
        storm = (
            training_tracks.loc[training_tracks["sid"] == storm_id]
            .sort_values("iso_time", kind="stable")
            .reset_index(drop=True)
        )
        for index in range(1, len(storm)):
            columns = ["x_km", "y_km", "vx_km", "vy_km"]
            previous = storm.loc[index - 1, columns].to_numpy(dtype=float, na_value=0.0)
            actual = storm.loc[index, columns].to_numpy(dtype=float, na_value=0.0)
            state_changes.append(actual - STATE_TRANSITION @ previous)
    if not state_changes:
        return None, None
    process_covariance = np.cov(np.asarray(state_changes).T) + np.eye(4) * 1e-6
    return process_covariance, np.eye(2) * 0.25
