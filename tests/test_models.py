import numpy as np
import pandas as pd
import pytest

from storm_tracks.models import (
    HurricaneKalmanFilter,
    estimate_noise_covariances,
    kalman_forecast,
    persistence_forecast,
)


def metric_track(sid: str = "A", length: int = 8) -> pd.DataFrame:
    steps = np.arange(length, dtype=float)
    return pd.DataFrame(
        {
            "sid": sid,
            "iso_time": pd.date_range("2020-01-01", periods=length, freq="6h"),
            "x_km": steps * 10,
            "y_km": steps * 5,
            "vx_km": np.full(length, 10.0),
            "vy_km": np.full(length, 5.0),
            "track_curvature": np.zeros(length),
            "is_approaching_land": np.zeros(length),
            "motion_regime": np.ones(length),
            "latitude_regime": np.zeros(length),
            "land_gradient": np.zeros(length),
            "storm_stage": np.full(length, 2),
        }
    )


def test_persistence_forecast_extends_last_displacement() -> None:
    result = persistence_forecast(metric_track(), origin_index=2, lead_steps=2)

    assert result is not None
    np.testing.assert_allclose(result["forecasts"], [[30.0, 15.0], [40.0, 20.0]])
    np.testing.assert_allclose(result["distances_km"], [0.0, 0.0])
    assert result["lead_times_hours"].tolist() == [6, 12]


def test_persistence_forecast_rejects_origin_without_history() -> None:
    assert persistence_forecast(metric_track(), origin_index=0, lead_steps=2) is None


def test_kalman_filter_predicts_and_updates_finite_state() -> None:
    model = HurricaneKalmanFilter(adaptive=False)
    model.initialize(np.array([0.0, 0.0, 10.0, 5.0]))

    predicted_state, predicted_covariance = model.predict()
    updated_state, updated_covariance = model.update(np.array([10.0, 5.0]))

    np.testing.assert_allclose(predicted_state, [10.0, 5.0, 10.0, 5.0])
    assert np.isfinite(predicted_covariance).all()
    assert np.isfinite(updated_state).all()
    assert np.isfinite(updated_covariance).all()


def test_adaptive_filter_preserves_notebook_regime_two_scaling() -> None:
    model = HurricaneKalmanFilter(adaptive=True)

    model.adapt_parameters({"motion_regime": 2})

    np.testing.assert_allclose(model.Q, model.Q_base * 1.3)


def test_adaptive_filter_leaves_noise_unchanged_without_features() -> None:
    model = HurricaneKalmanFilter(adaptive=True)

    model.adapt_parameters({})

    np.testing.assert_allclose(model.Q, model.Q_base)


def test_kalman_forecast_follows_straight_track() -> None:
    result = kalman_forecast(
        metric_track(),
        HurricaneKalmanFilter(adaptive=False),
        origin_index=2,
        lead_steps=2,
        use_features=False,
    )

    np.testing.assert_allclose(result["forecasts"], [[30.0, 15.0], [40.0, 20.0]])
    np.testing.assert_allclose(result["distances_km"], [0.0, 0.0], atol=1e-8)


def test_estimated_covariances_are_symmetric_and_finite() -> None:
    training_tracks = pd.concat([metric_track("A"), metric_track("B")], ignore_index=True)

    process_covariance, observation_covariance = estimate_noise_covariances(training_tracks)

    assert process_covariance is not None
    assert observation_covariance is not None
    np.testing.assert_allclose(process_covariance, process_covariance.T)
    np.testing.assert_allclose(observation_covariance, observation_covariance.T)
    assert np.isfinite(process_covariance).all()
    assert np.isfinite(observation_covariance).all()
    assert np.diag(observation_covariance).tolist() == pytest.approx([0.25, 0.25])
