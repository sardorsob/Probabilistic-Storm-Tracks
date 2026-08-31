import numpy as np
import pandas as pd
import pytest

from storm_tracks.features import (
    add_motion_features,
    classify_motion_regime,
    compute_velocity_from_positions,
    select_agency_tracks,
)


def straight_track() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["A", "A", "A"],
            "iso_time": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 12:00"]
            ),
            "lat": [10.0, 11.0, 12.0],
            "lon": [-50.0, -48.0, -46.0],
            "storm_speed": [20.0, 20.0, 20.0],
            "storm_dir": [45.0, 45.0, 45.0],
            "nature": ["TS", "TS", "TS"],
            "dist2land": [400.0, 250.0, 150.0],
        }
    )


def test_velocity_from_positions_uses_elapsed_time() -> None:
    track = pd.DataFrame(
        {
            "iso_time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00"]),
            "lat": [0.0, 0.0],
            "lon": [0.0, 1.0],
        }
    )

    speed, direction = compute_velocity_from_positions(track)

    assert np.isnan(speed.iloc[0])
    assert speed.iloc[1] == pytest.approx(111.0 / 6.0 / 1.852)
    assert direction.iloc[1] == pytest.approx(90.0)


def test_add_motion_features_builds_model_columns() -> None:
    result = add_motion_features(straight_track())

    assert result["storm_age_hours"].tolist() == [0.0, 6.0, 12.0]
    assert result.loc[1:, "track_curvature"].abs().max() == pytest.approx(0.0)
    assert result.loc[1:, ["vx_km", "vy_km", "v_lat", "v_lon"]].notna().all().all()
    assert result["motion_regime"].tolist() == [0, 0, 0]
    assert result["storm_stage"].tolist() == [2, 2, 2]
    assert result["is_approaching_land"].tolist() == [0, 0, 1]
    assert np.isnan(result.loc[0, "land_gradient"])
    assert result.loc[1, "land_gradient"] == pytest.approx(150.0)


def test_add_motion_features_fills_missing_motion_from_positions() -> None:
    track = straight_track()
    track.loc[1, ["storm_speed", "storm_dir"]] = np.nan

    result = add_motion_features(track)

    assert result.loc[1, "storm_speed"] > 0
    assert result.loc[1, "storm_dir"] == pytest.approx(62.7, abs=0.5)


def test_classify_motion_regime_preserves_notebook_codes() -> None:
    assert classify_motion_regime(storm_speed=4.0, storm_direction=90.0) == 2
    assert classify_motion_regime(storm_speed=10.0, storm_direction=270.0) == 0
    assert classify_motion_regime(storm_speed=10.0, storm_direction=90.0) == 1
    assert np.isnan(classify_motion_regime(storm_speed=np.nan, storm_direction=90.0))


def test_select_agency_tracks_uses_authoritative_positions() -> None:
    frame = pd.DataFrame(
        {
            "sid": ["A", "B"],
            "basin": ["NA", "NA"],
            "usa_agency": ["hurdat_atl", "other"],
            "usa_lat": [12.5, 8.0],
            "usa_lon": [-55.5, -40.0],
            "lat": [12.0, 7.0],
            "lon": [-55.0, -39.0],
        }
    )

    result = select_agency_tracks(frame)

    assert result["use_for_modeling"].tolist() == [True, False]
    assert result["modeling_agency"].tolist()[0] == "hurdat_atl"
    assert result.loc[0, "lat"] == pytest.approx(12.5)
    assert result.loc[0, "lon"] == pytest.approx(-55.5)
