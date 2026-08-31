from pathlib import Path

import pandas as pd
import pytest

from storm_tracks.data import load_ibtracs, prepare_tracks, split_by_storm


def test_load_ibtracs_normalizes_columns_and_types(tmp_path: Path) -> None:
    csv_path = tmp_path / "ibtracs.csv"
    csv_path.write_text(
        "SID,ISO_TIME,LAT,LON,STORM_DIR,STORM_SPEED,USA_WIND,USA_LAT,USA_LON\n"
        " , ,degrees,degrees,degrees,knots,knots,degrees,degrees\n"
        "A,2020-01-01 00:00:00,10,-50,90,12,40,10.1,-50.1\n"
        "A,2020-01-01 06:00:00,11,-49, ,13,41,11.1,-49.1\n",
        encoding="utf-8",
    )

    frame = load_ibtracs(csv_path)

    assert list(frame.columns) == [
        "sid",
        "iso_time",
        "lat",
        "lon",
        "storm_dir",
        "storm_speed",
        "usa_wind",
        "usa_lat",
        "usa_lon",
    ]
    assert str(frame["storm_dir"].dtype) == "Int64"
    assert str(frame["usa_lat"].dtype) == "Float64"
    assert pd.api.types.is_datetime64_any_dtype(frame["iso_time"])
    assert pd.isna(frame.loc[1, "storm_dir"])


def test_prepare_tracks_filters_short_storms_and_sorts_observations() -> None:
    frame = pd.DataFrame(
        {
            "sid": ["A", "B", "A"],
            "iso_time": pd.to_datetime(
                ["2020-01-01 06:00", "2020-01-01 00:00", "2020-01-01 00:00"]
            ),
            "lat": [11.0, 5.0, 10.0],
            "lon": [-49.0, 20.0, -50.0],
        }
    )

    result = prepare_tracks(frame, min_observations=2)

    assert result["sid"].tolist() == ["A", "A"]
    assert result["lat"].tolist() == [10.0, 11.0]
    assert result.index.tolist() == [0, 1]


def test_split_by_storm_keeps_storms_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "sid": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "value": range(8),
        }
    )

    split = split_by_storm(
        frame,
        id_column="sid",
        train_fraction=0.5,
        validation_fraction=0.25,
        seed=7,
    )

    train_ids = set(split.train["sid"])
    validation_ids = set(split.validation["sid"])
    test_ids = set(split.test["sid"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == {"A", "B", "C", "D"}


def test_split_by_storm_rejects_invalid_fractions() -> None:
    frame = pd.DataFrame({"sid": ["A", "B"]})

    with pytest.raises(ValueError, match="sum to less than 1"):
        split_by_storm(
            frame,
            id_column="sid",
            train_fraction=0.8,
            validation_fraction=0.2,
        )
