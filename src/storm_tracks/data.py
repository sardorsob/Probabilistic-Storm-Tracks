from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = {
    "storm_dir": "Int64",
    "storm_speed": "Int64",
    "usa_wind": "Int64",
    "usa_lat": "Float64",
    "usa_lon": "Float64",
}


@dataclass(frozen=True)
class StormSplit:
    """Storm-level train, validation, and test frames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_ibtracs(path: str | Path) -> pd.DataFrame:
    """Load the two-header IBTrACS CSV and normalize core columns."""
    frame = pd.read_csv(Path(path), skiprows=[1], low_memory=False)
    frame.columns = frame.columns.str.lower()
    frame = frame.replace(" ", np.nan)

    for column, dtype in NUMERIC_COLUMNS.items():
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(dtype)
    if "iso_time" in frame:
        frame["iso_time"] = pd.to_datetime(frame["iso_time"], utc=True)
    return frame


def unwrap_longitudes(longitudes: pd.Series) -> pd.Series:
    """Return continuous longitudes while preserving missing-value gaps."""
    values = pd.to_numeric(longitudes, errors="coerce").to_numpy(dtype=float)
    result = values.copy()
    valid_indexes = np.flatnonzero(np.isfinite(values))
    if valid_indexes.size:
        breaks = np.flatnonzero(np.diff(valid_indexes) != 1) + 1
        for indexes in np.split(valid_indexes, breaks):
            result[indexes] = np.degrees(np.unwrap(np.radians(values[indexes])))
    return pd.Series(result, index=longitudes.index, name=longitudes.name)


def prepare_tracks(frame: pd.DataFrame, min_observations: int = 2) -> pd.DataFrame:
    """Filter short storms and sort observations into trajectory order."""
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    required = {"sid", "iso_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    counts = frame.groupby("sid")["sid"].transform("size")
    result = (
        frame.loc[counts >= min_observations]
        .sort_values(["sid", "iso_time"], kind="stable")
        .reset_index(drop=True)
    )
    if "lon" in result:
        result["lon_unwrapped"] = result.groupby("sid", sort=False)["lon"].transform(
            unwrap_longitudes
        )
    return result


def split_by_storm(
    frame: pd.DataFrame,
    *,
    id_column: str = "sid",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.0,
    seed: int = 42,
) -> StormSplit:
    """Split complete storms into deterministic train, validation, and test sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")
    if id_column not in frame:
        raise ValueError(f"missing storm identifier column: {id_column}")

    storm_ids = frame[id_column].dropna().unique()
    if len(storm_ids) == 0:
        raise ValueError("cannot split a frame without storm identifiers")
    shuffled = np.random.default_rng(seed).permutation(storm_ids)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    train_ids = set(shuffled[:train_end])
    validation_ids = set(shuffled[train_end:validation_end])
    test_ids = set(shuffled[validation_end:])

    def select(ids: set[object]) -> pd.DataFrame:
        return frame.loc[frame[id_column].isin(ids)].reset_index(drop=True)

    return StormSplit(
        train=select(train_ids),
        validation=select(validation_ids),
        test=select(test_ids),
    )


def validate_disjoint_storm_splits(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    id_column: str = "sid",
) -> None:
    """Raise when a storm identifier appears in more than one data split."""
    splits = {"train": train, "test": test}
    if validation is not None:
        splits["validation"] = validation
    missing = [name for name, frame in splits.items() if id_column not in frame]
    if missing:
        raise ValueError(f"missing {id_column!r} in splits: {missing}")

    identifiers = {
        name: set(frame[id_column].dropna().unique()) for name, frame in splits.items()
    }
    names = list(identifiers)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = identifiers[left_name] & identifiers[right_name]
            if overlap:
                raise ValueError(
                    f"storm identifiers overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)}"
                )
