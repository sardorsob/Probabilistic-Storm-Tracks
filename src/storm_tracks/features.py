import numpy as np
import pandas as pd

KM_PER_DEGREE = 111.0
KNOT_TO_KMH = 1.852
PREFERRED_AGENCIES = {
    "EP": ("hurdat_epa",),
    "NA": ("hurdat_atl",),
    "WP": ("jtwc_wp", "jtwc_cp", "jtwc_ep"),
    "NI": ("jtwc_io",),
    "SI": ("jtwc_sh", "jtwc_io"),
    "SP": ("jtwc_sh",),
    "SA": ("jtwc_sh",),
}
STORM_STAGE_CODES = {"DS": 0, "NR": 1, "SS": 1, "TS": 2, "MX": 3, "ET": 4}


def compute_velocity_from_positions(
    storm: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Estimate speed in knots and direction in degrees from track positions."""
    ordered = storm.sort_values("iso_time", kind="stable")
    elapsed_hours = ordered["iso_time"].diff().dt.total_seconds() / 3600
    latitude_km = ordered["lat"].diff() * KM_PER_DEGREE
    longitude_delta = ordered["lon"].diff()
    longitude_delta = longitude_delta.where(longitude_delta <= 180, longitude_delta - 360)
    longitude_delta = longitude_delta.where(longitude_delta >= -180, longitude_delta + 360)
    longitude_km = longitude_delta * KM_PER_DEGREE * np.cos(np.radians(ordered["lat"]))
    speed = np.hypot(latitude_km, longitude_km) / elapsed_hours.replace(0, np.nan)
    direction = np.degrees(np.arctan2(longitude_km, latitude_km)) % 360
    return speed / KNOT_TO_KMH, direction


def classify_motion_regime(storm_speed: float, storm_direction: float) -> float | int:
    """Return the notebook's numeric motion-regime code."""
    if pd.isna(storm_speed) or pd.isna(storm_direction):
        return np.nan
    if storm_speed < 5:
        return 2
    if storm_direction <= 45 or storm_direction >= 225:
        return 0
    return 1


def select_agency_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark basin-preferred agency rows and use their reported positions."""
    result = frame.copy()
    result["use_for_modeling"] = False
    result["modeling_agency"] = pd.NA
    if "usa_agency" not in result:
        result["use_for_modeling"] = True
        result["modeling_agency"] = "all"
        return result

    for basin, agencies in PREFERRED_AGENCIES.items():
        selected = (result["basin"] == basin) & result["usa_agency"].isin(agencies)
        result.loc[selected, "use_for_modeling"] = True
        result.loc[selected, "modeling_agency"] = result.loc[selected, "usa_agency"]

    position_columns = {"usa_lat", "usa_lon", "lat", "lon"}
    if position_columns.issubset(result.columns):
        selected = (
            result["use_for_modeling"]
            & result["usa_lat"].notna()
            & result["usa_lon"].notna()
        )
        result.loc[selected, "lat"] = result.loc[selected, "usa_lat"].astype(float)
        result.loc[selected, "lon"] = result.loc[selected, "usa_lon"].astype(float)
    return result


def add_motion_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the state and motion features used by the forecast notebooks."""
    required = {"sid", "iso_time", "lat", "lon", "storm_speed", "storm_dir"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    result = frame.sort_values(["sid", "iso_time"], kind="stable").reset_index(drop=True)
    result["storm_speed"] = pd.to_numeric(result["storm_speed"], errors="coerce")
    result["storm_dir"] = pd.to_numeric(result["storm_dir"], errors="coerce")
    grouped = result.groupby("sid", sort=False)

    for indexes in grouped.groups.values():
        speed, direction = compute_velocity_from_positions(result.loc[indexes])
        result.loc[indexes, "storm_speed"] = result.loc[indexes, "storm_speed"].fillna(speed)
        result.loc[indexes, "storm_dir"] = result.loc[indexes, "storm_dir"].fillna(direction)

    latitude_reference = grouped["lat"].transform("first")
    longitude_reference = grouped["lon"].transform("first")
    result["lat_ref"] = latitude_reference
    result["lon_ref"] = longitude_reference
    result["x_km"] = (
        KM_PER_DEGREE
        * np.cos(np.radians(latitude_reference))
        * (result["lon"] - longitude_reference)
    )
    result["y_km"] = KM_PER_DEGREE * (result["lat"] - latitude_reference)

    elapsed_hours = grouped["iso_time"].diff().dt.total_seconds() / 3600
    result["vx_km"] = grouped["x_km"].diff() / elapsed_hours.replace(0, np.nan) * 6
    result["vy_km"] = grouped["y_km"].diff() / elapsed_hours.replace(0, np.nan) * 6
    first_observation = grouped.cumcount() == 0
    speed_per_period = result["storm_speed"] * KNOT_TO_KMH * 6
    direction_radians = np.radians(result["storm_dir"])
    result.loc[first_observation, "vx_km"] = (
        speed_per_period * np.sin(direction_radians)
    )[first_observation]
    result.loc[first_observation, "vy_km"] = (
        speed_per_period * np.cos(direction_radians)
    )[first_observation]

    result["v_lat"] = (
        speed_per_period / KM_PER_DEGREE * np.cos(direction_radians)
    )
    result["v_lon"] = (
        speed_per_period
        / (KM_PER_DEGREE * np.cos(np.radians(result["lat"])))
        * np.sin(direction_radians)
    )
    grouped = result.groupby("sid", sort=False)
    result["a_lat"] = grouped["v_lat"].diff() / elapsed_hours.replace(0, np.nan)
    result["a_lon"] = grouped["v_lon"].diff() / elapsed_hours.replace(0, np.nan)

    direction_change = grouped["storm_dir"].diff()
    direction_change = direction_change.where(direction_change <= 180, direction_change - 360)
    direction_change = direction_change.where(direction_change >= -180, direction_change + 360)
    result["track_curvature"] = direction_change.abs() / (
        result["storm_speed"].replace(0, np.nan) + 1e-6
    )
    result["storm_age_hours"] = grouped["iso_time"].transform(
        lambda times: (times - times.min()).dt.total_seconds() / 3600
    )
    result["day_of_year"] = result["iso_time"].dt.dayofyear
    result["month"] = result["iso_time"].dt.month
    result["latitude_regime"] = pd.cut(
        result["lat"].abs(),
        bins=[0, 18, 28, 90],
        labels=[0, 1, 2],
        right=False,
    ).astype(float)
    result["hemisphere"] = (result["lat"] > 0).astype(int)
    result["motion_regime"] = [
        classify_motion_regime(speed, direction)
        for speed, direction in zip(result["storm_speed"], result["storm_dir"], strict=True)
    ]

    if "nature" in result:
        result["storm_stage"] = result["nature"].map(STORM_STAGE_CODES)
    if "dist2land" in result:
        result["is_approaching_land"] = (result["dist2land"] < 200).astype(int)
        result["land_gradient"] = grouped["dist2land"].diff() * -1

    absolute_latitude = np.radians(result["lat"].abs())
    result["beta_drift_lat"] = result["hemisphere"] * np.sin(absolute_latitude) * 0.1
    result["beta_drift_lon"] = (
        result["hemisphere"] * np.sign(result["lat"]) * np.cos(absolute_latitude) * 0.05
    )
    grouped = result.groupby("sid", sort=False)
    for column in ("v_lat", "v_lon"):
        result[f"{column}_smooth"] = grouped[column].transform(
            lambda values: values.rolling(3, center=True, min_periods=1).mean()
        )
        result[f"{column}_avg_6h"] = grouped[column].transform(
            lambda values: values.rolling(2, min_periods=1).mean()
        )
        result[f"{column}_avg_12h"] = grouped[column].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    return result
