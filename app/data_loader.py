import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

SCHEMA_VERSION = "1.0.0"
DATA_DIR_ENV = "STORM_TRACKS_DASHBOARD_DATA_DIR"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_COLUMNS = {
    "model_comparison_summary.csv": {
        "model",
        "split",
        "forecast_horizon_hours",
        "forecast_count",
        "rmse_km",
    },
    "error_distributions.csv": {
        "storm_id",
        "origin_idx",
        "forecast_horizon_hours",
        "error_km",
        "split",
        "model",
    },
    "storm_performance.csv": {
        "storm_id",
        "model",
        "split",
        "forecast_horizon_hours",
        "forecast_count",
        "rmse_km",
    },
    "model_comparison_detailed.csv": {
        "storm_id",
        "split",
        "forecast_horizon_hours",
        "persistence_rmse_km",
        "kalman_rmse_km",
        "rmse_improvement_km",
        "rmse_improvement_pct",
        "better_model",
    },
    "storm_metadata.csv": {
        "storm_id",
        "split",
        "start_time_utc",
        "end_time_utc",
        "observation_count",
    },
    "storm_tracks.csv.gz": {
        "storm_id",
        "split",
        "time_utc",
        "latitude",
        "longitude",
    },
}


class DashboardDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class DashboardBundle:
    manifest: dict[str, Any]
    summary: pd.DataFrame
    errors: pd.DataFrame
    performance: pd.DataFrame
    comparison: pd.DataFrame
    metadata: pd.DataFrame | None
    tracks: pd.DataFrame | None

    @property
    def headline_split(self) -> str:
        return str(self.manifest["split_policy"]["headline_split"])


def resolve_data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)).expanduser().resolve()


def _read_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "dashboard_manifest.json"
    if not manifest_path.is_file():
        raise DashboardDataError(
            f"Dashboard data is not available at {data_dir}. "
            "Build it with scripts/build_dashboard_data.py."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read {manifest_path.name}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        found = manifest.get("schema_version", "missing")
        raise DashboardDataError(
            f"Dashboard schema {found!r} is not supported; expected {SCHEMA_VERSION!r}."
        )
    if not isinstance(manifest.get("files"), list):
        raise DashboardDataError("Dashboard manifest is missing its file list.")
    if not isinstance(manifest.get("row_counts"), dict):
        raise DashboardDataError("Dashboard manifest is missing row counts.")
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict) or not split_policy.get("headline_split"):
        raise DashboardDataError("Dashboard manifest is missing the headline split.")
    return manifest


def _read_table(
    data_dir: Path,
    manifest: dict[str, Any],
    filename: str,
    *,
    required: bool = True,
) -> pd.DataFrame | None:
    listed_files = set(manifest["files"])
    path = data_dir / filename
    if filename not in listed_files:
        if required:
            raise DashboardDataError(f"Dashboard manifest does not list {filename}.")
        return None
    if not path.is_file():
        raise DashboardDataError(f"Dashboard file is missing: {filename}.")
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DashboardDataError(f"Could not read {filename}: {exc}") from exc
    expected_rows = manifest["row_counts"].get(filename)
    if expected_rows != len(table):
        raise DashboardDataError(
            f"Dashboard row count for {filename} is {len(table)}; manifest says {expected_rows}."
        )
    required_columns = REQUIRED_COLUMNS.get(filename, set())
    missing = required_columns.difference(table.columns)
    if missing:
        raise DashboardDataError(f"{filename} is missing columns: {sorted(missing)}.")
    return table


def _parse_utc_columns(table: pd.DataFrame, filename: str, columns: tuple[str, ...]) -> None:
    try:
        for column in columns:
            table[column] = pd.to_datetime(
                table[column],
                errors="raise",
                format="mixed",
                utc=True,
            )
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"{filename} contains invalid UTC timestamps: {exc}") from exc


def _manifest_digest(data_dir: Path) -> str:
    try:
        content = (data_dir / "dashboard_manifest.json").read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(content).hexdigest()


@st.cache_data(show_spinner=False)
def _load_dashboard_data(source: Path, manifest_digest: str) -> DashboardBundle:
    manifest = _read_manifest(source)
    summary = _read_table(source, manifest, "model_comparison_summary.csv")
    errors = _read_table(source, manifest, "error_distributions.csv")
    performance = _read_table(source, manifest, "storm_performance.csv")
    comparison = _read_table(source, manifest, "model_comparison_detailed.csv")
    metadata = _read_table(source, manifest, "storm_metadata.csv", required=False)
    tracks = _read_table(source, manifest, "storm_tracks.csv.gz", required=False)
    assert summary is not None
    assert errors is not None
    assert performance is not None
    assert comparison is not None
    if metadata is not None:
        _parse_utc_columns(
            metadata,
            "storm_metadata.csv",
            ("start_time_utc", "end_time_utc"),
        )
    if tracks is not None:
        _parse_utc_columns(tracks, "storm_tracks.csv.gz", ("time_utc",))
    return DashboardBundle(
        manifest=manifest,
        summary=summary,
        errors=errors,
        performance=performance,
        comparison=comparison,
        metadata=metadata,
        tracks=tracks,
    )


def load_dashboard_data(data_dir: str | Path) -> DashboardBundle:
    source = Path(data_dir).expanduser().resolve()
    return _load_dashboard_data(source, _manifest_digest(source))


def clear_dashboard_data_cache() -> None:
    _load_dashboard_data.clear()
