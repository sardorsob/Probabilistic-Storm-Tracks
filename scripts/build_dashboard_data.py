"""Build the validated CSV contract consumed by the Streamlit app."""

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from storm_tracks.dashboard_data import DashboardData, build_dashboard_data  # noqa: E402

SCHEMA_VERSION = "1.0.0"
OPTIONAL_TABLE_FILES = {"storm_metadata.csv", "storm_tracks.csv.gz"}


def _temporary_path(destination: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _atomic_write_csv(
    frame: pd.DataFrame,
    destination: Path,
    *,
    compression: str | None = None,
) -> None:
    temporary = _temporary_path(destination)
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression=compression,
            date_format="%Y-%m-%dT%H:%M:%SZ",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(data: dict[str, object], destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _table_files(tables: DashboardData) -> dict[str, tuple[pd.DataFrame, str | None]]:
    files = {
        "model_comparison_summary.csv": (tables.model_comparison_summary, None),
        "error_distributions.csv": (tables.error_distributions, None),
        "storm_performance.csv": (tables.storm_performance, None),
        "model_comparison_detailed.csv": (tables.model_comparison_detailed, None),
    }
    if tables.storm_metadata is not None:
        files["storm_metadata.csv"] = (tables.storm_metadata, None)
    if tables.storm_tracks is not None:
        files["storm_tracks.csv.gz"] = (tables.storm_tracks, "gzip")
    return files


def build_dashboard_files(
    persistence_path: str | Path,
    kalman_path: str | Path,
    output_dir: str | Path,
    *,
    tracks_path: str | Path | None = None,
    headline_split: str = "test",
) -> dict[str, Path]:
    """Build and atomically write dashboard tables and their manifest."""
    destination = Path(output_dir).resolve()
    if destination == PROJECT_ROOT.resolve():
        raise ValueError("dashboard output directory cannot be the repository root")

    persistence_source = Path(persistence_path)
    kalman_source = Path(kalman_path)
    tracks_source = Path(tracks_path) if tracks_path is not None else None
    persistence = pd.read_pickle(persistence_source)
    kalman = pd.read_pickle(kalman_source)
    tracks = pd.read_pickle(tracks_source) if tracks_source is not None else None
    tables = build_dashboard_data(
        persistence,
        kalman,
        tracks=tracks,
        headline_split=headline_split,
    )

    destination.mkdir(parents=True, exist_ok=True)
    table_files = _table_files(tables)
    manifest_path = destination / "dashboard_manifest.json"
    manifest_path.unlink(missing_ok=True)
    for filename in OPTIONAL_TABLE_FILES.difference(table_files):
        (destination / filename).unlink(missing_ok=True)
    outputs: dict[str, Path] = {}
    for filename, (table, compression) in table_files.items():
        output_path = destination / filename
        _atomic_write_csv(table, output_path, compression=compression)
        outputs[filename] = output_path

    sources = {
        "persistence_results": persistence_source.name,
        "kalman_results": kalman_source.name,
    }
    if tracks_source is not None:
        sources["tracks"] = tracks_source.name
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_artifacts": sources,
        "split_policy": {
            "headline_split": headline_split,
            "storm_membership": "Each storm belongs to exactly one split.",
        },
        "allowed_forecast_horizons_hours": [6, 12, 24, 48, 72],
        "metric_definitions": {
            "error_km": (
                "Euclidean distance between forecast and observed positions, in kilometers."
            ),
            "rmse_km": "Square root of mean squared position error, in kilometers.",
            "rmse_improvement_km": (
                "Persistence RMSE minus Kalman RMSE; positive values favor Kalman."
            ),
            "rmse_improvement_pct": (
                "RMSE improvement divided by persistence RMSE; null when persistence RMSE is zero."
            ),
        },
        "models": ["persistence", "kalman"],
        "files": sorted(table_files),
        "row_counts": {
            filename: int(len(table)) for filename, (table, _) in table_files.items()
        },
    }
    _atomic_write_json(manifest, manifest_path)
    outputs[manifest_path.name] = manifest_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistence-results", type=Path, required=True)
    parser.add_argument("--kalman-results", type=Path, required=True)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headline-split", default="test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = build_dashboard_files(
        args.persistence_results,
        args.kalman_results,
        args.output_dir,
        tracks_path=args.tracks,
        headline_split=args.headline_split,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
