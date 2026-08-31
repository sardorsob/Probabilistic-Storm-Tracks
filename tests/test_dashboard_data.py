import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import build_dashboard_data as dashboard_script
from storm_tracks.dashboard_data import build_dashboard_data

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    persistence = pd.read_csv(FIXTURES / "persistence_results.csv")
    kalman = pd.read_csv(FIXTURES / "kalman_results.csv")
    return persistence, kalman


def test_summary_uses_test_results_and_true_rmse() -> None:
    persistence, kalman = load_results()

    tables = build_dashboard_data(persistence, kalman)

    summary = tables.model_comparison_summary
    assert set(summary["split"]) == {"test"}
    assert set(summary["model"]) == {"persistence", "kalman"}
    persistence_at_6h = summary.loc[
        (summary["model"] == "persistence")
        & (summary["forecast_horizon_hours"] == 6)
    ].iloc[0]
    assert persistence_at_6h["forecast_count"] == 4
    assert persistence_at_6h["mean_error_km"] == pytest.approx(5.25)
    assert persistence_at_6h["rmse_km"] == pytest.approx(np.sqrt(31.25))
    assert persistence_at_6h["error_p25_km"] == pytest.approx(3.75)
    assert persistence_at_6h["error_p75_km"] == pytest.approx(6.5)


def test_error_distribution_keeps_split_model_and_forecast_identity() -> None:
    persistence, kalman = load_results()

    errors = build_dashboard_data(persistence, kalman).error_distributions

    assert list(errors.columns) == [
        "storm_id",
        "origin_idx",
        "forecast_horizon_hours",
        "error_km",
        "split",
        "model",
    ]
    assert set(errors["split"]) == {"test", "validation"}
    assert set(errors["model"]) == {"persistence", "kalman"}
    assert "lead_time_hours" not in errors


def test_storm_performance_has_one_row_per_model_split_storm_and_horizon() -> None:
    persistence, kalman = load_results()

    performance = build_dashboard_data(persistence, kalman).storm_performance

    row = performance.loc[
        (performance["model"] == "persistence")
        & (performance["split"] == "test")
        & (performance["storm_id"] == "A")
        & (performance["forecast_horizon_hours"] == 6)
    ].iloc[0]
    assert row["forecast_count"] == 2
    assert row["mean_error_km"] == pytest.approx(3.5)
    assert row["rmse_km"] == pytest.approx(np.sqrt(12.5))


def test_detailed_comparison_uses_signed_rmse_improvement() -> None:
    persistence, kalman = load_results()

    comparison = build_dashboard_data(persistence, kalman).model_comparison_detailed

    row = comparison.loc[
        (comparison["split"] == "test")
        & (comparison["storm_id"] == "A")
        & (comparison["forecast_horizon_hours"] == 6)
    ].iloc[0]
    assert row["forecast_count"] == 2
    assert row["persistence_rmse_km"] == pytest.approx(np.sqrt(12.5))
    assert row["kalman_rmse_km"] == pytest.approx(np.sqrt(3.125))
    assert row["rmse_improvement_km"] == pytest.approx(np.sqrt(12.5) - np.sqrt(3.125))
    assert row["rmse_improvement_pct"] == pytest.approx(50.0)
    assert row["better_model"] == "kalman"


def test_zero_persistence_rmse_has_no_percentage_improvement() -> None:
    persistence, kalman = load_results()
    matching_rows = (
        (persistence["storm_id"] == "A") & (persistence["lead_time_hours"] == 6)
    )
    persistence.loc[matching_rows, "error_km"] = 0.0
    kalman.loc[matching_rows, "error_km"] = 0.0

    with np.errstate(divide="raise", invalid="raise"):
        comparison = build_dashboard_data(persistence, kalman).model_comparison_detailed

    row = comparison.loc[
        (comparison["split"] == "test")
        & (comparison["storm_id"] == "A")
        & (comparison["forecast_horizon_hours"] == 6)
    ].iloc[0]
    assert np.isnan(row["rmse_improvement_pct"])
    assert row["better_model"] == "tie"


@pytest.mark.parametrize("invalid_error", [-1.0, np.nan, np.inf])
def test_results_reject_invalid_errors(invalid_error: float) -> None:
    persistence, kalman = load_results()
    persistence.loc[0, "error_km"] = invalid_error

    with pytest.raises(ValueError, match="finite and non-negative"):
        build_dashboard_data(persistence, kalman)


def test_results_reject_unknown_forecast_horizons() -> None:
    persistence, kalman = load_results()
    persistence.loc[0, "lead_time_hours"] = 18

    with pytest.raises(ValueError, match="unsupported forecast horizons.*18"):
        build_dashboard_data(persistence, kalman)


def test_results_reject_duplicate_forecast_keys() -> None:
    persistence, kalman = load_results()
    persistence = pd.concat([persistence, persistence.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate persistence forecast keys"):
        build_dashboard_data(persistence, kalman)


def test_results_require_matching_model_forecast_keys() -> None:
    persistence, kalman = load_results()
    kalman = kalman.iloc[1:].reset_index(drop=True)

    with pytest.raises(ValueError, match="matching forecast keys"):
        build_dashboard_data(persistence, kalman)


def test_results_reject_storms_shared_across_splits() -> None:
    persistence, kalman = load_results()
    persistence_leak = persistence.iloc[[0]].assign(split="validation")
    kalman_leak = kalman.iloc[[0]].assign(split="validation")
    persistence = pd.concat([persistence, persistence_leak], ignore_index=True)
    kalman = pd.concat([kalman, kalman_leak], ignore_index=True)

    with pytest.raises(ValueError, match="storms appear in more than one split.*A"):
        build_dashboard_data(persistence, kalman)


def test_summary_requires_the_requested_split() -> None:
    persistence, kalman = load_results()

    with pytest.raises(ValueError, match="no forecast results found for split 'holdout'"):
        build_dashboard_data(persistence, kalman, headline_split="holdout")


def test_tracks_are_utc_and_limited_to_evaluated_storms() -> None:
    persistence, kalman = load_results()
    tracks = pd.read_csv(FIXTURES / "tracks.csv")

    tables = build_dashboard_data(persistence, kalman, tracks=tracks)

    assert tables.storm_tracks is not None
    assert list(tables.storm_tracks.columns) == [
        "storm_id",
        "split",
        "time_utc",
        "latitude",
        "longitude",
    ]
    assert set(tables.storm_tracks["storm_id"]) == {"A", "B", "C"}
    assert str(tables.storm_tracks["time_utc"].dtype) == "datetime64[ns, UTC]"
    assert tables.storm_metadata is not None
    storm_a = tables.storm_metadata.loc[tables.storm_metadata["storm_id"] == "A"].iloc[0]
    assert storm_a["split"] == "test"
    assert storm_a["observation_count"] == 2
    assert storm_a["start_time_utc"] == pd.Timestamp("2020-08-01 00:00:00", tz="UTC")
    assert storm_a["end_time_utc"] == pd.Timestamp("2020-08-01 06:00:00", tz="UTC")


def test_track_tables_are_optional() -> None:
    persistence, kalman = load_results()

    tables = build_dashboard_data(persistence, kalman)

    assert tables.storm_tracks is None
    assert tables.storm_metadata is None


def test_tracks_reject_duplicate_storm_times() -> None:
    persistence, kalman = load_results()
    tracks = pd.read_csv(FIXTURES / "tracks.csv")
    tracks = pd.concat([tracks, tracks.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate storm timestamps"):
        build_dashboard_data(persistence, kalman, tracks=tracks)


def test_tracks_require_every_evaluated_storm() -> None:
    persistence, kalman = load_results()
    tracks = pd.read_csv(FIXTURES / "tracks.csv").query("sid != 'B'")

    with pytest.raises(ValueError, match="missing track points.*B"):
        build_dashboard_data(persistence, kalman, tracks=tracks)


def test_build_dashboard_data_script_writes_tables_and_manifest_last(
    tmp_path: Path,
    repository_root: Path,
) -> None:
    persistence, kalman = load_results()
    persistence_path = tmp_path / "persistence.pkl"
    kalman_path = tmp_path / "kalman.pkl"
    tracks_path = tmp_path / "tracks.pkl"
    output_dir = tmp_path / "dashboard"
    persistence.to_pickle(persistence_path)
    kalman.to_pickle(kalman_path)
    pd.read_csv(FIXTURES / "tracks.csv").to_pickle(tracks_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_dashboard_data.py",
            "--persistence-results",
            str(persistence_path),
            "--kalman-results",
            str(kalman_path),
            "--tracks",
            str(tracks_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository_root,
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected_files = {
        "model_comparison_summary.csv",
        "error_distributions.csv",
        "storm_performance.csv",
        "model_comparison_detailed.csv",
        "storm_metadata.csv",
        "storm_tracks.csv.gz",
        "dashboard_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files
    manifest_path = output_dir / "dashboard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["split_policy"]["headline_split"] == "test"
    assert manifest["source_artifacts"] == {
        "kalman_results": "kalman.pkl",
        "persistence_results": "persistence.pkl",
        "tracks": "tracks.pkl",
    }
    assert manifest["row_counts"]["model_comparison_summary.csv"] == 4
    assert manifest["metric_definitions"]["rmse_km"].startswith("Square root")
    data_mtimes = [
        (output_dir / filename).stat().st_mtime_ns
        for filename in expected_files - {"dashboard_manifest.json"}
    ]
    assert manifest_path.stat().st_mtime_ns >= max(data_mtimes)
    assert not list(output_dir.glob("*.tmp"))


def test_dashboard_writer_refuses_the_repository_root(repository_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_dashboard_data.py",
            "--persistence-results",
            "unused-persistence.pkl",
            "--kalman-results",
            "unused-kalman.pkl",
            "--output-dir",
            str(repository_root),
        ],
        cwd=repository_root,
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "repository root" in result.stderr


def test_failed_table_write_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence, kalman = load_results()
    persistence_path = tmp_path / "persistence.pkl"
    kalman_path = tmp_path / "kalman.pkl"
    output_dir = tmp_path / "dashboard"
    output_dir.mkdir()
    manifest_path = output_dir / "dashboard_manifest.json"
    manifest_path.write_text('{"schema_version": "stale"}\n', encoding="utf-8")
    persistence.to_pickle(persistence_path)
    kalman.to_pickle(kalman_path)

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(dashboard_script, "_atomic_write_csv", fail_write)

    with pytest.raises(OSError, match="simulated write failure"):
        dashboard_script.build_dashboard_files(
            persistence_path,
            kalman_path,
            output_dir,
        )

    assert not manifest_path.exists()


def test_rebuild_without_tracks_removes_stale_track_files(tmp_path: Path) -> None:
    persistence, kalman = load_results()
    persistence_path = tmp_path / "persistence.pkl"
    kalman_path = tmp_path / "kalman.pkl"
    tracks_path = tmp_path / "tracks.pkl"
    output_dir = tmp_path / "dashboard"
    persistence.to_pickle(persistence_path)
    kalman.to_pickle(kalman_path)
    pd.read_csv(FIXTURES / "tracks.csv").to_pickle(tracks_path)

    dashboard_script.build_dashboard_files(
        persistence_path,
        kalman_path,
        output_dir,
        tracks_path=tracks_path,
    )
    assert (output_dir / "storm_metadata.csv").exists()
    assert (output_dir / "storm_tracks.csv.gz").exists()

    dashboard_script.build_dashboard_files(
        persistence_path,
        kalman_path,
        output_dir,
    )

    assert not (output_dir / "storm_metadata.csv").exists()
    assert not (output_dir / "storm_tracks.csv.gz").exists()
