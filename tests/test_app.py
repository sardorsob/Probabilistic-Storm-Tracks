import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.data_loader import clear_dashboard_data_cache
from scripts.build_dashboard_data import build_dashboard_files

FIXTURES = Path(__file__).parent / "fixtures" / "dashboard"


def build_contract(tmp_path: Path, *, include_tracks: bool = True) -> Path:
    persistence_path = tmp_path / "persistence.pkl"
    kalman_path = tmp_path / "kalman.pkl"
    tracks_path = tmp_path / "tracks.pkl"
    output_dir = tmp_path / "dashboard"
    pd.read_csv(FIXTURES / "persistence_results.csv").to_pickle(persistence_path)
    pd.read_csv(FIXTURES / "kalman_results.csv").to_pickle(kalman_path)
    if include_tracks:
        pd.read_csv(FIXTURES / "tracks.csv").to_pickle(tracks_path)
    build_dashboard_files(
        persistence_path,
        kalman_path,
        output_dir,
        tracks_path=tracks_path if include_tracks else None,
    )
    return output_dir


def test_loader_reads_manifest_validated_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.data_loader import load_dashboard_data, resolve_data_dir

    data_dir = build_contract(tmp_path)
    monkeypatch.setenv("STORM_TRACKS_DASHBOARD_DATA_DIR", str(data_dir))
    clear_dashboard_data_cache()

    bundle = load_dashboard_data(resolve_data_dir())

    assert bundle.headline_split == "test"
    assert bundle.manifest["schema_version"] == "1.0.0"
    assert set(bundle.summary["model"]) == {"persistence", "kalman"}
    assert bundle.tracks is not None
    assert str(bundle.tracks["time_utc"].dtype) == "datetime64[ns, UTC]"


def test_loader_refreshes_after_contract_rebuild(tmp_path: Path) -> None:
    from app.data_loader import load_dashboard_data

    data_dir = build_contract(tmp_path)
    clear_dashboard_data_cache()
    first = load_dashboard_data(data_dir)
    initial_rmse = float(first.summary.loc[0, "rmse_km"])

    summary_path = data_dir / "model_comparison_summary.csv"
    summary = pd.read_csv(summary_path)
    summary.loc[0, "rmse_km"] = initial_rmse + 1
    summary.to_csv(summary_path, index=False)
    manifest_path = data_dir / "dashboard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generated_at_utc"] = "2030-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    refreshed = load_dashboard_data(data_dir)

    assert float(refreshed.summary.loc[0, "rmse_km"]) == pytest.approx(initial_rmse + 1)


def test_loader_rejects_manifest_row_count_mismatch(tmp_path: Path) -> None:
    from app.data_loader import DashboardDataError, load_dashboard_data

    data_dir = build_contract(tmp_path)
    manifest_path = data_dir / "dashboard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_counts"]["error_distributions.csv"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    clear_dashboard_data_cache()

    with pytest.raises(DashboardDataError, match="row count"):
        load_dashboard_data(data_dir)


def test_loader_rejects_manifest_listed_missing_file(tmp_path: Path) -> None:
    from app.data_loader import DashboardDataError, load_dashboard_data

    data_dir = build_contract(tmp_path)
    (data_dir / "error_distributions.csv").unlink()
    clear_dashboard_data_cache()

    with pytest.raises(DashboardDataError, match="Dashboard file is missing"):
        load_dashboard_data(data_dir)


def test_loader_allows_contract_without_track_tables(tmp_path: Path) -> None:
    from app.data_loader import load_dashboard_data

    data_dir = build_contract(tmp_path, include_tracks=False)
    clear_dashboard_data_cache()

    bundle = load_dashboard_data(data_dir)

    assert bundle.metadata is None
    assert bundle.tracks is None


def test_observed_track_figure_sorts_points_and_labels_time() -> None:
    from app.pages.storm_explorer import build_track_figure

    points = pd.DataFrame(
        {
            "storm_id": ["A", "A", "A"],
            "time_utc": pd.to_datetime(
                ["2020-08-01 12:00Z", "2020-08-01 00:00Z", "2020-08-01 06:00Z"],
                utc=True,
            ),
            "latitude": [12.0, 10.0, 11.0],
            "longitude": [-48.0, -50.0, -49.0],
        }
    )

    figure = build_track_figure(points)

    observed = figure.data[0]
    assert observed.type == "scattermap"
    assert list(observed.lat) == [10.0, 11.0, 12.0]
    assert list(observed.lon) == [-50.0, -49.0, -48.0]
    assert observed.name == "Observed track"
    assert "UTC" in observed.hovertemplate
    assert figure.layout.map.style == "white-bg"
    endpoint_colors = {figure.data[-2].marker.color, figure.data[-1].marker.color}
    assert endpoint_colors.isdisjoint({"#087F8C", "#C26D0A"})
    assert list(figure.data[-2].text) == ["S"]
    assert list(figure.data[-1].text) == ["E"]


def test_observed_track_figure_splits_at_the_antimeridian() -> None:
    from app.pages.storm_explorer import build_track_figure

    points = pd.DataFrame(
        {
            "storm_id": ["A", "A", "A", "A"],
            "time_utc": pd.date_range("2020-08-01", periods=4, freq="6h", tz="UTC"),
            "latitude": [10.0, 10.5, 11.0, 11.5],
            "longitude": [178.0, 179.5, -179.5, -178.0],
        }
    )

    figure = build_track_figure(points)

    observed_segments = [trace for trace in figure.data if trace.name == "Observed track"]
    assert len(observed_segments) == 2
    for segment in observed_segments:
        longitudes = pd.Series(segment.lon)
        assert longitudes.diff().abs().dropna().lt(180).all()


def test_loader_rejects_incomplete_track_columns(tmp_path: Path) -> None:
    from app.data_loader import DashboardDataError, load_dashboard_data

    data_dir = build_contract(tmp_path)
    tracks_path = data_dir / "storm_tracks.csv.gz"
    tracks = pd.read_csv(tracks_path).drop(columns="longitude")
    tracks.to_csv(tracks_path, index=False, compression="gzip")
    clear_dashboard_data_cache()

    with pytest.raises(DashboardDataError, match="storm_tracks.csv.gz is missing columns"):
        load_dashboard_data(data_dir)


def test_loader_rejects_columns_required_by_dashboard_pages(tmp_path: Path) -> None:
    from app.data_loader import DashboardDataError, load_dashboard_data

    data_dir = build_contract(tmp_path)
    comparison_path = data_dir / "model_comparison_detailed.csv"
    comparison = pd.read_csv(comparison_path).drop(columns="better_model")
    comparison.to_csv(comparison_path, index=False)
    clear_dashboard_data_cache()

    with pytest.raises(
        DashboardDataError,
        match="model_comparison_detailed.csv is missing columns",
    ):
        load_dashboard_data(data_dir)


def test_loader_reports_invalid_track_timestamps(tmp_path: Path) -> None:
    from app.data_loader import DashboardDataError, load_dashboard_data

    data_dir = build_contract(tmp_path)
    tracks_path = data_dir / "storm_tracks.csv.gz"
    tracks = pd.read_csv(tracks_path)
    tracks.loc[0, "time_utc"] = "not-a-timestamp"
    tracks.to_csv(tracks_path, index=False, compression="gzip")
    clear_dashboard_data_cache()

    with pytest.raises(DashboardDataError, match="invalid UTC timestamps"):
        load_dashboard_data(data_dir)


def test_app_uses_native_navigation_and_supported_width_api(repository_root: Path) -> None:
    source_files = [
        repository_root / "app" / "app.py",
        *(repository_root / "app/pages").glob("*.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    assert "st.navigation" in source
    assert "st.Page" in source
    assert "use_container_width" not in source
    assert "cdn" not in source.lower()
    assert 'width="stretch"' in source


def test_methodology_report_link_targets_the_tracked_report(repository_root: Path) -> None:
    from app.pages.methodology import REPORT_URL

    report_path = Path(
        "docs/reports/sequential_bayesian_inference_applied_to_hurricane_tracking.pdf"
    )

    assert (repository_root / report_path).is_file()
    assert REPORT_URL.endswith(report_path.as_posix())


def test_app_shows_clear_message_when_dashboard_data_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("STORM_TRACKS_DASHBOARD_DATA_DIR", str(missing))

    app = AppTest.from_file(repository_root / "app" / "app.py").run(timeout=10)

    assert not app.exception
    assert any("Dashboard data is not available" in item.value for item in app.warning)
    assert any("build_dashboard_data.py" in item.value for item in app.markdown)


def test_app_renders_overview_from_small_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    data_dir = build_contract(tmp_path)
    monkeypatch.setenv("STORM_TRACKS_DASHBOARD_DATA_DIR", str(data_dir))

    app = AppTest.from_file(repository_root / "app" / "app.py").run(timeout=10)

    assert not app.exception
    assert any(item.value == "Storm track forecast evaluation" for item in app.title)
    assert any("Test split" in item.value for item in app.caption)


def test_storm_explorer_explains_when_tracks_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = build_contract(tmp_path, include_tracks=False)
    monkeypatch.setenv("STORM_TRACKS_DASHBOARD_DATA_DIR", str(data_dir))
    page = tmp_path / "storm_page.py"
    page.write_text(
        "from app.data_loader import load_dashboard_data, resolve_data_dir\n"
        "from app.pages.storm_explorer import render\n"
        "render(load_dashboard_data(resolve_data_dir()))\n",
        encoding="utf-8",
    )

    app = AppTest.from_file(page).run(timeout=10)

    assert not app.exception
    assert any("Observed track points are not included" in item.value for item in app.info)
