import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_pipeline_dry_run_lists_stages_without_execution(
    repository_root: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--dry-run",
            "--config",
            "configs/pipeline.toml",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.index("prepare_data") < result.stdout.index("run_models")
    assert result.stdout.index("run_models") < result.stdout.index("build_dashboard_data")
    assert "EXECUTING" not in result.stdout
    assert "required inputs" in result.stdout.lower()
    assert "expected outputs" in result.stdout.lower()
    assert "scripts/build_dashboard_data.py" in result.stdout
    assert "data/test_persistence_results.pkl" in result.stdout
    assert "data/test_kalman_results.pkl" in result.stdout
    assert "--output-dir app/data" in result.stdout
    assert "streamboard" not in result.stdout


def test_prepare_data_runs_on_a_small_ibtracs_file(tmp_path: Path) -> None:
    from scripts.prepare_data import prepare_data

    raw_path = tmp_path / "ibtracs.csv"
    output_path = tmp_path / "processed.pkl"
    raw_path.write_text(
        "SID,ISO_TIME,LAT,LON,STORM_DIR,STORM_SPEED\n"
        " , ,degrees,degrees,degrees,knots\n"
        "A,2020-01-01 00:00:00,10,179,90,12\n"
        "A,2020-01-01 06:00:00,10,-179,90,12\n",
        encoding="utf-8",
    )

    prepared = prepare_data(raw_path, output_path, min_observations=2)

    assert output_path.exists()
    assert len(prepared) == 2
    assert str(prepared["iso_time"].dtype) == "datetime64[ns, UTC]"
    assert prepared.loc[1, "x_km"] > 0
    pd.testing.assert_frame_equal(prepared, pd.read_pickle(output_path))


def test_prepare_data_keeps_only_selected_agency_rows(tmp_path: Path) -> None:
    from scripts.prepare_data import prepare_data

    raw_path = tmp_path / "ibtracs.csv"
    output_path = tmp_path / "processed.pkl"
    raw_path.write_text(
        "SID,ISO_TIME,BASIN,LAT,LON,STORM_DIR,STORM_SPEED,USA_AGENCY,USA_LAT,USA_LON\n"
        " , , ,degrees,degrees,degrees,knots, ,degrees,degrees\n"
        "A,2020-01-01 00:00:00,NA,10,-50,90,12,hurdat_atl,10,-50\n"
        "A,2020-01-01 06:00:00,NA,10,-49,90,12,hurdat_atl,10,-49\n"
        "B,2020-01-01 00:00:00,NA,10,-50,90,12,other,10,-50\n"
        "B,2020-01-01 06:00:00,NA,10,-49,90,12,other,10,-49\n",
        encoding="utf-8",
    )

    prepared = prepare_data(raw_path, output_path, min_observations=2)

    assert set(prepared["sid"]) == {"A"}
    assert prepared["use_for_modeling"].all()


def test_run_models_writes_split_aware_results(tmp_path: Path) -> None:
    from scripts.run_models import run_models

    processed_path = tmp_path / "processed.pkl"
    output_dir = tmp_path / "results"
    steps = np.arange(10, dtype=float)
    tracks = []
    for offset, storm_id in enumerate("ABCDEF"):
        tracks.append(
            pd.DataFrame(
                {
                    "sid": storm_id,
                    "iso_time": pd.date_range(
                        "2020-01-01", periods=len(steps), freq="6h", tz="UTC"
                    ),
                    "x_km": steps * 10 + offset,
                    "y_km": steps * 5 + offset,
                    "vx_km": np.full(len(steps), 10.0),
                    "vy_km": np.full(len(steps), 5.0),
                }
            )
        )
    pd.concat(tracks, ignore_index=True).to_pickle(processed_path)

    outputs = run_models(
        processed_path,
        output_dir,
        split="test",
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
        lead_times_hours=(6, 12),
        min_origin_index=2,
        max_origins_per_storm=1,
        covariance_min_storm_length=5,
    )

    persistence = pd.read_pickle(outputs["persistence_results"])
    kalman = pd.read_pickle(outputs["kalman_results"])
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert set(persistence["split"]) == {"test"}
    assert set(kalman["split"]) == {"test"}
    assert metadata["split"] == "test"
    assert metadata["seed"] == 7
    assert "not regenerated" in metadata["historical_results_warning"].lower()
