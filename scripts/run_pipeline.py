"""Display or execute the configured storm-track workflow."""

import argparse
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    runtime: str


def load_config(path: Path) -> dict[str, object]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def _value(section: Mapping[str, object], name: str) -> object:
    if name not in section:
        raise ValueError(f"missing configuration value: {name}")
    return section[name]


def build_stages(config: Mapping[str, object]) -> tuple[Stage, ...]:
    """Build the only ordered list of executable workflow stages."""
    paths = config["paths"]
    preparation = config["preparation"]
    splits = config["splits"]
    forecast = config["forecast"]
    model = config["model"]
    if not all(
        isinstance(section, Mapping)
        for section in (paths, preparation, splits, forecast, model)
    ):
        raise ValueError("pipeline sections must be TOML tables")

    raw_data = str(_value(paths, "raw_data"))
    processed_data = str(_value(paths, "processed_data"))
    results_dir = str(_value(paths, "results_dir"))
    dashboard_data_dir = str(_value(paths, "dashboard_data_dir"))
    split = str(_value(splits, "evaluation"))
    kalman_result = f"{results_dir}/{split}_kalman_results.pkl"
    persistence_result = f"{results_dir}/{split}_persistence_results.pkl"
    lead_times = tuple(str(value) for value in _value(forecast, "lead_times_hours"))
    model_command = (
        sys.executable,
        "scripts/run_models.py",
        "--input",
        processed_data,
        "--output-dir",
        results_dir,
        "--split",
        split,
        "--seed",
        str(_value(splits, "seed")),
        "--train-fraction",
        str(_value(splits, "train_fraction")),
        "--validation-fraction",
        str(_value(splits, "validation_fraction")),
        "--lead-times",
        *lead_times,
        "--min-origin-index",
        str(_value(forecast, "min_origin_index")),
        "--max-origins",
        str(_value(forecast, "max_origins_per_storm")),
        "--covariance-min-storm-length",
        str(_value(model, "covariance_min_storm_length")),
        "--covariance-sample-size",
        str(_value(model, "covariance_sample_size")),
    )
    return (
        Stage(
            name="prepare_data",
            command=(
                sys.executable,
                "scripts/prepare_data.py",
                "--input",
                raw_data,
                "--output",
                processed_data,
                "--min-observations",
                str(_value(preparation, "min_observations")),
            ),
            required_inputs=(raw_data,),
            expected_outputs=(processed_data,),
            runtime="long: parses and engineers the full IBTrACS history",
        ),
        Stage(
            name="run_models",
            command=model_command,
            required_inputs=(processed_data,),
            expected_outputs=(kalman_result, persistence_result),
            runtime="long: estimates covariance and evaluates forecast origins",
        ),
        Stage(
            name="build_dashboard_data",
            command=(
                sys.executable,
                "scripts/build_dashboard_data.py",
                "--persistence-results",
                persistence_result,
                "--kalman-results",
                kalman_result,
                "--tracks",
                processed_data,
                "--output-dir",
                dashboard_data_dir,
                "--headline-split",
                split,
            ),
            required_inputs=(processed_data, kalman_result, persistence_result),
            expected_outputs=(
                f"{dashboard_data_dir}/model_comparison_summary.csv",
                f"{dashboard_data_dir}/error_distributions.csv",
                f"{dashboard_data_dir}/storm_performance.csv",
                f"{dashboard_data_dir}/model_comparison_detailed.csv",
                f"{dashboard_data_dir}/storm_metadata.csv",
                f"{dashboard_data_dir}/storm_tracks.csv.gz",
                f"{dashboard_data_dir}/dashboard_manifest.json",
            ),
            runtime="medium: validates and exports the Streamlit data contract",
        ),
    )


def display_stages(stages: Sequence[Stage]) -> None:
    print("DRY RUN: no commands will execute")
    for index, stage in enumerate(stages, start=1):
        print(f"\n{index}. {stage.name}")
        print(f"   command: {shlex.join(stage.command)}")
        print(f"   required inputs: {', '.join(stage.required_inputs)}")
        print(f"   expected outputs: {', '.join(stage.expected_outputs)}")
        print(f"   runtime: {stage.runtime}")


def execute_stages(stages: Sequence[Stage]) -> None:
    for stage in stages:
        missing = [
            path
            for path in stage.required_inputs
            if not (PROJECT_ROOT / path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{stage.name} is missing required inputs: {', '.join(missing)}"
            )
        print(f"EXECUTING {stage.name}: {shlex.join(stage.command)}")
        subprocess.run(stage.command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.toml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Display stages without running")
    mode.add_argument("--execute", action="store_true", help="Run every stage in order")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    stages = build_stages(load_config(config_path))
    if args.execute:
        execute_stages(stages)
    else:
        display_stages(stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
