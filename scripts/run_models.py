"""Fit covariance inputs and evaluate persistence and Kalman forecasts."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from storm_tracks.data import split_by_storm, validate_disjoint_storm_splits
from storm_tracks.evaluation import (
    evaluate_kalman,
    evaluate_persistence,
    results_metadata,
    summarize_lead_time_errors,
)
from storm_tracks.models import HurricaneKalmanFilter, estimate_noise_covariances


def _output_paths(output_dir: Path, split: str, legacy_names: bool) -> dict[str, Path]:
    if legacy_names:
        kalman_name = (
            "sliding_results_final.pkl"
            if split == "test"
            else "val_sliding_results_final.pkl"
        )
        persistence_name = (
            "null_model_test_results.pkl"
            if split == "test"
            else "null_model_val_results.pkl"
        )
    else:
        kalman_name = f"{split}_kalman_results.pkl"
        persistence_name = f"{split}_persistence_results.pkl"
    return {
        "kalman_results": output_dir / kalman_name,
        "persistence_results": output_dir / persistence_name,
        "kalman_summary": output_dir / f"{split}_kalman_summary.csv",
        "persistence_summary": output_dir / f"{split}_persistence_summary.csv",
        "metadata": output_dir / f"{split}_results_metadata.json",
    }


def run_models(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    lead_times_hours: Sequence[int] = (6, 12, 24, 48, 72),
    min_origin_index: int = 5,
    max_origins_per_storm: int = 5,
    covariance_min_storm_length: int = 5,
    covariance_sample_size: int | None = None,
    legacy_names: bool = False,
) -> dict[str, Path]:
    """Fit on training storms and evaluate one explicitly named split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")

    tracks = pd.read_pickle(Path(input_path))
    storm_split = split_by_storm(
        tracks,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    validate_disjoint_storm_splits(
        train=storm_split.train,
        validation=storm_split.validation,
        test=storm_split.test,
    )
    evaluation_tracks = getattr(storm_split, split)
    if evaluation_tracks.empty:
        raise ValueError(f"the configured {split!r} split is empty")

    process_covariance, observation_covariance = estimate_noise_covariances(
        storm_split.train,
        min_storm_length=covariance_min_storm_length,
        sample_size=covariance_sample_size,
        seed=seed,
    )
    model = HurricaneKalmanFilter(
        q_base=process_covariance,
        r_base=observation_covariance,
    )
    storm_ids = evaluation_tracks["sid"].dropna().unique()
    evaluation_options = {
        "split": split,
        "lead_times_hours": lead_times_hours,
        "min_origin_index": min_origin_index,
        "max_origins_per_storm": max_origins_per_storm,
    }
    persistence = evaluate_persistence(
        evaluation_tracks,
        storm_ids,
        **evaluation_options,
    )
    kalman = evaluate_kalman(
        evaluation_tracks,
        storm_ids,
        model,
        **evaluation_options,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(destination, split, legacy_names)
    persistence.to_pickle(outputs["persistence_results"])
    kalman.to_pickle(outputs["kalman_results"])
    summarize_lead_time_errors(persistence, split=split).reset_index().to_csv(
        outputs["persistence_summary"], index=False
    )
    summarize_lead_time_errors(kalman, split=split).reset_index().to_csv(
        outputs["kalman_summary"], index=False
    )

    metadata: dict[str, object] = results_metadata(split=split)
    metadata.update(
        {
            "seed": seed,
            "training_storms": int(storm_split.train["sid"].nunique()),
            "evaluation_storms": int(evaluation_tracks["sid"].nunique()),
            "lead_times_hours": list(lead_times_hours),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
    )
    outputs["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--lead-times", nargs="+", type=int, default=[6, 12, 24, 48, 72])
    parser.add_argument("--min-origin-index", type=int, default=5)
    parser.add_argument("--max-origins", type=int, default=5)
    parser.add_argument("--covariance-min-storm-length", type=int, default=5)
    parser.add_argument("--covariance-sample-size", type=int)
    parser.add_argument("--legacy-compatible-names", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_models(
        args.input,
        args.output_dir,
        split=args.split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        lead_times_hours=args.lead_times,
        min_origin_index=args.min_origin_index,
        max_origins_per_storm=args.max_origins,
        covariance_min_storm_length=args.covariance_min_storm_length,
        covariance_sample_size=args.covariance_sample_size,
        legacy_names=args.legacy_compatible_names,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
