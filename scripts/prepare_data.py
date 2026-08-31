"""Prepare IBTrACS observations for storm-track forecasting."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from storm_tracks.data import load_ibtracs, prepare_tracks
from storm_tracks.features import add_motion_features, select_agency_tracks


def prepare_data(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_observations: int = 2,
) -> pd.DataFrame:
    """Load, filter, select positions, engineer features, and save tracks."""
    tracks = load_ibtracs(input_path)
    tracks = prepare_tracks(tracks, min_observations=min_observations)
    tracks = select_agency_tracks(tracks)
    tracks = tracks.loc[tracks["use_for_modeling"]].reset_index(drop=True)
    tracks = add_motion_features(tracks)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_pickle(destination)
    return tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw IBTrACS CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Processed pickle path")
    parser.add_argument("--min-observations", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tracks = prepare_data(
        args.input,
        args.output,
        min_observations=args.min_observations,
    )
    print(
        f"Saved {len(tracks):,} observations from {tracks['sid'].nunique():,} storms "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
