#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.normalize_cricket_json import (
    normalize_dataset_dir,
    write_training_dataset_csv,
    load_player_roles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized player-match training dataset from cricket JSON folders.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Path to a dataset directory containing Cricsheet-style JSON files. Repeat for multiple datasets.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset file limit for sampling.")
    parser.add_argument(
        "--gender",
        dest="genders",
        action="append",
        default=None,
        help="Optional gender filter. Repeat for multiple values, e.g. --gender male.",
    )
    parser.add_argument(
        "--player-roles",
        default="data/player_roles_detailed.csv",
        help="Path to player roles CSV (default: data/player_roles_detailed.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player_roles = load_player_roles(args.player_roles)
    print(f"Loaded {len(player_roles)} player roles")

    all_rows = []
    for dataset in args.datasets:
        gender_filter = set(args.genders) if args.genders else None
        rows = normalize_dataset_dir(
            Path(dataset),
            limit=args.limit,
            gender_filter=gender_filter,
            player_roles=player_roles,
        )
        all_rows.extend(rows)
        print(f"  {dataset}: {len(rows)} rows")

    write_training_dataset_csv(all_rows, Path(args.output))
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
