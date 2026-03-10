#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.build_features import build_feature_rows, read_base_dataset_csv, write_feature_dataset_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-ready feature dataset from a normalized player-match CSV."
    )
    parser.add_argument("--input", required=True, help="Input normalized base CSV.")
    parser.add_argument("--output", required=True, help="Output feature CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = read_base_dataset_csv(args.input)
    feature_rows = build_feature_rows(base_rows)
    write_feature_dataset_csv(feature_rows, args.output)
    print(f"Wrote {len(feature_rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
