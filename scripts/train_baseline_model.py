#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.train_baseline_model import (
    DEFAULT_MODEL_NAME,
    SUPPORTED_MODELS,
    load_feature_rows,
    save_training_artifacts,
    split_rows_by_date,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline Dream11 fantasy points model.")
    parser.add_argument("--input", required=True, help="Input feature CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write model and metrics artifacts.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of most recent rows to reserve for validation.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default=DEFAULT_MODEL_NAME,
        help="Model family to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_feature_rows(args.input)
    train_rows, validation_rows = split_rows_by_date(rows, validation_fraction=args.validation_fraction)
    result = train_and_evaluate(train_rows, validation_rows, model_name=args.model)
    save_training_artifacts(result, args.output_dir)
    print(
        f"Trained {result['model_name']} model on {result['train_row_count']} rows, "
        f"validated on {result['validation_row_count']} rows, "
        f"RMSE={result['metrics']['rmse']:.4f}, MAE={result['metrics']['mae']:.4f}"
    )


if __name__ == "__main__":
    main()
