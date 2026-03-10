#!/usr/bin/env python3
"""Add won_toss feature to the features dataset permanently."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_toss_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add won_toss feature."""
    df = df.copy()

    def get_toss_advantage(row):
        if pd.isna(row['toss_winner']):
            return 0.5
        return 1.0 if row['team'] == row['toss_winner'] else 0.0

    df['won_toss'] = df.apply(get_toss_advantage, axis=1)
    return df


def main():
    input_path = "tmp/full_player_match_features_v3.csv"
    output_path = "tmp/full_player_match_features_v4.csv"

    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Rows: {len(df)}")

    print("Adding won_toss feature...")
    df = add_toss_feature(df)
    print(f"  won_toss mean: {df['won_toss'].mean():.2f}")

    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
