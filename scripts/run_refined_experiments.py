#!/usr/bin/env python3
"""Run refined model experiments with best features."""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pd.options.mode.chained_assignment = None

TARGET_COLUMN = "dream11_points_total"

# Best 16 features from initial experiments
BEST_16_FEATURES = [
    "rolling_batting_position_avg_5_all",
    "rolling_balls_bowled_avg_5_all",
    "venue_points_avg_all",
    "rolling_points_avg_10_all",
    "rolling_death_balls_share_avg_5_all",
    "prior_matches_all",
    "rolling_bowling_balls_share_avg_5_all",
    "opponent_points_avg_all",
    "rolling_balls_faced_avg_5_all",
    "rolling_economy_rate_5_all",
    "prior_matches_recent_t20",
    "rolling_points_avg_5_ipl",
    "boundary_rate_5_all",
    "rolling_points_p75_10_all",
    "rolling_points_avg_5_recent_t20",
    "rolling_points_p90_10_all",
]

# Additional features to test adding
ADDITIONAL_FEATURES_TO_TEST = [
    "player_role",
    "ema_points_5_all",
    "ema_points_10_all",
    "rolling_points_std_5_all",
    "rolling_strike_rate_5_all",
    "rolling_wickets_avg_5_all",
    "rolling_runs_avg_5_all",
    "batting_points_pct_5_all",
    "bowling_points_pct_5_all",
    "points_trend_3_vs_10_all",
    "prior_matches_ipl",
    "selection_rate_5_all",
    "rolling_points_avg_3_all",
    "rolling_batting_balls_share_avg_5_all",
    "batting_hand",
    "bowling_style",
]


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    name: str
    model_name: str
    rmse: float
    mae: float
    features_used: list[str]
    feature_count: int
    params: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def split_by_date(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df[feature_columns].copy()
    categorical_cols = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        else:
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().sum() == X[col].replace("", pd.NA).notna().sum():
                X[col] = converted.fillna(0.0)
            else:
                X[col] = X[col].fillna("").astype(str)
                categorical_cols.append(col)

    return X, categorical_cols


def train_catboost(train_X, train_y, val_X, val_y, cat_cols, params):
    from catboost import CatBoostRegressor

    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False,
    }
    default_params.update(params)

    model = CatBoostRegressor(**default_params)
    model.fit(train_X, train_y, cat_features=cat_cols)
    predictions = model.predict(val_X)

    rmse = math.sqrt(mean_squared_error(val_y, predictions))
    mae = mean_absolute_error(val_y, predictions)
    importance = dict(zip(train_X.columns, model.get_feature_importance()))

    return rmse, mae, importance


def run_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    name: str,
) -> ExperimentResult:
    """Run a single experiment."""
    # Filter to available features
    available = [f for f in features if f in train_df.columns]

    train_X, cat_cols = prepare_features(train_df, available)
    val_X, _ = prepare_features(val_df, available)
    train_y = train_df[TARGET_COLUMN].astype(float).values
    val_y = val_df[TARGET_COLUMN].astype(float).values

    rmse, mae, importance = train_catboost(train_X, train_y, val_X, val_y, cat_cols, params)

    return ExperimentResult(
        name=name,
        model_name="catboost",
        rmse=rmse,
        mae=mae,
        features_used=available,
        feature_count=len(available),
        params=params,
        feature_importance=importance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input feature CSV")
    parser.add_argument("--output-dir", default="tmp/experiments_v2", help="Output directory")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    df = load_dataset(args.input)
    train_df, val_df = split_by_date(df)
    print(f"  Train rows: {len(train_df)}, Validation rows: {len(val_df)}")

    results = []

    # =========================================================================
    # Phase 1: Hyperparameter grid search on best 16 features
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Hyperparameter Grid Search on Best 16 Features")
    print("="*80)

    param_grid = [
        {"iterations": 500, "depth": 6, "learning_rate": 0.05},
        {"iterations": 800, "depth": 6, "learning_rate": 0.03},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02},
        {"iterations": 1200, "depth": 6, "learning_rate": 0.015},
        {"iterations": 1500, "depth": 6, "learning_rate": 0.01},
        {"iterations": 800, "depth": 8, "learning_rate": 0.03},
        {"iterations": 1000, "depth": 8, "learning_rate": 0.02},
        {"iterations": 800, "depth": 4, "learning_rate": 0.03},
        {"iterations": 1000, "depth": 4, "learning_rate": 0.02},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 3},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 5},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10},
    ]

    best_rmse = float('inf')
    best_params = None

    for i, params in enumerate(param_grid):
        print(f"\n  [{i+1}/{len(param_grid)}] Testing: {params}")
        result = run_experiment(train_df, val_df, BEST_16_FEATURES, params, f"hyperparam_v{i+1}")
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")
        if result.rmse < best_rmse:
            best_rmse = result.rmse
            best_params = params

    print(f"\nBest hyperparameters: {best_params} (RMSE: {best_rmse:.4f})")

    # =========================================================================
    # Phase 2: Feature addition experiments
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Feature Addition Experiments")
    print("="*80)

    for feature in ADDITIONAL_FEATURES_TO_TEST:
        if feature in train_df.columns and feature not in BEST_16_FEATURES:
            features = BEST_16_FEATURES + [feature]
            print(f"\n  Testing: +{feature} ({len(features)} features)")
            result = run_experiment(train_df, val_df, features, best_params, f"add_{feature}")
            results.append(result)
            print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # =========================================================================
    # Phase 3: Feature removal experiments
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Feature Removal Experiments")
    print("="*80)

    # Try removing each feature one at a time
    for feature in BEST_16_FEATURES:
        features = [f for f in BEST_16_FEATURES if f != feature]
        print(f"\n  Testing: -{feature} ({len(features)} features)")
        result = run_experiment(train_df, val_df, features, best_params, f"remove_{feature}")
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # =========================================================================
    # Phase 4: Best combinations
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Best Feature Combinations")
    print("="*80)

    # Find best additions
    addition_results = [r for r in results if r.name.startswith("add_")]
    if addition_results:
        best_additions = sorted(addition_results, key=lambda x: x.rmse)[:3]
        for r in best_additions:
            print(f"  Best addition: {r.name} -> RMSE: {r.rmse:.4f}")

    # Find any removals that improved RMSE
    removal_results = [r for r in results if r.name.startswith("remove_")]
    improved_removals = [r for r in removal_results if r.rmse < best_rmse]
    for r in improved_removals:
        print(f"  Improved by removing: {r.name} -> RMSE: {r.rmse:.4f}")

    # Try combining best additions
    if len(best_additions) >= 2:
        added_features = [r.name.replace("add_", "") for r in best_additions[:2]]
        features = BEST_16_FEATURES + added_features
        print(f"\n  Testing: +{added_features} ({len(features)} features)")
        result = run_experiment(train_df, val_df, features, best_params, "combined_additions")
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x.rmse)

    print(f"\n{'Rank':<5} {'Experiment':<45} {'Features':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*80)
    for i, r in enumerate(sorted_results[:20], 1):
        print(f"{i:<5} {r.name:<45} {r.feature_count:<10} {r.rmse:<10.4f} {r.mae:<10.4f}")

    best = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best.name}")
    print(f"  RMSE: {best.rmse:.4f}")
    print(f"  MAE: {best.mae:.4f}")
    print(f"  Features: {best.feature_count}")
    print(f"  Params: {best.params}")
    print(f"\n  Feature list:")
    for f in best.features_used:
        imp = best.feature_importance.get(f, 0)
        print(f"    {f}: {imp:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = [
        {
            "name": r.name,
            "rmse": float(r.rmse),
            "mae": float(r.mae),
            "feature_count": r.feature_count,
            "features": r.features_used,
            "params": r.params,
        }
        for r in sorted_results
    ]
    (output_dir / "all_experiments.json").write_text(json.dumps(results_data, indent=2))

    best_config = {
        "name": best.name,
        "rmse": float(best.rmse),
        "mae": float(best.mae),
        "features": best.features_used,
        "params": best.params,
        "feature_importance": {k: float(v) for k, v in best.feature_importance.items()},
    }
    (output_dir / "best_model_config.json").write_text(json.dumps(best_config, indent=2))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
