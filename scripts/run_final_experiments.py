#!/usr/bin/env python3
"""Run final experiments to find optimal feature set."""
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

# Best 18 features from refined experiments
BEST_18_FEATURES = [
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
    "bowling_style",
    "player_role",
]

# Best hyperparameters
BEST_PARAMS = {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10}

# More features to test
MORE_FEATURES_TO_TEST = [
    "rolling_wickets_avg_5_all",
    "rolling_strike_rate_5_all",
    "bowling_points_pct_5_all",
    "ema_points_5_all",
    "batting_hand",
    "batting_points_pct_5_all",
    "rolling_points_avg_3_all",
    "rolling_runs_avg_5_all",
    "ar_subtype",
    "ema_batting_points_5_all",
    "ema_bowling_points_5_all",
    "rolling_powerplay_balls_share_avg_5_all",
    "rolling_middle_balls_share_avg_5_all",
    "duck_rate_10_all",
    "selection_rate_5_all",
    "prior_matches_ipl",
    "rolling_points_std_5_all",
    "bowling_match_rate_5_all",
    "batting_match_rate_5_all",
]


@dataclass
class ExperimentResult:
    name: str
    rmse: float
    mae: float
    features: list[str]
    params: dict = field(default_factory=dict)
    importance: dict = field(default_factory=dict)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def split_by_date(df: pd.DataFrame, val_fraction: float = 0.2):
    df_sorted = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def prepare_features(df: pd.DataFrame, feature_columns: list[str]):
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
    return rmse, mae, importance, model


def run_experiment(train_df, val_df, features, params, name):
    available = [f for f in features if f in train_df.columns]
    train_X, cat_cols = prepare_features(train_df, available)
    val_X, _ = prepare_features(val_df, available)
    train_y = train_df[TARGET_COLUMN].astype(float).values
    val_y = val_df[TARGET_COLUMN].astype(float).values
    rmse, mae, importance, model = train_catboost(train_X, train_y, val_X, val_y, cat_cols, params)
    return ExperimentResult(name, rmse, mae, available, params, importance), model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="tmp/experiments_final")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    df = load_dataset(args.input)
    train_df, val_df = split_by_date(df)
    print(f"  Train rows: {len(train_df)}, Validation rows: {len(val_df)}")

    results = []

    # =========================================================================
    # Baseline: Best 18 features
    # =========================================================================
    print("\n" + "="*80)
    print("BASELINE: Best 18 Features")
    print("="*80)

    result, _ = run_experiment(train_df, val_df, BEST_18_FEATURES, BEST_PARAMS, "baseline_18")
    results.append(result)
    print(f"  RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")
    baseline_rmse = result.rmse

    # =========================================================================
    # Try adding features one by one
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Try Adding Features")
    print("="*80)

    improvements = []
    for feature in MORE_FEATURES_TO_TEST:
        if feature in train_df.columns and feature not in BEST_18_FEATURES:
            features = BEST_18_FEATURES + [feature]
            result, _ = run_experiment(train_df, val_df, features, BEST_PARAMS, f"add_{feature}")
            results.append(result)
            improvement = baseline_rmse - result.rmse
            improvements.append((feature, result.rmse, improvement))
            status = "✓" if improvement > 0 else " "
            print(f"  {status} +{feature}: RMSE={result.rmse:.4f} ({improvement:+.4f})")

    # Sort by improvement
    improvements.sort(key=lambda x: x[2], reverse=True)
    print("\n  Best improvements:")
    for feat, rmse, imp in improvements[:5]:
        print(f"    {feat}: {imp:+.4f}")

    # =========================================================================
    # Greedy feature addition
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Greedy Feature Addition")
    print("="*80)

    current_features = BEST_18_FEATURES.copy()
    current_rmse = baseline_rmse
    added_features = []

    for i in range(5):  # Try adding up to 5 more features
        best_feature = None
        best_rmse = current_rmse

        for feature in MORE_FEATURES_TO_TEST:
            if feature in train_df.columns and feature not in current_features:
                features = current_features + [feature]
                result, _ = run_experiment(train_df, val_df, features, BEST_PARAMS, f"greedy_add_{feature}")
                if result.rmse < best_rmse:
                    best_rmse = result.rmse
                    best_feature = feature

        if best_feature and best_rmse < current_rmse - 0.0001:  # At least 0.0001 improvement
            current_features.append(best_feature)
            current_rmse = best_rmse
            added_features.append(best_feature)
            print(f"  Round {i+1}: Added {best_feature} -> RMSE={current_rmse:.4f}")
        else:
            print(f"  Round {i+1}: No improvement, stopping")
            break

    if added_features:
        result, model = run_experiment(train_df, val_df, current_features, BEST_PARAMS, "greedy_best")
        results.append(result)

    # =========================================================================
    # Try different hyperparameters on best features
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Hyperparameter Refinement")
    print("="*80)

    best_features = current_features

    more_params = [
        {"iterations": 1500, "depth": 6, "learning_rate": 0.015, "l2_leaf_reg": 10},
        {"iterations": 2000, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 10},
        {"iterations": 1000, "depth": 7, "learning_rate": 0.02, "l2_leaf_reg": 10},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 15},
        {"iterations": 1000, "depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 5},
        {"iterations": 1200, "depth": 6, "learning_rate": 0.018, "l2_leaf_reg": 8},
    ]

    best_final_rmse = current_rmse
    best_final_params = BEST_PARAMS
    best_final_result = None

    for i, params in enumerate(more_params):
        result, model = run_experiment(train_df, val_df, best_features, params, f"hyperparam_v{i+1}")
        results.append(result)
        print(f"  {params}: RMSE={result.rmse:.4f}")
        if result.rmse < best_final_rmse:
            best_final_rmse = result.rmse
            best_final_params = params
            best_final_result = result

    # =========================================================================
    # Final model
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    sorted_results = sorted(results, key=lambda x: x.rmse)

    print(f"\n{'Rank':<5} {'Experiment':<40} {'Features':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*75)
    for i, r in enumerate(sorted_results[:15], 1):
        print(f"{i:<5} {r.name:<40} {len(r.features):<10} {r.rmse:<10.4f} {r.mae:<10.4f}")

    best = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best.name}")
    print(f"  RMSE: {best.rmse:.4f}")
    print(f"  MAE: {best.mae:.4f}")
    print(f"  Features: {len(best.features)}")
    print(f"  Params: {best.params}")
    print(f"\n  Feature importance:")
    sorted_imp = sorted(best.importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        print(f"    {feat}: {imp:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_config = {
        "name": best.name,
        "rmse": float(best.rmse),
        "mae": float(best.mae),
        "features": best.features,
        "params": best.params,
        "feature_importance": {k: float(v) for k, v in best.importance.items()},
    }
    (output_dir / "best_model_config.json").write_text(json.dumps(best_config, indent=2))

    # Also save the stable feature set
    stable_features = {
        "features": best.features,
        "params": best.params,
    }
    (output_dir / "stable_features.json").write_text(json.dumps(stable_features, indent=2))

    print(f"\nResults saved to {output_dir}")

    # Train final model and save
    print("\nTraining final model...")
    from catboost import CatBoostRegressor
    import joblib

    train_X, cat_cols = prepare_features(train_df, best.features)
    val_X, _ = prepare_features(val_df, best.features)
    train_y = train_df[TARGET_COLUMN].astype(float).values

    final_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
    }
    final_params.update(best.params)

    final_model = CatBoostRegressor(**final_params)
    final_model.fit(train_X, train_y, cat_features=cat_cols)

    joblib.dump(final_model, output_dir / "model.joblib")
    print(f"Final model saved to {output_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
