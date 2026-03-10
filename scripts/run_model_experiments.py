#!/usr/bin/env python3
"""Run comprehensive model experiments with feature selection."""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PerformanceWarning.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.options.mode.chained_assignment = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_COLUMN = "dream11_points_total"

# Columns that should never be used as features (leak target or are identifiers)
EXCLUDED_COLUMNS = {
    "match_id",
    "match_date",
    TARGET_COLUMN,
    "winner",
    # Current match stats (would leak target)
    "runs",
    "batting_position",
    "batting_order_bucket",
    "balls_faced",
    "batting_balls_share",
    "fours",
    "sixes",
    "duck",
    "balls_bowled",
    "overs_bowled",
    "bowling_balls_share",
    "powerplay_balls",
    "middle_balls",
    "death_balls",
    "powerplay_balls_share",
    "middle_balls_share",
    "death_balls_share",
    "maidens",
    "runs_conceded",
    "wickets",
    "catches",
    "stumpings",
    "run_out_direct",
    "run_out_assist",
    "batting_points",
    "bowling_points",
    "fielding_points",
    "other_points",
}

# Feature groups for systematic feature selection
FEATURE_GROUPS = {
    "context": [
        "competition", "source_dataset", "season", "team_type", "match_type",
        "venue", "city", "team", "opponent", "toss_winner", "toss_decision",
    ],
    "player_info": [
        "player_name", "player_role", "batting_hand", "bowling_arm",
        "bowling_style", "ar_subtype", "playing_xi",
    ],
    "experience": [
        "prior_matches_all", "prior_matches_ipl", "prior_matches_recent_t20",
    ],
    "rolling_points_basic": [
        "rolling_points_avg_3_all", "rolling_points_avg_5_all",
        "rolling_points_avg_10_all", "rolling_points_std_5_all",
    ],
    "rolling_points_ipl": [
        "rolling_points_avg_3_ipl", "rolling_points_avg_5_ipl",
    ],
    "rolling_points_t20": [
        "rolling_points_avg_3_recent_t20", "rolling_points_avg_5_recent_t20",
    ],
    "rolling_batting": [
        "rolling_runs_avg_3_all", "rolling_runs_avg_5_all",
        "rolling_batting_points_avg_5_all", "rolling_balls_faced_avg_5_all",
        "rolling_strike_rate_5_all", "batting_match_rate_5_all",
    ],
    "rolling_bowling": [
        "rolling_wickets_avg_3_all", "rolling_wickets_avg_5_all",
        "rolling_bowling_points_avg_5_all", "rolling_balls_bowled_avg_5_all",
        "rolling_economy_rate_5_all", "bowling_match_rate_5_all",
    ],
    "rolling_fielding": [
        "rolling_fielding_points_avg_5_all",
    ],
    "ema_features": [
        "ema_points_5_all", "ema_points_10_all", "ema_runs_5_all",
        "ema_wickets_5_all", "ema_batting_points_5_all", "ema_bowling_points_5_all",
    ],
    "ceiling_volatility": [
        "rolling_points_p75_10_all", "rolling_points_p90_10_all",
        "rolling_points_max_10_all", "rolling_points_min_10_all",
    ],
    "selection": [
        "selection_rate_10_all", "selection_rate_5_all",
    ],
    "contribution_mix": [
        "batting_points_pct_5_all", "bowling_points_pct_5_all",
        "fielding_points_pct_5_all",
    ],
    "position_usage": [
        "batting_position_known_rate_5_all", "rolling_batting_position_avg_5_all",
        "rolling_batting_balls_share_avg_5_all", "rolling_bowling_balls_share_avg_5_all",
        "rolling_powerplay_balls_share_avg_5_all", "rolling_middle_balls_share_avg_5_all",
        "rolling_death_balls_share_avg_5_all",
    ],
    "stability": [
        "batting_position_std_5_all", "bowling_balls_share_std_5_all",
    ],
    "tendency": [
        "duck_rate_10_all", "boundary_rate_5_all",
    ],
    "venue_opponent": [
        "prior_matches_at_venue", "venue_points_avg_all",
        "prior_matches_vs_opponent", "opponent_points_avg_all",
    ],
    "trend": [
        "points_trend_3_vs_10_all",
    ],
}


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    name: str
    model_name: str
    rmse: float
    mae: float
    features_used: list[str]
    feature_count: int
    train_rows: int
    val_rows: int
    feature_importance: dict[str, float] = field(default_factory=dict)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load feature dataset as DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    return df


def split_by_date(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically by date."""
    df_sorted = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - val_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare feature matrix with proper type handling."""
    X = df[feature_columns].copy()
    categorical_cols = []
    numeric_cols = []

    for col in feature_columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            numeric_cols.append(col)
        else:
            # Try to convert to numeric
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().sum() == X[col].replace("", pd.NA).notna().sum():
                X[col] = converted.fillna(0.0)
                numeric_cols.append(col)
            else:
                X[col] = X[col].fillna("").astype(str)
                categorical_cols.append(col)

    return X, categorical_cols, numeric_cols


def build_model(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Build a model with optional custom parameters."""
    default_params = {
        "catboost": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": 42,
            "verbose": False,
        },
        "xgboost": {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "lightgbm": {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        },
    }

    model_params = default_params.get(model_name, {})
    if params:
        model_params.update(params)

    if model_name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**model_params)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**model_params)
    elif model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    experiment_name: str,
    model_params: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Train a model and evaluate on validation set."""

    # Prepare features
    train_X, cat_cols, num_cols = prepare_features(train_df, feature_columns)
    val_X, _, _ = prepare_features(val_df, feature_columns)

    train_y = train_df[TARGET_COLUMN].astype(float).values
    val_y = val_df[TARGET_COLUMN].astype(float).values

    # Build and train model
    model = build_model(model_name, model_params)

    if model_name == "catboost":
        model.fit(train_X, train_y, cat_features=cat_cols)
        predictions = model.predict(val_X)
        # Get feature importance
        importance = dict(zip(feature_columns, model.get_feature_importance()))
    else:
        # One-hot encode categorical features for XGBoost/LightGBM
        if cat_cols:
            train_X_encoded = pd.get_dummies(train_X, columns=cat_cols, dummy_na=False)
            val_X_encoded = pd.get_dummies(val_X, columns=cat_cols, dummy_na=False)
            # Align columns efficiently
            missing_cols = set(train_X_encoded.columns) - set(val_X_encoded.columns)
            if missing_cols:
                missing_df = pd.DataFrame(0, index=val_X_encoded.index, columns=list(missing_cols))
                val_X_encoded = pd.concat([val_X_encoded, missing_df], axis=1)
            val_X_encoded = val_X_encoded.reindex(columns=train_X_encoded.columns, fill_value=0)
            train_X_encoded = train_X_encoded.astype(float)
            val_X_encoded = val_X_encoded.astype(float)
        else:
            train_X_encoded = train_X.astype(float)
            val_X_encoded = val_X.astype(float)

        # Sanitize column names for LightGBM (remove special chars and ensure uniqueness)
        if model_name == "lightgbm":
            import re
            clean_cols = []
            seen = {}
            for c in train_X_encoded.columns:
                clean_name = re.sub(r'[^\w]', '_', str(c))
                if clean_name in seen:
                    seen[clean_name] += 1
                    clean_name = f"{clean_name}_{seen[clean_name]}"
                else:
                    seen[clean_name] = 0
                clean_cols.append(clean_name)
            train_X_encoded.columns = clean_cols
            val_X_encoded.columns = clean_cols

        model.fit(train_X_encoded, train_y)
        predictions = model.predict(val_X_encoded)

        # Get feature importance (for original features, not one-hot)
        if hasattr(model, "feature_importances_"):
            encoded_importance = dict(zip(train_X_encoded.columns, model.feature_importances_))
            # Aggregate importance for categorical features
            importance = {}
            for col in feature_columns:
                if col in cat_cols:
                    # Sum importance of all one-hot columns
                    related_cols = [c for c in encoded_importance if c.startswith(col + "_")]
                    importance[col] = sum(encoded_importance.get(c, 0) for c in related_cols)
                else:
                    importance[col] = encoded_importance.get(col, 0)
        else:
            importance = {}

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(val_y, predictions))
    mae = mean_absolute_error(val_y, predictions)

    return ExperimentResult(
        name=experiment_name,
        model_name=model_name,
        rmse=rmse,
        mae=mae,
        features_used=feature_columns,
        feature_count=len(feature_columns),
        train_rows=len(train_df),
        val_rows=len(val_df),
        feature_importance=importance,
    )


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Get all available feature columns from dataset."""
    return [col for col in df.columns if col not in EXCLUDED_COLUMNS]


def get_features_by_groups(groups: list[str], available: list[str]) -> list[str]:
    """Get features from specified groups that are available in dataset."""
    features = []
    for group in groups:
        if group in FEATURE_GROUPS:
            for feat in FEATURE_GROUPS[group]:
                if feat in available and feat not in features:
                    features.append(feat)
    return features


def run_feature_selection_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    available_features: list[str],
) -> list[ExperimentResult]:
    """Run experiments with different feature combinations."""
    results = []

    # Experiment 1: All features
    print(f"\n  Running: {model_name} - All features ({len(available_features)})")
    result = train_and_evaluate(
        train_df, val_df, available_features, model_name,
        f"{model_name}_all_features"
    )
    results.append(result)
    print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment 2: Core features only (no venue/opponent specifics)
    core_groups = [
        "player_info", "experience", "rolling_points_basic",
        "rolling_batting", "rolling_bowling", "rolling_fielding",
        "ema_features", "ceiling_volatility", "contribution_mix",
        "trend", "tendency",
    ]
    core_features = get_features_by_groups(core_groups, available_features)
    if core_features:
        print(f"\n  Running: {model_name} - Core features ({len(core_features)})")
        result = train_and_evaluate(
            train_df, val_df, core_features, model_name,
            f"{model_name}_core_features"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment 3: Minimal (player info + points rolling only)
    minimal_groups = ["player_info", "experience", "rolling_points_basic"]
    minimal_features = get_features_by_groups(minimal_groups, available_features)
    if minimal_features:
        print(f"\n  Running: {model_name} - Minimal features ({len(minimal_features)})")
        result = train_and_evaluate(
            train_df, val_df, minimal_features, model_name,
            f"{model_name}_minimal_features"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment 4: Points-focused (all points-related features)
    points_groups = [
        "player_info", "experience", "rolling_points_basic", "rolling_points_ipl",
        "rolling_points_t20", "ema_features", "ceiling_volatility", "trend",
    ]
    points_features = get_features_by_groups(points_groups, available_features)
    if points_features:
        print(f"\n  Running: {model_name} - Points-focused ({len(points_features)})")
        result = train_and_evaluate(
            train_df, val_df, points_features, model_name,
            f"{model_name}_points_focused"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment 5: No categorical (numeric only for faster training)
    numeric_features = [f for f in available_features
                       if f not in ["player_name", "team", "opponent", "venue",
                                   "city", "competition", "toss_winner", "source_dataset"]]
    if numeric_features:
        print(f"\n  Running: {model_name} - Numeric only ({len(numeric_features)})")
        result = train_and_evaluate(
            train_df, val_df, numeric_features, model_name,
            f"{model_name}_numeric_only"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    return results


def run_importance_based_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    available_features: list[str],
    base_result: ExperimentResult,
) -> list[ExperimentResult]:
    """Run experiments using feature importance for selection."""
    results = []

    if not base_result.feature_importance:
        return results

    # Sort features by importance
    sorted_features = sorted(
        base_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Experiment: Top 50% features by importance
    top_50_count = max(5, len(sorted_features) // 2)
    top_50_features = [f for f, _ in sorted_features[:top_50_count]]
    top_50_features = [f for f in top_50_features if f in available_features]

    if top_50_features:
        print(f"\n  Running: {model_name} - Top 50% by importance ({len(top_50_features)})")
        result = train_and_evaluate(
            train_df, val_df, top_50_features, model_name,
            f"{model_name}_top50_importance"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment: Top 25% features by importance
    top_25_count = max(5, len(sorted_features) // 4)
    top_25_features = [f for f, _ in sorted_features[:top_25_count]]
    top_25_features = [f for f in top_25_features if f in available_features]

    if top_25_features:
        print(f"\n  Running: {model_name} - Top 25% by importance ({len(top_25_features)})")
        result = train_and_evaluate(
            train_df, val_df, top_25_features, model_name,
            f"{model_name}_top25_importance"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    # Experiment: Remove low-importance features (bottom 25%)
    cutoff = len(sorted_features) - len(sorted_features) // 4
    pruned_features = [f for f, _ in sorted_features[:cutoff]]
    pruned_features = [f for f in pruned_features if f in available_features]

    if pruned_features and len(pruned_features) < len(available_features):
        print(f"\n  Running: {model_name} - Pruned bottom 25% ({len(pruned_features)})")
        result = train_and_evaluate(
            train_df, val_df, pruned_features, model_name,
            f"{model_name}_pruned_bottom25"
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    return results


def run_hyperparameter_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    best_features: list[str],
    model_name: str,
) -> list[ExperimentResult]:
    """Run experiments with different hyperparameters."""
    results = []

    param_grids = {
        "catboost": [
            {"iterations": 800, "depth": 8, "learning_rate": 0.03},
            {"iterations": 1000, "depth": 6, "learning_rate": 0.02},
            {"iterations": 500, "depth": 10, "learning_rate": 0.05},
        ],
        "xgboost": [
            {"n_estimators": 800, "max_depth": 8, "learning_rate": 0.03},
            {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02},
            {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.05},
        ],
        "lightgbm": [
            {"n_estimators": 800, "max_depth": 8, "learning_rate": 0.03},
            {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02},
            {"n_estimators": 500, "max_depth": 12, "learning_rate": 0.05, "num_leaves": 64},
        ],
    }

    for i, params in enumerate(param_grids.get(model_name, [])):
        print(f"\n  Running: {model_name} - Hyperparams variant {i+1}")
        print(f"    Params: {params}")
        result = train_and_evaluate(
            train_df, val_df, best_features, model_name,
            f"{model_name}_hyperparam_v{i+1}",
            model_params=params,
        )
        results.append(result)
        print(f"    RMSE: {result.rmse:.4f}, MAE: {result.mae:.4f}")

    return results


def print_summary(results: list[ExperimentResult]) -> ExperimentResult:
    """Print summary of all experiments and return best result."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Sort by RMSE
    sorted_results = sorted(results, key=lambda x: x.rmse)

    print(f"\n{'Rank':<5} {'Experiment':<40} {'Model':<10} {'Features':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*85)

    for i, r in enumerate(sorted_results[:20], 1):
        print(f"{i:<5} {r.name:<40} {r.model_name:<10} {r.feature_count:<10} {r.rmse:<10.4f} {r.mae:<10.4f}")

    best = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best.name}")
    print(f"  Model: {best.model_name}")
    print(f"  RMSE: {best.rmse:.4f}")
    print(f"  MAE: {best.mae:.4f}")
    print(f"  Features: {best.feature_count}")

    # Print top features by importance
    if best.feature_importance:
        print(f"\n  Top 15 features by importance:")
        sorted_imp = sorted(best.feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        for feat, imp in sorted_imp:
            print(f"    {feat}: {imp:.4f}")

    return best


def save_results(results: list[ExperimentResult], best: ExperimentResult, output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save all results as JSON
    results_data = [
        {
            "name": r.name,
            "model_name": r.model_name,
            "rmse": float(r.rmse),
            "mae": float(r.mae),
            "feature_count": int(r.feature_count),
            "features": r.features_used,
            "train_rows": int(r.train_rows),
            "val_rows": int(r.val_rows),
        }
        for r in results
    ]
    (output_dir / "all_experiments.json").write_text(json.dumps(results_data, indent=2))

    # Save best model config
    best_config = {
        "name": best.name,
        "model_name": best.model_name,
        "rmse": float(best.rmse),
        "mae": float(best.mae),
        "feature_count": int(best.feature_count),
        "features": best.features_used,
        "feature_importance": {k: to_native(v) for k, v in best.feature_importance.items()},
    }
    (output_dir / "best_model_config.json").write_text(json.dumps(best_config, indent=2))

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run model experiments with feature selection")
    parser.add_argument("--input", required=True, help="Input feature CSV")
    parser.add_argument("--output-dir", default="tmp/experiments", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["catboost", "xgboost", "lightgbm"],
                       help="Models to experiment with")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--skip-hyperparams", action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    df = load_dataset(args.input)
    print(f"  Total rows: {len(df)}")

    # Split data
    train_df, val_df = split_by_date(df, args.val_fraction)
    print(f"  Train rows: {len(train_df)}")
    print(f"  Validation rows: {len(val_df)}")

    # Get available features
    available_features = get_available_features(df)
    print(f"  Available features: {len(available_features)}")

    all_results = []
    best_per_model = {}

    # Run experiments for each model
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTING WITH: {model_name.upper()}")
        print("="*80)

        # Phase 1: Feature selection experiments
        print("\nPhase 1: Feature Selection")
        results = run_feature_selection_experiments(
            train_df, val_df, model_name, available_features
        )
        all_results.extend(results)

        # Get best result from this model so far
        best_so_far = min(results, key=lambda x: x.rmse)

        # Phase 2: Importance-based feature selection
        print("\nPhase 2: Importance-based Selection")
        importance_results = run_importance_based_selection(
            train_df, val_df, model_name, available_features, best_so_far
        )
        all_results.extend(importance_results)

        # Update best if improved
        if importance_results:
            best_importance = min(importance_results, key=lambda x: x.rmse)
            if best_importance.rmse < best_so_far.rmse:
                best_so_far = best_importance

        # Phase 3: Hyperparameter tuning on best features
        if not args.skip_hyperparams:
            print("\nPhase 3: Hyperparameter Tuning")
            hyperparam_results = run_hyperparameter_experiments(
                train_df, val_df, best_so_far.features_used, model_name
            )
            all_results.extend(hyperparam_results)

            if hyperparam_results:
                best_hyperparam = min(hyperparam_results, key=lambda x: x.rmse)
                if best_hyperparam.rmse < best_so_far.rmse:
                    best_so_far = best_hyperparam

        best_per_model[model_name] = best_so_far
        print(f"\nBest {model_name}: RMSE={best_so_far.rmse:.4f}, MAE={best_so_far.mae:.4f}")

    # Print final summary
    best_overall = print_summary(all_results)

    # Save results
    save_results(all_results, best_overall, Path(args.output_dir))

    return best_overall


if __name__ == "__main__":
    main()
