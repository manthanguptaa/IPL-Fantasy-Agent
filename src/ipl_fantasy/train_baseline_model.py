from __future__ import annotations

import json
import math
import importlib
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.ipl_fantasy.build_features import read_base_dataset_csv


TARGET_COLUMN = "dream11_points_total"
DEFAULT_MODEL_NAME = "catboost"
SUPPORTED_MODELS = {"catboost", "xgboost", "lightgbm"}

# Optimal feature set found through extensive experimentation
# These 21 features achieve RMSE=31.64, MAE=24.19 on validation
OPTIMAL_FEATURES = [
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
    "rolling_strike_rate_5_all",
    "ema_bowling_points_5_all",
    "prior_matches_ipl",
]

# Optimal hyperparameters for CatBoost
OPTIMAL_CATBOOST_PARAMS = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.02,
    "l2_leaf_reg": 10,
}

EXCLUDED_COLUMNS = {
    "match_id",
    "match_date",
    TARGET_COLUMN,
    "winner",
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


def split_rows_by_date(rows: list[dict[str, str]], validation_fraction: float = 0.2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sorted_rows = sorted(rows, key=lambda row: (row["match_date"], row["match_id"], row["player_name"]))
    split_idx = max(1, min(len(sorted_rows) - 1, int(len(sorted_rows) * (1 - validation_fraction))))
    return sorted_rows[:split_idx], sorted_rows[split_idx:]


def _is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() == series.replace("", pd.NA).notna().sum()


def prepare_training_matrices(rows: list[dict[str, str]], use_optimal_features: bool = True) -> tuple[pd.DataFrame, list[float], list[str]]:
    if use_optimal_features:
        # Use only the optimal feature set
        feature_columns = [col for col in OPTIMAL_FEATURES if col in rows[0].keys()]
    else:
        feature_columns = [column for column in rows[0].keys() if column not in EXCLUDED_COLUMNS]
    X = pd.DataFrame([{column: row.get(column, "") for column in feature_columns} for row in rows])
    for column in feature_columns:
        if _is_numeric_series(X[column]):
            X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0.0)
        else:
            X[column] = X[column].fillna("").astype(str)
    y = [float(row[TARGET_COLUMN]) for row in rows]
    return X, y, feature_columns


def _build_model(model_name: str, use_optimal: bool = True) -> Any:
    if model_name == "catboost":
        if use_optimal:
            return CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=OPTIMAL_CATBOOST_PARAMS["iterations"],
                learning_rate=OPTIMAL_CATBOOST_PARAMS["learning_rate"],
                depth=OPTIMAL_CATBOOST_PARAMS["depth"],
                l2_leaf_reg=OPTIMAL_CATBOOST_PARAMS["l2_leaf_reg"],
                random_seed=42,
                verbose=False,
            )
        return CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False,
        )
    if model_name == "xgboost":
        xgboost = importlib.import_module("xgboost")
        return xgboost.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    if model_name == "lightgbm":
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Expected one of {sorted(SUPPORTED_MODELS)}.")


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if pd.api.types.is_object_dtype(frame[column])]


def _align_frame_columns(frame: pd.DataFrame, encoded_feature_columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=encoded_feature_columns, fill_value=0.0)
    return aligned.astype(float)


def _sanitize_encoded_columns(frame: pd.DataFrame) -> pd.DataFrame:
    used_counts: dict[str, int] = {}
    renamed_columns: list[str] = []
    for column in frame.columns:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(column)).strip("_") or "feature"
        next_count = used_counts.get(sanitized, 0)
        used_counts[sanitized] = next_count + 1
        if next_count:
            sanitized = f"{sanitized}_{next_count}"
        renamed_columns.append(sanitized)
    result = frame.copy()
    result.columns = renamed_columns
    return result


def _fit_and_predict(
    model: Any,
    model_name: str,
    train_X: pd.DataFrame,
    train_y: list[float],
    val_X: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[list[float], list[str]]:
    if model_name == "catboost":
        model.fit(train_X, train_y, cat_features=categorical_columns)
        return model.predict(val_X), list(train_X.columns)

    encoded_train_X = pd.get_dummies(train_X, columns=categorical_columns, dummy_na=False)
    encoded_train_X = _sanitize_encoded_columns(encoded_train_X).astype(float)
    encoded_val_X = pd.get_dummies(val_X, columns=categorical_columns, dummy_na=False)
    encoded_val_X = _sanitize_encoded_columns(encoded_val_X)
    encoded_val_X = _align_frame_columns(encoded_val_X, list(encoded_train_X.columns))
    model.fit(encoded_train_X, train_y)
    return model.predict(encoded_val_X), list(encoded_train_X.columns)


def train_and_evaluate(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    train_X, train_y, feature_columns = prepare_training_matrices(train_rows)
    val_X, val_y, _ = prepare_training_matrices(validation_rows)

    model = _build_model(model_name)
    categorical_columns = _categorical_columns(train_X)
    predictions, encoded_feature_columns = _fit_and_predict(
        model,
        model_name,
        train_X,
        train_y,
        val_X,
        categorical_columns,
    )

    metrics = {
        "rmse": math.sqrt(mean_squared_error(val_y, predictions)),
        "mae": mean_absolute_error(val_y, predictions),
    }

    return {
        "pipeline": model,
        "model_name": model_name,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "encoded_feature_columns": encoded_feature_columns,
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
    }


def save_training_artifacts(result: dict[str, Any], output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], output_dir / "model.joblib")
    metrics_payload = {
        "model_name": result["model_name"],
        "metrics": result["metrics"],
        "feature_columns": result["feature_columns"],
        "categorical_columns": result["categorical_columns"],
        "encoded_feature_columns": result["encoded_feature_columns"],
        "train_row_count": result["train_row_count"],
        "validation_row_count": result["validation_row_count"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))


def load_feature_rows(path: Path | str) -> list[dict[str, str]]:
    return read_base_dataset_csv(path)
