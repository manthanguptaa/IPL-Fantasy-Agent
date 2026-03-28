"""Quantile regression models for distributional fantasy point predictions.

This module trains multiple quantile regression models to predict:
- Floor (10th percentile): Conservative/worst-case estimate
- Median (50th percentile): Central estimate
- Ceiling (90th percentile): Upside/best-case estimate

This enables uncertainty estimation and simulation-based team selection.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Optimal features from experimentation
OPTIMAL_FEATURES = [
    "rolling_batting_position_avg_5_all",
    "rolling_balls_bowled_avg_5_all",
    "venue_points_avg_all",
    "rolling_points_avg_10_all",
    "rolling_death_balls_share_avg_5_all",
    "prior_matches_all",
    "rolling_bowling_balls_share_avg_5_all",
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
    "won_toss",
    # Role-stratified opponent features (replaces generic opponent_points_avg_all)
    "opponent_points_avg_bat",
    "opponent_points_avg_bowl",
    "opponent_points_avg_ar",
    "opponent_points_avg_wk",
    "opponent_role_relative",
]

# Default quantiles to predict
DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

TARGET_COLUMN = "dream11_points_total"


@dataclass
class QuantilePrediction:
    """Prediction with uncertainty estimates."""
    player_name: str
    team: str
    role: str

    # Point estimates at different quantiles
    q10: float  # Floor (10th percentile)
    q25: float  # Lower quartile
    q50: float  # Median
    q75: float  # Upper quartile
    q90: float  # Ceiling (90th percentile)

    # Derived metrics
    expected: float  # Mean estimate (from main model)
    variance: float  # Estimated variance
    upside: float    # Ceiling - Expected
    downside: float  # Expected - Floor

    @property
    def iqr(self) -> float:
        """Interquartile range (measure of spread)."""
        return self.q75 - self.q25

    @property
    def range_90(self) -> float:
        """90% prediction interval width."""
        return self.q90 - self.q10

    @property
    def skew_ratio(self) -> float:
        """Ratio of upside to downside potential."""
        if self.downside <= 0:
            return float('inf')
        return self.upside / self.downside


class QuantileModelEnsemble:
    """Ensemble of quantile regression models for distributional predictions."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        features: list[str] | None = None,
    ):
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.features = features or OPTIMAL_FEATURES
        self.models: dict[float, CatBoostRegressor] = {}
        self.mean_model: CatBoostRegressor | None = None
        self.categorical_features: list[str] = []

    def _get_model_params(self, quantile: float) -> dict[str, Any]:
        """Get CatBoost parameters for a specific quantile."""
        return {
            "loss_function": f"Quantile:alpha={quantile}",
            "eval_metric": f"Quantile:alpha={quantile}",
            "iterations": 800,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 10,
            "random_seed": 42,
            "verbose": False,
        }

    def _get_mean_model_params(self) -> dict[str, Any]:
        """Get CatBoost parameters for mean prediction."""
        return {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1000,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 10,
            "random_seed": 42,
            "verbose": False,
        }

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Prepare feature matrix."""
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features].copy()

        categorical_cols = []
        for col in available_features:
            if X[col].dtype == object or not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna("").astype(str)
                categorical_cols.append(col)
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        return X, categorical_cols

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Train quantile models on training data.

        Args:
            train_df: Training data with features and target
            val_df: Optional validation data for metrics

        Returns:
            Dictionary of training metrics
        """
        train_X, self.categorical_features = self._prepare_features(train_df)
        train_y = train_df[TARGET_COLUMN].astype(float).values

        if val_df is not None:
            val_X, _ = self._prepare_features(val_df)
            val_y = val_df[TARGET_COLUMN].astype(float).values

        metrics = {}

        # Train mean model
        print("Training mean model...")
        self.mean_model = CatBoostRegressor(**self._get_mean_model_params())
        self.mean_model.fit(train_X, train_y, cat_features=self.categorical_features)

        if val_df is not None:
            mean_pred = self.mean_model.predict(val_X)
            metrics["mean_rmse"] = math.sqrt(mean_squared_error(val_y, mean_pred))
            metrics["mean_mae"] = mean_absolute_error(val_y, mean_pred)
            print(f"  Mean model - RMSE: {metrics['mean_rmse']:.4f}, MAE: {metrics['mean_mae']:.4f}")

        # Train quantile models
        for q in self.quantiles:
            print(f"Training quantile model (q={q})...")
            model = CatBoostRegressor(**self._get_model_params(q))
            model.fit(train_X, train_y, cat_features=self.categorical_features)
            self.models[q] = model

            if val_df is not None:
                q_pred = model.predict(val_X)
                # Calculate quantile loss (pinball loss)
                errors = val_y - q_pred
                q_loss = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
                metrics[f"q{int(q*100)}_loss"] = q_loss

                # Calculate coverage (what fraction of actuals fall below prediction)
                coverage = np.mean(val_y <= q_pred)
                metrics[f"q{int(q*100)}_coverage"] = coverage
                print(f"  q={q} - Loss: {q_loss:.4f}, Coverage: {coverage:.2%} (target: {q:.0%})")

        return metrics

    def predict(self, df: pd.DataFrame) -> list[QuantilePrediction]:
        """
        Generate distributional predictions for players.

        Args:
            df: DataFrame with player features

        Returns:
            List of QuantilePrediction objects
        """
        if not self.models or self.mean_model is None:
            raise ValueError("Models not trained. Call fit() first.")

        X, cat_cols = self._prepare_features(df)
        pool = Pool(X, cat_features=cat_cols)

        # Get predictions from all models
        mean_pred = self.mean_model.predict(pool)
        quantile_preds = {q: model.predict(pool) for q, model in self.models.items()}

        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Get quantile values (with monotonicity enforcement)
            q_values = sorted([
                (q, max(0, quantile_preds[q][i]))  # Ensure non-negative
                for q in self.quantiles
            ])

            # Enforce monotonicity (higher quantiles should be >= lower quantiles)
            for j in range(1, len(q_values)):
                if q_values[j][1] < q_values[j-1][1]:
                    q_values[j] = (q_values[j][0], q_values[j-1][1])

            q_dict = dict(q_values)
            expected = max(0, mean_pred[i])

            pred = QuantilePrediction(
                player_name=row.get("player_name", f"Player_{i}"),
                team=row.get("team", "Unknown"),
                role=row.get("player_role", "BAT"),
                q10=q_dict.get(0.10, expected * 0.5),
                q25=q_dict.get(0.25, expected * 0.7),
                q50=q_dict.get(0.50, expected),
                q75=q_dict.get(0.75, expected * 1.3),
                q90=q_dict.get(0.90, expected * 1.5),
                expected=expected,
                variance=((q_dict.get(0.90, expected*1.5) - q_dict.get(0.10, expected*0.5)) / 2.56) ** 2,  # Approx variance
                upside=q_dict.get(0.90, expected * 1.5) - expected,
                downside=expected - q_dict.get(0.10, expected * 0.5),
            )
            predictions.append(pred)

        return predictions

    def save(self, output_dir: Path | str) -> None:
        """Save all models to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save mean model
        joblib.dump(self.mean_model, output_dir / "mean_model.joblib")

        # Save quantile models
        for q, model in self.models.items():
            joblib.dump(model, output_dir / f"quantile_{int(q*100)}_model.joblib")

        # Save config
        config = {
            "quantiles": self.quantiles,
            "features": self.features,
            "categorical_features": self.categorical_features,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))

        print(f"Models saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path | str) -> "QuantileModelEnsemble":
        """Load models from directory."""
        model_dir = Path(model_dir)

        # Load config
        config = json.loads((model_dir / "config.json").read_text())

        ensemble = cls(
            quantiles=config["quantiles"],
            features=config["features"],
        )
        ensemble.categorical_features = config["categorical_features"]

        # Load mean model
        ensemble.mean_model = joblib.load(model_dir / "mean_model.joblib")

        # Load quantile models
        for q in ensemble.quantiles:
            model_path = model_dir / f"quantile_{int(q*100)}_model.joblib"
            if model_path.exists():
                ensemble.models[q] = joblib.load(model_path)

        return ensemble


def train_quantile_ensemble(
    features_path: Path | str,
    output_dir: Path | str,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """
    Train a quantile model ensemble from features dataset.

    Args:
        features_path: Path to features CSV
        output_dir: Directory to save models
        val_fraction: Fraction of data for validation

    Returns:
        Training metrics
    """
    print(f"Loading data from {features_path}...")
    df = pd.read_csv(features_path, low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Sort by date and split
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"  Train rows: {len(train_df)}")
    print(f"  Validation rows: {len(val_df)}")

    # Train ensemble
    ensemble = QuantileModelEnsemble()
    metrics = ensemble.fit(train_df, val_df)

    # Save models
    ensemble.save(output_dir)

    # Save metrics
    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/full_player_match_features_v3.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "tmp/quantile_models"

    metrics = train_quantile_ensemble(features_path, output_dir)
    print("\nTraining complete!")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
