"""Test-time training via residual adaptation for in-season prediction correction.

Observes (predicted, actual) pairs after each match and learns systematic
biases the offline CatBoost model misses — new team roles, changed batting
positions, venue/condition drift, impact-sub patterns, etc.

Three levels of adaptation (progressively enabled as data accumulates):

  Level 1 — Per-player EMA residual  (≥1 match per player)
      correction_i = ema_alpha * (actual - predicted) + (1 - ema_alpha) * prev_correction

  Level 2 — Per-role bias              (≥ min_matches_role total matches)
      correction_role = mean(residuals for role)

  Level 3 — Contextual ridge on features  (≥ min_matches_ridge total matches)
      correction = w @ x  where w = (X^T X + λI)^{-1} X^T r

The final correction blends the three levels with configurable weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.ipl_fantasy.team_optimizer import Player


@dataclass
class PlayerRecord:
    """Observed prediction-vs-actual record for one player-match."""
    player_name: str
    role: str
    team: str
    predicted: float
    actual: float
    residual: float  # actual - predicted
    match_id: str = ""
    match_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "role": self.role,
            "team": self.team,
            "predicted": round(self.predicted, 2),
            "actual": round(self.actual, 2),
            "residual": round(self.residual, 2),
            "match_id": self.match_id,
            "match_date": self.match_date,
        }


@dataclass
class AdapterConfig:
    """Configuration for residual adaptation."""
    # EMA smoothing factor (higher = more weight on recent matches)
    ema_alpha: float = 0.4

    # Blending weights for the three correction levels
    player_weight: float = 0.55   # Level 1: per-player correction
    role_weight: float = 0.25     # Level 2: per-role bias
    ridge_weight: float = 0.20    # Level 3: contextual ridge

    # Minimum data thresholds to activate each level
    min_matches_player: int = 1   # min observations for a player
    min_matches_role: int = 5     # total matches before role bias kicks in
    min_matches_ridge: int = 15   # total matches before ridge kicks in

    # Ridge regression regularisation
    ridge_lambda: float = 10.0

    # Safety cap: never shift a prediction by more than this fraction
    max_correction_frac: float = 0.35

    # Decay: down-weight older observations (matches) by this factor per match
    observation_decay: float = 0.95


class ResidualAdapter:
    """Online residual correction layer for test-time training.

    Usage (per match cycle):

        adapter = ResidualAdapter.load(state_path)

        # --- pre-match ---
        players = predict_fn(match_df)
        players = adapter.adjust(players)        # apply corrections
        # ... run optimizer, select team ...

        # --- post-match ---
        adapter.observe(match_id, match_date, predictions_dict, actuals_dict)
        adapter.save(state_path)
    """

    def __init__(self, config: AdapterConfig | None = None):
        self.config = config or AdapterConfig()

        # ---- Level 1: per-player tracking ----
        # player_name -> EMA of residuals
        self.player_ema: dict[str, float] = {}
        # player_name -> number of observations
        self.player_obs_count: dict[str, int] = {}
        # player_name -> role (latest seen)
        self.player_roles: dict[str, str] = {}

        # ---- Level 2: per-role tracking ----
        # role -> list of (residual, decay_weight) pairs
        self.role_residuals: dict[str, list[tuple[float, float]]] = {
            "WK": [], "BAT": [], "AR": [], "BOWL": [],
        }

        # ---- Level 3: contextual ridge ----
        # Each row is (feature_vector, residual, decay_weight)
        self.ridge_X: list[np.ndarray] = []
        self.ridge_r: list[float] = []
        self.ridge_w: list[float] = []  # decay weights
        self._ridge_coef: np.ndarray | None = None

        # ---- bookkeeping ----
        self.total_matches: int = 0
        self.history: list[dict[str, Any]] = []  # condensed per-match summaries

    # ------------------------------------------------------------------
    # Observation (post-match)
    # ------------------------------------------------------------------

    def observe(
        self,
        match_id: str,
        match_date: str,
        predictions: dict[str, dict[str, Any]],
        actuals: dict[str, float],
    ) -> dict[str, Any]:
        """Record one match's outcomes and update all correction models.

        Args:
            match_id: Unique match identifier.
            match_date: ISO date string.
            predictions: Dict of player_name -> {
                "predicted": float, "role": str, "team": str,
                "features": list[float] (optional, for ridge)
            }
            actuals: Dict of player_name -> actual fantasy points.

        Returns:
            Summary dict with per-player residuals and aggregate stats.
        """
        self.total_matches += 1
        alpha = self.config.ema_alpha
        decay = self.config.observation_decay

        # Decay all prior observations
        self._apply_decay(decay)

        records: list[PlayerRecord] = []

        for name, pred_info in predictions.items():
            if name not in actuals:
                continue  # player didn't play or no data
            predicted = pred_info["predicted"]
            actual = actuals[name]
            residual = actual - predicted
            role = pred_info.get("role", "BAT")
            team = pred_info.get("team", "")

            # Level 1: per-player EMA
            if name in self.player_ema:
                self.player_ema[name] = alpha * residual + (1 - alpha) * self.player_ema[name]
            else:
                self.player_ema[name] = residual
            self.player_obs_count[name] = self.player_obs_count.get(name, 0) + 1
            self.player_roles[name] = role

            # Level 2: per-role
            if role in self.role_residuals:
                self.role_residuals[role].append((residual, 1.0))

            # Level 3: contextual ridge features
            features = pred_info.get("features")
            if features is not None:
                self.ridge_X.append(np.array(features, dtype=np.float64))
                self.ridge_r.append(residual)
                self.ridge_w.append(1.0)

            records.append(PlayerRecord(
                player_name=name, role=role, team=team,
                predicted=predicted, actual=actual, residual=residual,
                match_id=match_id, match_date=match_date,
            ))

        # Refit ridge if we have enough data
        if self.total_matches >= self.config.min_matches_ridge and len(self.ridge_X) > 0:
            self._fit_ridge()

        # Build summary
        residuals = [r.residual for r in records]
        summary = {
            "match_id": match_id,
            "match_date": match_date,
            "n_players": len(records),
            "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
            "std_residual": float(np.std(residuals)) if residuals else 0.0,
            "total_matches_observed": self.total_matches,
            "players_tracked": len(self.player_ema),
        }
        self.history.append(summary)
        return summary

    # ------------------------------------------------------------------
    # Prediction adjustment (pre-match)
    # ------------------------------------------------------------------

    def adjust(
        self,
        players: list[Player],
        features_by_player: dict[str, list[float]] | None = None,
    ) -> list[Player]:
        """Apply residual corrections to base model predictions.

        Args:
            players: List of Player objects from the base prediction pipeline.
            features_by_player: Optional dict player_name -> feature vector
                for Level 3 ridge correction.

        Returns:
            New list of Player objects with adjusted predictions.
        """
        if self.total_matches == 0:
            return players  # no observations yet

        features_by_player = features_by_player or {}
        cfg = self.config
        adjusted = []

        for p in players:
            correction = 0.0
            weight_sum = 0.0

            # Level 1: per-player EMA
            if (p.name in self.player_ema
                    and self.player_obs_count.get(p.name, 0) >= cfg.min_matches_player):
                correction += cfg.player_weight * self.player_ema[p.name]
                weight_sum += cfg.player_weight

            # Level 2: per-role bias
            if self.total_matches >= cfg.min_matches_role:
                role_corr = self._role_correction(p.role)
                if role_corr is not None:
                    correction += cfg.role_weight * role_corr
                    weight_sum += cfg.role_weight

            # Level 3: contextual ridge
            if (self._ridge_coef is not None
                    and self.total_matches >= cfg.min_matches_ridge
                    and p.name in features_by_player):
                feat = np.array(features_by_player[p.name], dtype=np.float64)
                ridge_corr = float(self._ridge_coef @ feat)
                correction += cfg.ridge_weight * ridge_corr
                weight_sum += cfg.ridge_weight

            # Normalise by active weight sum
            if weight_sum > 0:
                correction = correction / weight_sum * min(weight_sum, 1.0)

            # Safety cap
            max_shift = cfg.max_correction_frac * abs(p.predicted_points)
            correction = max(-max_shift, min(max_shift, correction))

            # Build adjusted player
            new_predicted = max(0.0, p.predicted_points + correction)
            new_ceiling = p.ceiling
            new_floor = p.floor
            if p.ceiling is not None:
                new_ceiling = max(0.0, p.ceiling + correction * 0.7)
            if p.floor is not None:
                new_floor = max(0.0, p.floor + correction * 0.5)

            adjusted.append(Player(
                name=p.name,
                team=p.team,
                role=p.role,
                predicted_points=new_predicted,
                credits=p.credits,
                ceiling=new_ceiling,
                floor=new_floor,
                variance=p.variance,
                is_foreign=p.is_foreign,
            ))

        return adjusted

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_top_corrections(self, n: int = 15) -> list[dict[str, Any]]:
        """Return players with the largest absolute EMA corrections."""
        items = [
            {
                "player": name,
                "role": self.player_roles.get(name, "?"),
                "correction": round(ema, 2),
                "obs": self.player_obs_count.get(name, 0),
            }
            for name, ema in self.player_ema.items()
        ]
        items.sort(key=lambda x: abs(x["correction"]), reverse=True)
        return items[:n]

    def get_role_biases(self) -> dict[str, float]:
        """Return current per-role bias estimates."""
        biases = {}
        for role in ("WK", "BAT", "AR", "BOWL"):
            corr = self._role_correction(role)
            biases[role] = round(corr, 2) if corr is not None else 0.0
        return biases

    def get_summary(self) -> str:
        """Human-readable summary of adapter state."""
        lines = [
            "=" * 60,
            "RESIDUAL ADAPTER — TEST-TIME TRAINING STATE",
            "=" * 60,
        ]

        if self.total_matches == 0:
            lines.append("No matches observed yet. Adapter is inactive.")
            return "\n".join(lines)

        lines.append(f"Matches observed: {self.total_matches}")
        lines.append(f"Players tracked:  {len(self.player_ema)}")

        # Active levels
        levels = ["Level 1: per-player EMA (ACTIVE)"]
        if self.total_matches >= self.config.min_matches_role:
            levels.append("Level 2: per-role bias  (ACTIVE)")
        else:
            levels.append(f"Level 2: per-role bias  (need {self.config.min_matches_role - self.total_matches} more matches)")
        if self._ridge_coef is not None:
            levels.append("Level 3: contextual ridge (ACTIVE)")
        else:
            need = max(0, self.config.min_matches_ridge - self.total_matches)
            levels.append(f"Level 3: contextual ridge (need {need} more matches)")
        lines.append("\nCorrection levels:")
        for lv in levels:
            lines.append(f"  {lv}")

        # Role biases
        biases = self.get_role_biases()
        lines.append("\nRole biases (avg residual):")
        for role in ("WK", "BAT", "AR", "BOWL"):
            lines.append(f"  {role:<5}: {biases[role]:+.1f} pts")

        # Top corrections
        top = self.get_top_corrections(10)
        if top:
            lines.append("\nTop player corrections:")
            lines.append(f"  {'Player':<28} {'Role':<5} {'Correction':>10} {'Obs':>4}")
            lines.append("  " + "-" * 49)
            for t in top:
                lines.append(f"  {t['player']:<28} {t['role']:<5} {t['correction']:>+10.1f} {t['obs']:>4}")

        # Learning curve
        if len(self.history) >= 4:
            mid = len(self.history) // 2
            first_half = [h["mean_residual"] for h in self.history[:mid]]
            second_half = [h["mean_residual"] for h in self.history[mid:]]
            lines.append(f"\nBias trend:")
            lines.append(f"  First half  mean residual: {np.mean(first_half):+.1f}")
            lines.append(f"  Second half mean residual: {np.mean(second_half):+.1f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist adapter state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "ema_alpha": self.config.ema_alpha,
                "player_weight": self.config.player_weight,
                "role_weight": self.config.role_weight,
                "ridge_weight": self.config.ridge_weight,
                "min_matches_player": self.config.min_matches_player,
                "min_matches_role": self.config.min_matches_role,
                "min_matches_ridge": self.config.min_matches_ridge,
                "ridge_lambda": self.config.ridge_lambda,
                "max_correction_frac": self.config.max_correction_frac,
                "observation_decay": self.config.observation_decay,
            },
            "total_matches": self.total_matches,
            "player_ema": self.player_ema,
            "player_obs_count": self.player_obs_count,
            "player_roles": self.player_roles,
            "role_residuals": {
                role: [(r, w) for r, w in pairs]
                for role, pairs in self.role_residuals.items()
            },
            "history": self.history,
        }

        # Save ridge data separately (numpy arrays)
        if self.ridge_X:
            state["ridge_X"] = [x.tolist() for x in self.ridge_X]
            state["ridge_r"] = self.ridge_r
            state["ridge_w"] = self.ridge_w
        if self._ridge_coef is not None:
            state["ridge_coef"] = self._ridge_coef.tolist()

        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "ResidualAdapter":
        """Load adapter state from disk. Returns fresh adapter if file missing."""
        path = Path(path)
        if not path.exists():
            return cls()

        state = json.loads(path.read_text())

        cfg_data = state.get("config", {})
        config = AdapterConfig(
            ema_alpha=cfg_data.get("ema_alpha", 0.4),
            player_weight=cfg_data.get("player_weight", 0.55),
            role_weight=cfg_data.get("role_weight", 0.25),
            ridge_weight=cfg_data.get("ridge_weight", 0.20),
            min_matches_player=cfg_data.get("min_matches_player", 1),
            min_matches_role=cfg_data.get("min_matches_role", 5),
            min_matches_ridge=cfg_data.get("min_matches_ridge", 15),
            ridge_lambda=cfg_data.get("ridge_lambda", 10.0),
            max_correction_frac=cfg_data.get("max_correction_frac", 0.35),
            observation_decay=cfg_data.get("observation_decay", 0.95),
        )

        adapter = cls(config=config)
        adapter.total_matches = state.get("total_matches", 0)
        adapter.player_ema = state.get("player_ema", {})
        adapter.player_obs_count = state.get("player_obs_count", {})
        adapter.player_roles = state.get("player_roles", {})

        for role in ("WK", "BAT", "AR", "BOWL"):
            pairs = state.get("role_residuals", {}).get(role, [])
            adapter.role_residuals[role] = [(r, w) for r, w in pairs]

        adapter.history = state.get("history", [])

        if "ridge_X" in state:
            adapter.ridge_X = [np.array(x) for x in state["ridge_X"]]
            adapter.ridge_r = state["ridge_r"]
            adapter.ridge_w = state["ridge_w"]
        if "ridge_coef" in state:
            adapter._ridge_coef = np.array(state["ridge_coef"])

        return adapter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_decay(self, decay: float) -> None:
        """Decay all stored observation weights."""
        for role in self.role_residuals:
            self.role_residuals[role] = [
                (r, w * decay) for r, w in self.role_residuals[role]
            ]
        self.ridge_w = [w * decay for w in self.ridge_w]

    def _role_correction(self, role: str) -> float | None:
        """Weighted mean residual for a role."""
        pairs = self.role_residuals.get(role, [])
        if not pairs:
            return None
        residuals = np.array([r for r, _ in pairs])
        weights = np.array([w for _, w in pairs])
        total_w = weights.sum()
        if total_w < 1e-8:
            return None
        return float(np.dot(residuals, weights) / total_w)

    def _fit_ridge(self) -> None:
        """Fit weighted ridge regression on accumulated feature-residual pairs."""
        if len(self.ridge_X) < 5:
            return

        X = np.stack(self.ridge_X)
        r = np.array(self.ridge_r)
        w = np.array(self.ridge_w)

        # Weighted least squares: (X^T W X + λI)^{-1} X^T W r
        W = np.diag(w)
        XtW = X.T @ W
        A = XtW @ X + self.config.ridge_lambda * np.eye(X.shape[1])
        b = XtW @ r

        try:
            self._ridge_coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self._ridge_coef = None
