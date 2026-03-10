"""Live Reinforcement Learning layer for IPL Fantasy Agent.

Implements a LinUCB contextual bandit that learns which optimization
strategy to deploy given the match context. Each "arm" is a distinct
optimizer configuration (ceiling weights, captain weights, etc.).

The agent:
1. Extracts context features from the match (venue, teams, player pool stats).
2. Selects an arm (policy configuration) using LinUCB.
3. Runs the optimizer with that configuration.
4. After the match, observes the reward and updates the bandit.

This is Phase 4 of the project roadmap — contextual bandit / RL layer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ipl_fantasy.enhanced_prediction import PredictionConfig, create_enhanced_predict_fn
from src.ipl_fantasy.improved_optimizer import ImprovedDream11Optimizer, OptimizationConfig
from src.ipl_fantasy.quantile_model import QuantileModelEnsemble
from src.ipl_fantasy.reward_model import compute_reward, RewardConfig
from src.ipl_fantasy.team_optimizer import Player, Dream11Constraints
from src.ipl_fantasy.team_reranker import (
    RerankingConfig,
    select_best_team,
    select_top_k,
)


CONTEXT_DIM = 12


def extract_match_context(match_df: pd.DataFrame) -> np.ndarray:
    """
    Extract a fixed-size context vector from match-level data.

    Features (12-d):
        0: mean rolling avg points (pool quality)
        1: std of rolling avg points (pool variance)
        2: max rolling avg points (star player signal)
        3: mean ceiling (q90) signal
        4: ceiling-to-mean ratio (upside potential of pool)
        5: fraction of all-rounders
        6: fraction of bowlers
        7: mean experience (prior matches)
        8: won_toss indicator
        9: mean venue points avg (venue quality signal)
       10: mean opponent points avg (opponent difficulty)
       11: bias term (always 1.0)
    """
    ctx = np.zeros(CONTEXT_DIM, dtype=np.float64)

    def safe_col(col: str, default: float = 0.0) -> pd.Series:
        if col in match_df.columns:
            return pd.to_numeric(match_df[col], errors="coerce").fillna(default)
        return pd.Series([default] * len(match_df))

    points_10 = safe_col("rolling_points_avg_10_all", 30.0)
    points_p90 = safe_col("rolling_points_p90_10_all", 45.0)
    prior_matches = safe_col("prior_matches_all", 5.0)
    venue_avg = safe_col("venue_points_avg_all", 30.0)
    opp_avg = safe_col("opponent_points_avg_all", 30.0)

    mean_pts = points_10.mean()
    ctx[0] = mean_pts / 50.0  # Normalize to ~[0, 1.5]
    ctx[1] = (points_10.std() if len(points_10) > 1 else 0.0) / 20.0
    ctx[2] = points_10.max() / 80.0
    ctx[3] = points_p90.mean() / 60.0
    ctx[4] = (points_p90.mean() / mean_pts - 1.0) if mean_pts > 0 else 0.0  # Upside ratio

    if "player_role" in match_df.columns:
        roles = match_df["player_role"].fillna("BAT")
        ctx[5] = (roles == "AR").mean()
        ctx[6] = (roles == "BOWL").mean()
    else:
        ctx[5] = 0.2
        ctx[6] = 0.3

    ctx[7] = prior_matches.mean() / 50.0

    if "won_toss" in match_df.columns:
        ctx[8] = float(match_df["won_toss"].max())
    else:
        ctx[8] = 0.5

    ctx[9] = venue_avg.mean() / 50.0
    ctx[10] = opp_avg.mean() / 50.0
    ctx[11] = 1.0  # Bias

    return ctx



@dataclass
class PolicyArm:
    """A named optimizer configuration that forms one bandit arm."""
    name: str
    prediction_config: PredictionConfig
    optimization_config: OptimizationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role_ceiling_weights": self.prediction_config.role_ceiling_weights,
            "captain_ceiling_weight": self.prediction_config.captain_ceiling_weight,
            "expected_weight": self.optimization_config.expected_weight,
            "ceiling_weight": self.optimization_config.ceiling_weight,
            "floor_weight": self.optimization_config.floor_weight,
        }


def build_default_arms() -> list[PolicyArm]:
    """Create the default set of strategy arms."""
    arms = [
        # 0: Conservative — trust expected value, low ceiling
        PolicyArm(
            name="conservative",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.20, "WK": 0.15, "BAT": 0.10, "BOWL": 0.10},
                captain_ceiling_weight=0.30,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.80, ceiling_weight=0.15, floor_weight=0.05,
                captain_ceiling_weight=0.30,
            ),
        ),
        # 1: Balanced (current production default)
        PolicyArm(
            name="balanced",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.50, "WK": 0.45, "BAT": 0.35, "BOWL": 0.30},
                captain_ceiling_weight=0.55,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.60, ceiling_weight=0.30, floor_weight=0.10,
                captain_ceiling_weight=0.50,
            ),
        ),
        # 2: Aggressive — heavy ceiling weighting
        PolicyArm(
            name="aggressive",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.70, "WK": 0.65, "BAT": 0.55, "BOWL": 0.45},
                captain_ceiling_weight=0.70,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.35, ceiling_weight=0.55, floor_weight=0.10,
                captain_ceiling_weight=0.70,
            ),
        ),
        # 3: Captain-focused — maximize captain upside
        PolicyArm(
            name="captain_focused",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.45, "WK": 0.40, "BAT": 0.30, "BOWL": 0.25},
                captain_ceiling_weight=0.80,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.55, ceiling_weight=0.30, floor_weight=0.15,
                captain_ceiling_weight=0.80,
            ),
        ),
        # 4: Floor-safe — maximize floor, minimize variance
        PolicyArm(
            name="floor_safe",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.15, "WK": 0.10, "BAT": 0.10, "BOWL": 0.10},
                captain_ceiling_weight=0.25,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.60, ceiling_weight=0.10, floor_weight=0.30,
                captain_ceiling_weight=0.25,
            ),
        ),
        # 5: AR/WK specialist — exploit high-variance roles
        PolicyArm(
            name="ar_wk_specialist",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.75, "WK": 0.70, "BAT": 0.25, "BOWL": 0.20},
                captain_ceiling_weight=0.65,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.50, ceiling_weight=0.40, floor_weight=0.10,
                captain_ceiling_weight=0.65,
            ),
        ),
        # 6: Moderate ceiling — slight ceiling lean
        PolicyArm(
            name="moderate_ceiling",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.55, "WK": 0.50, "BAT": 0.40, "BOWL": 0.35},
                captain_ceiling_weight=0.60,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.50, ceiling_weight=0.40, floor_weight=0.10,
                captain_ceiling_weight=0.60,
            ),
        ),
        # 7: Bowler-heavy ceiling — exploit bowler breakouts
        PolicyArm(
            name="bowler_ceiling",
            prediction_config=PredictionConfig(
                role_ceiling_weights={"AR": 0.40, "WK": 0.35, "BAT": 0.30, "BOWL": 0.55},
                captain_ceiling_weight=0.50,
            ),
            optimization_config=OptimizationConfig(
                expected_weight=0.55, ceiling_weight=0.35, floor_weight=0.10,
                captain_ceiling_weight=0.50,
            ),
        ),
    ]
    return arms



class LinUCBAgent:
    """
    LinUCB (disjoint) contextual bandit.

    Maintains per-arm ridge regression models that predict reward from
    context. Selects the arm with the highest upper confidence bound.

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized
    News Article Recommendation", WWW 2010.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int = CONTEXT_DIM,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A: list[np.ndarray] = [
            lambda_reg * np.eye(self.d) for _ in range(n_arms)
        ]
        self.b: list[np.ndarray] = [
            np.zeros(self.d) for _ in range(n_arms)
        ]
        self.pull_counts: list[int] = [0] * n_arms

    def select_arm(self, context: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Select the arm with the highest UCB.

        Args:
            context: Context feature vector (d,)

        Returns:
            (chosen_arm_index, ucb_scores_for_all_arms)
        """
        x = context.reshape(-1, 1)  # Column vector
        ucbs = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta_a = A_inv @ self.b[a]

            # UCB = theta^T x + alpha * sqrt(x^T A^{-1} x)
            mean = float(theta_a @ context)
            exploration = self.alpha * math.sqrt(float(x.T @ A_inv @ x))
            ucbs[a] = mean + exploration

        chosen = int(np.argmax(ucbs))
        return chosen, ucbs

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update the model for the chosen arm with observed reward.

        Args:
            arm: Index of the pulled arm.
            context: Context vector (d,).
            reward: Observed scalar reward.
        """
        x = context.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * context
        self.pull_counts[arm] += 1

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Return per-arm statistics."""
        stats = []
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            stats.append({
                "arm": a,
                "pulls": self.pull_counts[a],
                "theta_norm": float(np.linalg.norm(theta)),
            })
        return stats

    def save(self, path: Path | str) -> None:
        """Persist bandit state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "n_arms": self.n_arms,
            "d": self.d,
            "alpha": self.alpha,
            "pull_counts": self.pull_counts,
        }
        (path / "config.json").write_text(json.dumps(state, indent=2))
        for a in range(self.n_arms):
            np.save(path / f"A_{a}.npy", self.A[a])
            np.save(path / f"b_{a}.npy", self.b[a])

    @classmethod
    def load(cls, path: Path | str) -> "LinUCBAgent":
        """Load bandit state from disk."""
        path = Path(path)
        state = json.loads((path / "config.json").read_text())
        agent = cls(
            n_arms=state["n_arms"],
            context_dim=state["d"],
            alpha=state["alpha"],
        )
        agent.pull_counts = state["pull_counts"]
        for a in range(agent.n_arms):
            agent.A[a] = np.load(path / f"A_{a}.npy")
            agent.b[a] = np.load(path / f"b_{a}.npy")
        return agent



@dataclass
class Experience:
    """A single (context, arm, reward) transition."""
    match_id: str
    match_date: str
    context: np.ndarray
    arm_index: int
    arm_name: str
    reward: float
    selected_score: float
    oracle_score: float
    regret: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "match_date": self.match_date,
            "arm_index": self.arm_index,
            "arm_name": self.arm_name,
            "reward": self.reward,
            "selected_score": self.selected_score,
            "oracle_score": self.oracle_score,
            "regret": self.regret,
        }


class ExperienceBuffer:
    """Stores and manages transition history."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences: list[Experience] = []

    def add(self, exp: Experience) -> None:
        self.experiences.append(exp)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def __len__(self) -> int:
        return len(self.experiences)

    def get_recent(self, n: int) -> list[Experience]:
        return self.experiences[-n:]

    def get_arm_rewards(self) -> dict[int, list[float]]:
        """Group rewards by arm index."""
        arm_rewards: dict[int, list[float]] = {}
        for exp in self.experiences:
            arm_rewards.setdefault(exp.arm_index, []).append(exp.reward)
        return arm_rewards

    def get_cumulative_regret(self) -> list[float]:
        """Return cumulative regret over time."""
        cumulative = []
        total = 0.0
        for exp in self.experiences:
            total += exp.regret
            cumulative.append(total)
        return cumulative

    def get_rolling_reward(self, window: int = 10) -> list[float]:
        """Return rolling average reward."""
        rewards = [e.reward for e in self.experiences]
        if len(rewards) < window:
            return [np.mean(rewards[:i+1]) for i in range(len(rewards))]
        rolling = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling.append(np.mean(rewards[start:i+1]))
        return rolling

    def save(self, path: Path | str) -> None:
        path = Path(path)
        data = [e.to_dict() for e in self.experiences]
        path.write_text(json.dumps(data, indent=2))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.experiences])



class LiveRLAgent:
    """
    Orchestrates contextual-bandit–driven team selection.

    Usage:
        agent = LiveRLAgent.from_models(ensemble_path)
        # For each match:
        team, arm = agent.select_team(match_df)
        # After match outcome:
        agent.observe(match_df, arm, actual_points_dict)
    """

    def __init__(
        self,
        ensemble: QuantileModelEnsemble,
        arms: list[PolicyArm] | None = None,
        alpha: float = 1.0,
        reward_config: RewardConfig | None = None,
    ):
        self.ensemble = ensemble
        self.arms = arms or build_default_arms()
        self.reward_config = reward_config or RewardConfig()
        self.bandit = LinUCBAgent(
            n_arms=len(self.arms),
            context_dim=CONTEXT_DIM,
            alpha=alpha,
        )
        self.buffer = ExperienceBuffer()
        self.constraints = Dream11Constraints()

    def select_team(
        self,
        match_df: pd.DataFrame,
        force_arm: int | None = None,
    ) -> tuple[Any, int, np.ndarray]:
        """
        Select a Dream11 team for the given match.

        Args:
            match_df: DataFrame of player rows for one match.
            force_arm: If set, use this arm index (for evaluation).

        Returns:
            (OptimizationResult, arm_index, context_vector)
        """
        context = extract_match_context(match_df)

        if force_arm is not None:
            arm_idx = force_arm
        else:
            arm_idx, _ = self.bandit.select_arm(context)

        arm = self.arms[arm_idx]

        predict_fn = create_enhanced_predict_fn(
            self.ensemble,
            config=arm.prediction_config,
            use_improved_credits=True,
        )
        players = predict_fn(match_df)

        optimizer = ImprovedDream11Optimizer(
            constraints=self.constraints,
            config=arm.optimization_config,
        )
        result = optimizer.optimize_ceiling_weighted(players)

        return result, arm_idx, context

    def select_team_with_reranking(
        self,
        match_df: pd.DataFrame,
        reranking_config: RerankingConfig | None = None,
        force_arm: int | None = None,
    ) -> tuple[Any, int, np.ndarray, str, str]:
        """
        Select a team using the bandit to choose prediction config, then
        generate K candidate teams and rerank via simulation.

        Combines Phase 2 (reranking) with Phase 4 (RL arm selection).

        Returns:
            (OptimizationResult, arm_index, context, sim_captain, sim_vc)
        """
        reranking_config = reranking_config or RerankingConfig(
            n_candidates=8, n_simulations=3000,
        )
        context = extract_match_context(match_df)

        if force_arm is not None:
            arm_idx = force_arm
        else:
            arm_idx, _ = self.bandit.select_arm(context)

        arm = self.arms[arm_idx]

        predict_fn = create_enhanced_predict_fn(
            self.ensemble,
            config=arm.prediction_config,
            use_improved_credits=True,
        )
        players = predict_fn(match_df)

        best = select_best_team(players, self.constraints, reranking_config)

        return best.result, arm_idx, context, best.sim_captain, best.sim_vc

    def observe(
        self,
        match_df: pd.DataFrame,
        arm_idx: int,
        context: np.ndarray,
        result: Any,
        actual_points: dict[str, float],
        override_captain: str | None = None,
        override_vc: str | None = None,
    ) -> Experience:
        """
        Observe the match outcome and update the bandit.

        Args:
            match_df: Match DataFrame.
            arm_idx: Arm that was used.
            context: Context vector that was used.
            result: OptimizationResult from select_team.
            actual_points: Dict mapping player_name -> actual fantasy points.
            override_captain: Simulation-optimal captain name (from reranker).
            override_vc: Simulation-optimal VC name (from reranker).

        Returns:
            Experience object with computed reward.
        """
        selected_players = [p.name for p in result.selected_players]
        captain_name = override_captain or (result.captain.name if result.captain else "")
        vc_name = override_vc or (result.vice_captain.name if result.vice_captain else "")

        base_score = sum(actual_points.get(p, 0) for p in selected_players)
        cap_bonus = actual_points.get(captain_name, 0)
        vc_bonus = actual_points.get(vc_name, 0) * 0.5
        selected_score = base_score + cap_bonus + vc_bonus

        from src.ipl_fantasy.backtesting import Backtester
        backtester = Backtester(constraints=self.constraints)
        all_players = self._build_oracle_players(match_df, actual_points)
        oracle_result = backtester.calculate_oracle_team(all_players)

        oracle_names = [p.name for p in oracle_result.selected_players]
        oracle_cap = oracle_result.captain.name if oracle_result.captain else ""
        oracle_vc = oracle_result.vice_captain.name if oracle_result.vice_captain else ""
        oracle_score = backtester.calculate_score_with_cv(
            oracle_names, oracle_cap, oracle_vc, actual_points,
        )

        captain_actual = actual_points.get(captain_name, 0)
        oracle_captain_actual = actual_points.get(oracle_cap, 0)

        reward = compute_reward(
            selected_score=selected_score,
            oracle_score=oracle_score,
            captain_actual=captain_actual,
            oracle_captain_actual=oracle_captain_actual,
            config=self.reward_config,
        )

        self.bandit.update(arm_idx, context, reward)

        match_id = str(match_df["match_id"].iloc[0]) if "match_id" in match_df.columns else "unknown"
        match_date = str(match_df["match_date"].iloc[0]) if "match_date" in match_df.columns else "unknown"

        exp = Experience(
            match_id=match_id,
            match_date=match_date,
            context=context,
            arm_index=arm_idx,
            arm_name=self.arms[arm_idx].name,
            reward=reward,
            selected_score=selected_score,
            oracle_score=oracle_score,
            regret=oracle_score - selected_score,
        )
        self.buffer.add(exp)

        return exp

    def _build_oracle_players(
        self,
        match_df: pd.DataFrame,
        actual_points: dict[str, float],
    ) -> list[tuple[Player, float]]:
        """Build Player-actual pairs for oracle calculation."""
        from src.ipl_fantasy.credit_estimation import estimate_credits_from_history

        players_with_actual = []
        for _, row in match_df.iterrows():
            name = row.get("player_name", "")
            role = row.get("player_role", "BAT")
            if role not in ("WK", "BAT", "AR", "BOWL"):
                role = "BAT"

            avg_all = row.get("rolling_points_avg_10_all",
                              row.get("rolling_points_avg_5_all", 30.0))
            avg_recent = row.get("rolling_points_avg_5_all", None)
            credits = estimate_credits_from_history(
                player_name=name,
                player_role=role,
                avg_points_all=avg_all if pd.notna(avg_all) else 30.0,
                avg_points_recent=avg_recent if pd.notna(avg_recent) else None,
            )

            player = Player(
                name=name,
                team=row.get("team", "Unknown"),
                role=role,
                predicted_points=actual_points.get(name, 0),
                credits=credits,
            )
            players_with_actual.append((player, actual_points.get(name, 0)))

        return players_with_actual

    def get_summary(self) -> str:
        """Return a human-readable summary of RL learning progress."""
        lines = [
            "=" * 60,
            "LIVE RL AGENT SUMMARY",
            "=" * 60,
        ]

        if len(self.buffer) == 0:
            lines.append("No matches played yet.")
            return "\n".join(lines)

        lines.append(f"Matches played: {len(self.buffer)}")

        arm_rewards = self.buffer.get_arm_rewards()
        lines.append("\nArm Performance:")
        lines.append(f"  {'Arm':<25} {'Pulls':>6} {'Avg Reward':>12} {'Avg Regret':>12}")
        lines.append("  " + "-" * 57)

        for i, arm in enumerate(self.arms):
            rewards = arm_rewards.get(i, [])
            pulls = len(rewards)
            avg_r = np.mean(rewards) if rewards else 0.0

            # Get avg regret for this arm
            arm_exps = [e for e in self.buffer.experiences if e.arm_index == i]
            avg_regret = np.mean([e.regret for e in arm_exps]) if arm_exps else 0.0

            lines.append(f"  {arm.name:<25} {pulls:>6} {avg_r:>12.3f} {avg_regret:>12.1f}")

        all_rewards = [e.reward for e in self.buffer.experiences]
        all_regrets = [e.regret for e in self.buffer.experiences]
        all_scores = [e.selected_score for e in self.buffer.experiences]

        lines.append(f"\nOverall:")
        lines.append(f"  Mean reward:         {np.mean(all_rewards):.3f}")
        lines.append(f"  Mean selected score: {np.mean(all_scores):.1f}")
        lines.append(f"  Mean regret:         {np.mean(all_regrets):.1f}")

        if len(self.buffer) >= 10:
            mid = len(self.buffer) // 2
            first_half = self.buffer.experiences[:mid]
            second_half = self.buffer.experiences[mid:]

            first_regret = np.mean([e.regret for e in first_half])
            second_regret = np.mean([e.regret for e in second_half])
            delta = second_regret - first_regret

            lines.append(f"\nLearning Trend:")
            lines.append(f"  First half avg regret:  {first_regret:.1f}")
            lines.append(f"  Second half avg regret: {second_regret:.1f}")
            lines.append(f"  Change: {delta:+.1f} ({'improving' if delta < 0 else 'degrading'})")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, output_dir: Path | str) -> None:
        """Save full agent state."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.bandit.save(output_dir / "bandit")
        self.buffer.save(output_dir / "experience_buffer.json")

        arms_data = [arm.to_dict() for arm in self.arms]
        (output_dir / "arms.json").write_text(json.dumps(arms_data, indent=2))

        print(f"RL agent saved to {output_dir}")

    @classmethod
    def from_models(
        cls,
        ensemble_path: Path | str,
        agent_path: Path | str | None = None,
        alpha: float = 1.0,
    ) -> "LiveRLAgent":
        """
        Create or resume a LiveRLAgent.

        Args:
            ensemble_path: Path to trained quantile models.
            agent_path: Path to saved agent state (for resuming).
            alpha: LinUCB exploration parameter.

        Returns:
            LiveRLAgent instance.
        """
        ensemble = QuantileModelEnsemble.load(ensemble_path)
        agent = cls(ensemble=ensemble, alpha=alpha)

        if agent_path and Path(agent_path).exists():
            bandit_path = Path(agent_path) / "bandit"
            if bandit_path.exists():
                agent.bandit = LinUCBAgent.load(bandit_path)
                print(f"Resumed bandit from {bandit_path}")

        return agent
