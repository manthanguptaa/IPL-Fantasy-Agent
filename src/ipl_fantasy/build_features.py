from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


FEATURE_FIELD_ORDER = [
    # Match context
    "match_id",
    "match_date",
    "competition",
    "source_dataset",
    "season",
    "team_type",
    "match_type",
    "venue",
    "city",
    "team",
    "opponent",
    "player_name",
    "player_role",
    "batting_hand",
    "bowling_arm",
    "bowling_style",
    "ar_subtype",
    "playing_xi",
    "toss_winner",
    "toss_decision",
    "winner",
    # Batting position
    "batting_position",
    "batting_order_bucket",
    # Batting stats
    "runs",
    "balls_faced",
    "batting_balls_share",
    "fours",
    "sixes",
    "duck",
    # Bowling stats
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
    # Fielding stats
    "catches",
    "stumpings",
    "run_out_direct",
    "run_out_assist",
    # Fantasy points
    "batting_points",
    "bowling_points",
    "fielding_points",
    "other_points",
    "dream11_points_total",
    # Prior match counts
    "prior_matches_all",
    "prior_matches_ipl",
    "prior_matches_recent_t20",
    # Simple rolling averages
    "rolling_points_avg_3_all",
    "rolling_points_avg_5_all",
    "rolling_points_avg_10_all",
    "rolling_points_std_5_all",
    "rolling_runs_avg_3_all",
    "rolling_runs_avg_5_all",
    "rolling_wickets_avg_3_all",
    "rolling_wickets_avg_5_all",
    "rolling_batting_points_avg_5_all",
    "rolling_bowling_points_avg_5_all",
    "rolling_fielding_points_avg_5_all",
    "batting_match_rate_5_all",
    "bowling_match_rate_5_all",
    "rolling_balls_faced_avg_5_all",
    "rolling_balls_bowled_avg_5_all",
    "rolling_strike_rate_5_all",
    "rolling_economy_rate_5_all",
    "rolling_points_avg_3_ipl",
    "rolling_points_avg_5_ipl",
    "rolling_points_avg_3_recent_t20",
    "rolling_points_avg_5_recent_t20",
    "points_trend_3_vs_10_all",
    # Venue and opponent context
    "prior_matches_at_venue",
    "venue_points_avg_all",
    "prior_matches_vs_opponent",
    "opponent_points_avg_all",
    # Position and usage features
    "batting_position_known_rate_5_all",
    "rolling_batting_position_avg_5_all",
    "rolling_batting_balls_share_avg_5_all",
    "rolling_bowling_balls_share_avg_5_all",
    "rolling_powerplay_balls_share_avg_5_all",
    "rolling_middle_balls_share_avg_5_all",
    "rolling_death_balls_share_avg_5_all",
    # NEW: EMA (recency-weighted) features
    "ema_points_5_all",
    "ema_points_10_all",
    "ema_runs_5_all",
    "ema_wickets_5_all",
    "ema_batting_points_5_all",
    "ema_bowling_points_5_all",
    # NEW: Ceiling/volatility features
    "rolling_points_p75_10_all",
    "rolling_points_p90_10_all",
    "rolling_points_max_10_all",
    "rolling_points_min_10_all",
    # NEW: Selection certainty
    "selection_rate_10_all",
    "selection_rate_5_all",
    # NEW: Contribution mix
    "batting_points_pct_5_all",
    "bowling_points_pct_5_all",
    "fielding_points_pct_5_all",
    # NEW: Role stability
    "batting_position_std_5_all",
    "bowling_balls_share_std_5_all",
    # NEW: Duck and boundary tendency
    "duck_rate_10_all",
    "boundary_rate_5_all",
    # NEW: Role-stratified opponent features
    "opponent_points_avg_bat",
    "opponent_points_avg_bowl",
    "opponent_points_avg_ar",
    "opponent_points_avg_wk",
    "opponent_role_relative",
]


def _to_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    if value in ("", None):
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return round(variance ** 0.5, 4)


def _percentile(values: list[float], pct: float) -> float:
    """Calculate percentile (0-100) of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * pct / 100.0
    lower_idx = int(idx)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = idx - lower_idx
    return round(sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight, 4)


def _ema(values: list[float], span: int) -> float:
    """Calculate exponential moving average with given span."""
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return round(result, 4)


def _recent_average(history: deque[dict[str, float]], key: str, window: int) -> float:
    values = [entry[key] for entry in list(history)[-window:]]
    return _mean(values)


def _recent_std(history: deque[dict[str, float]], key: str, window: int) -> float:
    values = [entry[key] for entry in list(history)[-window:]]
    return _std(values)


def _recent_rate(history: deque[dict[str, float]], key: str, window: int) -> float:
    values = [entry[key] for entry in list(history)[-window:]]
    return _mean(values)


def _recent_ema(history: deque[dict[str, float]], key: str, span: int) -> float:
    """Calculate EMA from recent history."""
    values = [entry[key] for entry in list(history)[-span:]]
    return _ema(values, span)


def _recent_percentile(history: deque[dict[str, float]], key: str, window: int, pct: float) -> float:
    """Calculate percentile from recent history."""
    values = [entry[key] for entry in list(history)[-window:]]
    return _percentile(values, pct)


def _recent_max(history: deque[dict[str, float]], key: str, window: int) -> float:
    """Calculate max from recent history."""
    values = [entry[key] for entry in list(history)[-window:]]
    return max(values) if values else 0.0


def _recent_min(history: deque[dict[str, float]], key: str, window: int) -> float:
    """Calculate min from recent history."""
    values = [entry[key] for entry in list(history)[-window:]]
    return min(values) if values else 0.0


def _contribution_pct(history: deque[dict[str, float]], component_key: str, total_key: str, window: int) -> float:
    """Calculate percentage contribution of a component to total."""
    entries = list(history)[-window:]
    total_component = sum(entry[component_key] for entry in entries)
    total_points = sum(entry[total_key] for entry in entries)
    if total_points <= 0:
        return 0.0
    return round(total_component / total_points, 4)


def _recent_average_present(history: deque[dict[str, float]], key: str, present_key: str, window: int) -> float:
    entries = list(history)[-window:]
    values = [entry[key] for entry in entries if entry[present_key] > 0]
    return _mean(values)


def _recent_strike_rate(history: deque[dict[str, float]], window: int) -> float:
    entries = list(history)[-window:]
    balls_faced = sum(entry["balls_faced"] for entry in entries)
    if balls_faced == 0:
        return 0.0
    runs = sum(entry["runs"] for entry in entries)
    return round((runs / balls_faced) * 100, 4)


def _recent_economy_rate(history: deque[dict[str, float]], window: int) -> float:
    entries = list(history)[-window:]
    balls_bowled = sum(entry["balls_bowled"] for entry in entries)
    if balls_bowled == 0:
        return 0.0
    runs_conceded = sum(entry["runs_conceded"] for entry in entries)
    return round((runs_conceded * 6) / balls_bowled, 4)


def _is_ipl(row: dict[str, Any]) -> bool:
    return row.get("source_dataset") == "ipl_json" or row.get("competition") == "Indian Premier League"


def _is_recent_t20_source(row: dict[str, Any]) -> bool:
    return row.get("source_dataset") != "ipl_json"


def build_feature_rows(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        base_rows,
        key=lambda row: (
            row.get("player_name", ""),
            row.get("match_date", ""),
            row.get("match_id", ""),
        ),
    )

    history_all: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    history_ipl: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    history_recent_t20: dict[str, deque[dict[str, float]]] = defaultdict(deque)
    history_by_venue: dict[tuple[str, str], deque[float]] = defaultdict(deque)
    history_by_opponent: dict[tuple[str, str], deque[float]] = defaultdict(deque)
    # Role-stratified opponent history: (role, opponent) -> deque of points
    history_role_opponent: dict[tuple[str, str], deque[float]] = defaultdict(deque)
    feature_rows: list[dict[str, Any]] = []

    for row in sorted_rows:
        player_name = row["player_name"]
        venue = row.get("venue", "")
        opponent = row.get("opponent", "")
        all_hist = history_all[player_name]
        ipl_hist = history_ipl[player_name]
        recent_t20_hist = history_recent_t20[player_name]
        venue_hist = history_by_venue[(player_name, venue)]
        opponent_hist = history_by_opponent[(player_name, opponent)]
        feature_row = dict(row)
        feature_row["prior_matches_all"] = len(all_hist)
        feature_row["prior_matches_ipl"] = len(ipl_hist)
        feature_row["prior_matches_recent_t20"] = len(recent_t20_hist)
        feature_row["rolling_points_avg_3_all"] = _recent_average(all_hist, "points", 3)
        feature_row["rolling_points_avg_5_all"] = _recent_average(all_hist, "points", 5)
        feature_row["rolling_points_avg_10_all"] = _recent_average(all_hist, "points", 10)
        feature_row["rolling_points_std_5_all"] = _recent_std(all_hist, "points", 5)
        feature_row["rolling_runs_avg_3_all"] = _recent_average(all_hist, "runs", 3)
        feature_row["rolling_runs_avg_5_all"] = _recent_average(all_hist, "runs", 5)
        feature_row["rolling_wickets_avg_3_all"] = _recent_average(all_hist, "wickets", 3)
        feature_row["rolling_wickets_avg_5_all"] = _recent_average(all_hist, "wickets", 5)
        feature_row["rolling_batting_points_avg_5_all"] = _recent_average(
            all_hist, "batting_points", 5
        )
        feature_row["rolling_bowling_points_avg_5_all"] = _recent_average(
            all_hist, "bowling_points", 5
        )
        feature_row["rolling_fielding_points_avg_5_all"] = _recent_average(
            all_hist, "fielding_points", 5
        )
        feature_row["batting_match_rate_5_all"] = _recent_rate(all_hist, "batting_active", 5)
        feature_row["bowling_match_rate_5_all"] = _recent_rate(all_hist, "bowling_active", 5)
        feature_row["rolling_balls_faced_avg_5_all"] = _recent_average(all_hist, "balls_faced", 5)
        feature_row["rolling_balls_bowled_avg_5_all"] = _recent_average(all_hist, "balls_bowled", 5)
        feature_row["rolling_strike_rate_5_all"] = _recent_strike_rate(all_hist, 5)
        feature_row["rolling_economy_rate_5_all"] = _recent_economy_rate(all_hist, 5)
        feature_row["rolling_points_avg_3_ipl"] = _recent_average(ipl_hist, "points", 3)
        feature_row["rolling_points_avg_5_ipl"] = _recent_average(ipl_hist, "points", 5)
        feature_row["rolling_points_avg_3_recent_t20"] = _recent_average(
            recent_t20_hist, "points", 3
        )
        feature_row["rolling_points_avg_5_recent_t20"] = _recent_average(
            recent_t20_hist, "points", 5
        )
        feature_row["points_trend_3_vs_10_all"] = round(
            feature_row["rolling_points_avg_3_all"] - feature_row["rolling_points_avg_10_all"], 4
        )
        feature_row["prior_matches_at_venue"] = len(venue_hist)
        feature_row["venue_points_avg_all"] = _mean(list(venue_hist))
        feature_row["prior_matches_vs_opponent"] = len(opponent_hist)
        feature_row["opponent_points_avg_all"] = _mean(list(opponent_hist))
        # Role-stratified opponent averages
        player_role = row.get("player_role", "")
        for role_code in ("BAT", "BOWL", "AR", "WK"):
            role_opp_hist = history_role_opponent[(role_code, opponent)]
            feature_row[f"opponent_points_avg_{role_code.lower()}"] = _mean(list(role_opp_hist))
        # Relative: this player's vs-opponent avg minus their role's vs-opponent avg
        role_opp_avg = feature_row.get(f"opponent_points_avg_{player_role.lower()}", 0.0)
        feature_row["opponent_role_relative"] = (
            feature_row["opponent_points_avg_all"] - role_opp_avg
        )
        feature_row["batting_position_known_rate_5_all"] = _recent_rate(all_hist, "batted", 5)
        feature_row["rolling_batting_position_avg_5_all"] = _recent_average_present(
            all_hist, "batting_position", "batted", 5
        )
        feature_row["rolling_batting_balls_share_avg_5_all"] = _recent_average(
            all_hist, "batting_balls_share", 5
        )
        feature_row["rolling_bowling_balls_share_avg_5_all"] = _recent_average(
            all_hist, "bowling_balls_share", 5
        )
        feature_row["rolling_powerplay_balls_share_avg_5_all"] = _recent_average(
            all_hist, "powerplay_balls_share", 5
        )
        feature_row["rolling_middle_balls_share_avg_5_all"] = _recent_average(
            all_hist, "middle_balls_share", 5
        )
        feature_row["rolling_death_balls_share_avg_5_all"] = _recent_average(
            all_hist, "death_balls_share", 5
        )

        # NEW: EMA (recency-weighted) features
        feature_row["ema_points_5_all"] = _recent_ema(all_hist, "points", 5)
        feature_row["ema_points_10_all"] = _recent_ema(all_hist, "points", 10)
        feature_row["ema_runs_5_all"] = _recent_ema(all_hist, "runs", 5)
        feature_row["ema_wickets_5_all"] = _recent_ema(all_hist, "wickets", 5)
        feature_row["ema_batting_points_5_all"] = _recent_ema(all_hist, "batting_points", 5)
        feature_row["ema_bowling_points_5_all"] = _recent_ema(all_hist, "bowling_points", 5)

        # NEW: Ceiling/volatility features
        feature_row["rolling_points_p75_10_all"] = _recent_percentile(all_hist, "points", 10, 75)
        feature_row["rolling_points_p90_10_all"] = _recent_percentile(all_hist, "points", 10, 90)
        feature_row["rolling_points_max_10_all"] = _recent_max(all_hist, "points", 10)
        feature_row["rolling_points_min_10_all"] = _recent_min(all_hist, "points", 10)

        # NEW: Selection certainty (playing XI rate)
        feature_row["selection_rate_10_all"] = len(all_hist) / 10.0 if len(all_hist) < 10 else 1.0
        feature_row["selection_rate_5_all"] = len(all_hist) / 5.0 if len(all_hist) < 5 else 1.0

        # NEW: Contribution mix
        feature_row["batting_points_pct_5_all"] = _contribution_pct(
            all_hist, "batting_points", "points", 5
        )
        feature_row["bowling_points_pct_5_all"] = _contribution_pct(
            all_hist, "bowling_points", "points", 5
        )
        feature_row["fielding_points_pct_5_all"] = _contribution_pct(
            all_hist, "fielding_points", "points", 5
        )

        # NEW: Role stability
        feature_row["batting_position_std_5_all"] = _recent_std(all_hist, "batting_position", 5)
        feature_row["bowling_balls_share_std_5_all"] = _recent_std(all_hist, "bowling_balls_share", 5)

        # NEW: Duck and boundary tendency
        feature_row["duck_rate_10_all"] = _recent_average(all_hist, "duck", 10)
        feature_row["boundary_rate_5_all"] = _recent_average(all_hist, "boundary_rate", 5)

        feature_rows.append(feature_row)

        # Calculate boundary rate for this match
        balls_faced = _to_float(row, "balls_faced")
        fours = _to_float(row, "fours")
        sixes = _to_float(row, "sixes")
        boundary_rate = (fours + sixes) / balls_faced if balls_faced > 0 else 0.0

        current_summary = {
            "points": _to_float(row, "dream11_points_total"),
            "runs": _to_float(row, "runs"),
            "wickets": _to_float(row, "wickets"),
            "batting_points": _to_float(row, "batting_points"),
            "bowling_points": _to_float(row, "bowling_points"),
            "fielding_points": _to_float(row, "fielding_points"),
            "balls_faced": balls_faced,
            "balls_bowled": _to_float(row, "balls_bowled"),
            "runs_conceded": _to_float(row, "runs_conceded"),
            "batting_active": 1.0 if balls_faced > 0 else 0.0,
            "bowling_active": 1.0 if _to_float(row, "balls_bowled") > 0 else 0.0,
            "batted": 1.0 if _to_float(row, "batting_position") > 0 else 0.0,
            "batting_position": _to_float(row, "batting_position"),
            "batting_balls_share": _to_float(row, "batting_balls_share"),
            "bowling_balls_share": _to_float(row, "bowling_balls_share"),
            "powerplay_balls_share": _to_float(row, "powerplay_balls_share"),
            "middle_balls_share": _to_float(row, "middle_balls_share"),
            "death_balls_share": _to_float(row, "death_balls_share"),
            "duck": _to_float(row, "duck"),
            "boundary_rate": boundary_rate,
        }
        all_hist.append(current_summary)
        venue_hist.append(current_summary["points"])
        opponent_hist.append(current_summary["points"])
        if player_role in ("BAT", "BOWL", "AR", "WK"):
            history_role_opponent[(player_role, opponent)].append(current_summary["points"])
        if _is_ipl(row):
            ipl_hist.append(current_summary)
        if _is_recent_t20_source(row):
            recent_t20_hist.append(current_summary)

    return feature_rows


def read_base_dataset_csv(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_feature_dataset_csv(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FEATURE_FIELD_ORDER})
