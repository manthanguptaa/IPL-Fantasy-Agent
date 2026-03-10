from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELD_ORDER = [
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
    "batting_position",
    "batting_order_bucket",
    "runs",
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
    "dream11_points_total",
]

NON_BOWLER_WICKET_KINDS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
}


def load_player_roles(path: Path | str) -> dict[str, dict[str, str]]:
    """Load player details including role, batting_hand, bowling_arm, bowling_style, ar_subtype."""
    path = Path(path)
    if not path.exists():
        return {}

    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {
            row["player_name"]: {
                "role": row.get("role", ""),
                "batting_hand": row.get("batting_hand", ""),
                "bowling_arm": row.get("bowling_arm", ""),
                "bowling_style": row.get("bowling_style", ""),
                "ar_subtype": row.get("ar_subtype", ""),
            }
            for row in rows
            if row.get("player_name")
        }


def _batting_order_bucket(position: int) -> str:
    if position <= 0:
        return ""
    if position <= 2:
        return "opener"
    if position <= 4:
        return "top"
    if position <= 7:
        return "middle"
    return "lower"


def _new_player_row(
    match_id: str,
    info: dict[str, Any],
    source_dataset: str,
    player_name: str,
    team: str,
    opponent: str,
    player_roles: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    date_values = info.get("dates", [])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    player_details = (player_roles or {}).get(player_name, {})
    return {
        "match_id": match_id,
        "match_date": date_values[0] if date_values else "",
        "competition": (info.get("event") or {}).get("name", ""),
        "source_dataset": source_dataset,
        "season": str(info.get("season", "")),
        "team_type": info.get("team_type", ""),
        "match_type": info.get("match_type", ""),
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "team": team,
        "opponent": opponent,
        "player_name": player_name,
        "player_role": player_details.get("role", ""),
        "batting_hand": player_details.get("batting_hand", ""),
        "bowling_arm": player_details.get("bowling_arm", ""),
        "bowling_style": player_details.get("bowling_style", ""),
        "ar_subtype": player_details.get("ar_subtype", ""),
        "playing_xi": 1,
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "winner": outcome.get("winner", ""),
        "batting_position": 0,
        "batting_order_bucket": "",
        "runs": 0,
        "balls_faced": 0,
        "batting_balls_share": 0.0,
        "fours": 0,
        "sixes": 0,
        "duck": 0,
        "balls_bowled": 0,
        "overs_bowled": 0.0,
        "bowling_balls_share": 0.0,
        "powerplay_balls": 0,
        "middle_balls": 0,
        "death_balls": 0,
        "powerplay_balls_share": 0.0,
        "middle_balls_share": 0.0,
        "death_balls_share": 0.0,
        "maidens": 0,
        "runs_conceded": 0,
        "wickets": 0,
        "catches": 0,
        "stumpings": 0,
        "run_out_direct": 0,
        "run_out_assist": 0,
        "batting_points": 0,
        "bowling_points": 0,
        "fielding_points": 0,
        "other_points": 0,
        "dream11_points_total": 0,
    }


def _ensure_player_row(
    rows_by_player: dict[str, dict[str, Any]],
    player_name: str,
    match_id: str,
    info: dict[str, Any],
    source_dataset: str,
    team: str = "",
    opponent: str = "",
    player_roles: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if player_name not in rows_by_player:
        rows_by_player[player_name] = _new_player_row(
            match_id=match_id,
            info=info,
            source_dataset=source_dataset,
            player_name=player_name,
            team=team,
            opponent=opponent,
            player_roles=player_roles,
        )
    return rows_by_player[player_name]


def _fielder_entries(wicket_event: dict[str, Any]) -> list[dict[str, Any]]:
    raw_fielders = wicket_event.get("fielders", [])
    entries: list[dict[str, Any]] = []
    for fielder in raw_fielders:
        if isinstance(fielder, dict):
            name = fielder.get("name")
            substitute = bool(fielder.get("substitute", False))
        else:
            name = str(fielder)
            substitute = False
        if name:
            entries.append({"name": name, "substitute": substitute})
    return entries


def _delivery_is_wide(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {})
    return "wides" in extras


def _delivery_is_no_ball(delivery: dict[str, Any]) -> bool:
    extras = delivery.get("extras", {})
    return "noballs" in extras


def _runs_conceded_for_delivery(delivery: dict[str, Any]) -> int:
    runs = delivery.get("runs", {})
    extras = delivery.get("extras", {})
    bye_runs = extras.get("byes", 0) + extras.get("legbyes", 0)
    return int(runs.get("total", 0)) - int(bye_runs)


def _apply_batting_points(row: dict[str, Any]) -> None:
    points = row["runs"] + row["fours"] + (2 * row["sixes"])
    if row["runs"] >= 100:
        points += 16
    elif row["runs"] >= 50:
        points += 8
    if row["duck"]:
        points -= 2
    row["batting_points"] = points


def _apply_bowling_points(row: dict[str, Any]) -> None:
    points = row["wickets"] * 25
    if row["wickets"] >= 5:
        points += 16
    elif row["wickets"] >= 4:
        points += 8
    points += row["maidens"] * 8
    row["bowling_points"] = points


def _apply_fielding_points(row: dict[str, Any]) -> None:
    points = row["catches"] * 8
    if row["catches"] >= 3:
        points += 4
    points += row["stumpings"] * 12
    points += row["run_out_direct"] * 12
    points += row["run_out_assist"] * 6
    row["fielding_points"] = points


def _apply_other_points(row: dict[str, Any]) -> None:
    points = 4 if row["playing_xi"] else 0

    if row["balls_bowled"] >= 12:
        economy = row["runs_conceded"] / (row["balls_bowled"] / 6)
        if economy < 4:
            points += 6
        elif economy < 5:
            points += 4
        elif economy < 6:
            points += 2
        elif economy <= 11:
            points -= 2
        elif economy <= 12:
            points -= 4
        else:
            points -= 6

    if row["balls_faced"] >= 10:
        strike_rate = (row["runs"] * 100) / row["balls_faced"] if row["balls_faced"] else 0
        if strike_rate > 170:
            points += 6
        elif strike_rate > 150:
            points += 4
        elif strike_rate >= 130:
            points += 2
        elif strike_rate < 50:
            points -= 6
        elif strike_rate < 60:
            points -= 4
        elif strike_rate <= 70:
            points -= 2

    row["other_points"] = points


def _finalize_rows(rows_by_player: dict[str, dict[str, Any]], dismissed_players: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player_name, row in rows_by_player.items():
        row["overs_bowled"] = round(row["balls_bowled"] / 6, 2)
        row["duck"] = int(player_name in dismissed_players and row["runs"] == 0)
        _apply_batting_points(row)
        _apply_bowling_points(row)
        _apply_fielding_points(row)
        _apply_other_points(row)
        row["dream11_points_total"] = (
            row["batting_points"]
            + row["bowling_points"]
            + row["fielding_points"]
            + row["other_points"]
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (row["team"], row["player_name"]))


def normalize_match_file(
    path: Path | str,
    source_dataset: str | None = None,
    gender_filter: set[str] | None = None,
    player_roles: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    source_dataset = source_dataset or path.parent.name
    data = json.loads(path.read_text())
    info = data.get("info", {})
    gender = info.get("gender")
    if gender_filter is not None and gender not in gender_filter:
        return []
    teams = info.get("teams", [])
    players_by_team = info.get("players", {})
    match_id = path.stem

    rows_by_player: dict[str, dict[str, Any]] = {}
    for team, players in players_by_team.items():
        opponent = next((name for name in teams if name != team), "")
        for player_name in players:
            rows_by_player[player_name] = _new_player_row(
                match_id=match_id,
                info=info,
                source_dataset=source_dataset,
                player_name=player_name,
                team=team,
                opponent=opponent,
                player_roles=player_roles,
            )

    dismissed_players: set[str] = set()

    for innings in data.get("innings", []):
        innings_team = innings.get("team", "")
        fielding_team = next((name for name in teams if name != innings_team), "")
        innings_batting_positions: dict[str, int] = {}
        innings_batting_legal_balls = 0
        innings_bowling_legal_balls = 0
        over_conceded = defaultdict(int)
        over_bowler = {}
        for over in innings.get("overs", []):
            over_number = over.get("over")
            for delivery in over.get("deliveries", []):
                batter = delivery["batter"]
                bowler = delivery["bowler"]
                runs = delivery.get("runs", {})
                batter_runs = int(runs.get("batter", 0))

                batter_row = _ensure_player_row(
                    rows_by_player,
                    batter,
                    match_id,
                    info,
                    source_dataset,
                    team=innings_team,
                    opponent=fielding_team,
                    player_roles=player_roles,
                )
                if batter not in innings_batting_positions:
                    innings_batting_positions[batter] = len(innings_batting_positions) + 1
                    batter_row["batting_position"] = innings_batting_positions[batter]
                    batter_row["batting_order_bucket"] = _batting_order_bucket(
                        innings_batting_positions[batter]
                    )
                if not _delivery_is_wide(delivery):
                    innings_batting_legal_balls += 1
                batter_row["runs"] += batter_runs
                if not _delivery_is_wide(delivery):
                    batter_row["balls_faced"] += 1
                if batter_runs == 4:
                    batter_row["fours"] += 1
                elif batter_runs == 6:
                    batter_row["sixes"] += 1

                bowler_row = _ensure_player_row(
                    rows_by_player,
                    bowler,
                    match_id,
                    info,
                    source_dataset,
                    team=fielding_team,
                    opponent=innings_team,
                    player_roles=player_roles,
                )
                conceded = _runs_conceded_for_delivery(delivery)
                bowler_row["runs_conceded"] += conceded
                if not _delivery_is_wide(delivery) and not _delivery_is_no_ball(delivery):
                    bowler_row["balls_bowled"] += 1
                    innings_bowling_legal_balls += 1
                    if over_number is not None:
                        if over_number < 6:
                            bowler_row["powerplay_balls"] += 1
                        elif over_number >= 16:
                            bowler_row["death_balls"] += 1
                        else:
                            bowler_row["middle_balls"] += 1

                if over_number is not None:
                    over_conceded[over_number] += int(runs.get("total", 0))
                    over_bowler[over_number] = bowler

                for wicket_event in delivery.get("wickets", []):
                    player_out = wicket_event.get("player_out")
                    if player_out:
                        dismissed_players.add(player_out)

                    kind = wicket_event.get("kind", "")
                    fielders = _fielder_entries(wicket_event)

                    if kind not in NON_BOWLER_WICKET_KINDS:
                        bowler_row["wickets"] += 1

                    if kind == "caught":
                        for fielder in fielders[:1]:
                            if fielder["substitute"]:
                                continue
                            _ensure_player_row(
                                rows_by_player,
                                fielder["name"],
                                match_id,
                                info,
                                source_dataset,
                                player_roles=player_roles,
                            )["catches"] += 1
                    elif kind == "caught and bowled":
                        bowler_row["catches"] += 1
                    elif kind == "stumped":
                        for fielder in fielders[:1]:
                            if fielder["substitute"]:
                                continue
                            _ensure_player_row(
                                rows_by_player,
                                fielder["name"],
                                match_id,
                                info,
                                source_dataset,
                                player_roles=player_roles,
                            )["stumpings"] += 1
                    elif kind == "run out":
                        if len(fielders) <= 1:
                            for fielder in fielders[:1]:
                                if fielder["substitute"]:
                                    continue
                                _ensure_player_row(
                                    rows_by_player,
                                    fielder["name"],
                                    match_id,
                                    info,
                                    source_dataset,
                                    player_roles=player_roles,
                                )["run_out_direct"] += 1
                        else:
                            for fielder in fielders[:2]:
                                if fielder["substitute"]:
                                    continue
                                _ensure_player_row(
                                    rows_by_player,
                                    fielder["name"],
                                    match_id,
                                    info,
                                    source_dataset,
                                    player_roles=player_roles,
                                )["run_out_assist"] += 1

        for over_number, total_runs in over_conceded.items():
            if total_runs == 0:
                bowler = over_bowler.get(over_number)
                if bowler:
                    rows_by_player[bowler]["maidens"] += 1

        if innings_batting_legal_balls:
            for row in rows_by_player.values():
                if row["team"] == innings_team:
                    row["batting_balls_share"] = round(
                        row["balls_faced"] / innings_batting_legal_balls, 4
                    )
        if innings_bowling_legal_balls:
            for row in rows_by_player.values():
                if row["team"] == fielding_team:
                    row["bowling_balls_share"] = round(
                        row["balls_bowled"] / innings_bowling_legal_balls, 4
                    )
                    row["powerplay_balls_share"] = round(
                        row["powerplay_balls"] / innings_bowling_legal_balls, 4
                    )
                    row["middle_balls_share"] = round(
                        row["middle_balls"] / innings_bowling_legal_balls, 4
                    )
                    row["death_balls_share"] = round(
                        row["death_balls"] / innings_bowling_legal_balls, 4
                    )

    return _finalize_rows(rows_by_player, dismissed_players)


def normalize_dataset_dir(
    dataset_dir: Path | str,
    limit: int | None = None,
    gender_filter: set[str] | None = None,
    player_roles: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    rows: list[dict[str, Any]] = []
    json_files = sorted(dataset_dir.glob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]
    for json_file in json_files:
        rows.extend(
            normalize_match_file(
                json_file,
                source_dataset=dataset_dir.name,
                gender_filter=gender_filter,
                player_roles=player_roles,
            )
        )
    return rows


def write_training_dataset_csv(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELD_ORDER})
