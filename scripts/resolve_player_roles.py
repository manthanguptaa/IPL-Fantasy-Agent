#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ipl_fantasy.player_roles import load_parallel_search_results, resolve_role_from_search_results


SEARCH_DOMAINS = "icc-cricket.com,espncricinfo.com,cricbuzz.com,wikipedia.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve player roles using cached web searches.")
    parser.add_argument("--input", required=True, help="Input player-match CSV.")
    parser.add_argument("--output", required=True, help="Output CSV for role mappings.")
    parser.add_argument(
        "--cache-dir",
        default="tmp/player_role_searches",
        help="Directory for cached parallel-cli search results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of most frequent unresolved players to search this run.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    lowered = value.lower()
    cleaned = "".join(char if char.isalnum() else "-" for char in lowered)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "player"


def _load_player_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "").strip()
            if name:
                counts[name] += 1
    return counts


def _load_existing_roles(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["player_name"]: row for row in rows}


def _search_player(player_name: str, cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return

    objective = (
        f"Find the cricket playing role for {player_name}. "
        "Identify whether the player is a wicketkeeper, batter, all-rounder, or bowler."
    )
    query = f'"{player_name}" cricketer role'
    command = [
        "parallel-cli",
        "search",
        objective,
        "-q",
        query,
        "--max-results",
        "5",
        "--excerpt-max-chars-total",
        "4000",
        "--include-domains",
        SEARCH_DOMAINS,
        "-o",
        str(cache_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0 and not cache_path.exists():
        raise RuntimeError(
            f"Search failed for {player_name}: {result.stderr.strip() or result.stdout.strip()}"
        )


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "player_name",
        "match_count",
        "role",
        "status",
        "source_url",
        "source_title",
        "cache_file",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    player_counts = _load_player_counts(input_path)
    existing_rows = _load_existing_roles(output_path)

    players_to_search = [
        player_name
        for player_name, _ in player_counts.most_common()
        if player_name not in existing_rows
    ][: args.limit]

    resolved_rows = list(existing_rows.values())
    processed = 0

    for player_name in players_to_search:
        cache_name = f"{_slugify(player_name)}.json"
        cache_path = cache_dir / cache_name
        _search_player(player_name, cache_path)
        results = load_parallel_search_results(cache_path)
        resolved = resolve_role_from_search_results(results)

        row = {
            "player_name": player_name,
            "match_count": str(player_counts[player_name]),
            "role": resolved["role"] if resolved else "",
            "status": "resolved" if resolved else "unresolved",
            "source_url": resolved["source_url"] if resolved else "",
            "source_title": resolved["source_title"] if resolved else "",
            "cache_file": str(cache_path),
        }
        resolved_rows.append(row)
        existing_rows[player_name] = row
        processed += 1

    resolved_rows.sort(key=lambda row: (-int(row["match_count"]), row["player_name"]))
    _write_rows(output_path, resolved_rows)

    resolved_count = sum(1 for row in resolved_rows if row["role"])
    unresolved_count = len(player_counts) - resolved_count
    print(
        {
            "processed_this_run": processed,
            "resolved_total": resolved_count,
            "unresolved_total": unresolved_count,
            "unique_players": len(player_counts),
            "output": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
