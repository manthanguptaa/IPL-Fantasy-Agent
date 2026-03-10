"""Extract batting handedness, bowling style, and all-rounder subtype from cached search results."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Batting handedness patterns
BATTING_RIGHT_PATTERNS = [
    r"batting style\s*[:\n]\s*right[\s-]?hand(ed)?\s*(bat(sman|ter)?)?",
    r"right[\s-]?hand(ed)?\s+bat(sman|ter)?",
    r"right[\s-]?hand\s+bat\b",
    r"\bright[\s-]?handed\s+bat\b",
    r"\|\s*right[\s-]?hand(ed)?\s+bat",
]

BATTING_LEFT_PATTERNS = [
    r"batting style\s*[:\n]\s*left[\s-]?hand(ed)?\s*(bat(sman|ter)?)?",
    r"left[\s-]?hand(ed)?\s+bat(sman|ter)?",
    r"left[\s-]?hand\s+bat\b",
    r"\bleft[\s-]?handed\s+bat\b",
    r"\|\s*left[\s-]?hand(ed)?\s+bat",
]

# Bowling style patterns - ordered from most specific to least
BOWLING_STYLE_PATTERNS = [
    # Fast bowling
    (r"(right|left)[\s-]?arm\s+fast[\s-]?medium", "fast-medium"),
    (r"(right|left)[\s-]?arm\s+medium[\s-]?fast", "medium-fast"),
    (r"(right|left)[\s-]?arm\s+fast\b", "fast"),
    (r"(right|left)[\s-]?arm\s+medium\b", "medium"),
    # Spin bowling
    (r"slow\s+left[\s-]?arm\s+orthodox", "slow-left-arm-orthodox"),
    (r"left[\s-]?arm\s+orthodox", "slow-left-arm-orthodox"),
    (r"left[\s-]?arm\s+wrist[\s-]?spin", "left-arm-wrist-spin"),
    (r"left[\s-]?arm\s+chinaman", "left-arm-wrist-spin"),
    (r"(right|left)[\s-]?arm\s+off[\s-]?spin", "off-spin"),
    (r"(right|left)[\s-]?arm\s+off[\s-]?break", "off-spin"),
    (r"leg[\s-]?spin", "leg-spin"),
    (r"leg[\s-]?break", "leg-spin"),
    (r"wrist[\s-]?spin", "leg-spin"),
    (r"off[\s-]?spin", "off-spin"),
    (r"off[\s-]?break", "off-spin"),
    # Generic
    (r"(right|left)[\s-]?arm\s+spin", "spin"),
    (r"\bspin(ner)?\b", "spin"),
    (r"\bpacer\b", "fast"),
    (r"\bpace\s+bowler\b", "fast"),
    (r"\bfast\s+bowler\b", "fast"),
    (r"\bseam(er)?\b", "medium"),
]

# Bowling arm patterns
BOWLING_ARM_PATTERNS = [
    (r"right[\s-]?arm", "right-arm"),
    (r"left[\s-]?arm", "left-arm"),
]


def _load_search_results(path: Path) -> list[dict[str, Any]]:
    """Load and parse cached search results."""
    if not path.exists():
        return []
    try:
        raw_text = path.read_text()
        # Handle truncated JSON (warnings section may be incomplete)
        warning_idx = raw_text.find('\n  "warnings"')
        if warning_idx != -1:
            prefix = raw_text[:warning_idx]
            if prefix.endswith(","):
                prefix = prefix[:-1]
            if prefix.endswith("\n  ],"):
                prefix = prefix[:-4] + "\n  ]"
            raw_text = prefix + "\n}\n"
        payload = json.loads(raw_text)
        results = payload.get("results", [])
        return [r for r in results if isinstance(r, dict)]
    except (json.JSONDecodeError, Exception):
        return []


def _get_content(results: list[dict[str, Any]]) -> str:
    """Extract all text content from search results."""
    parts = []
    for result in results:
        parts.append(str(result.get("title", "")))
        excerpts = result.get("excerpts", [])
        if isinstance(excerpts, list):
            parts.extend(str(e) for e in excerpts)
    return "\n".join(parts).lower()


def extract_batting_hand(content: str) -> str:
    """Extract batting handedness from content."""
    for pattern in BATTING_LEFT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return "left"
    for pattern in BATTING_RIGHT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return "right"
    return ""


def extract_bowling_style(content: str) -> tuple[str, str]:
    """Extract bowling arm and style from content.

    Returns (arm, style) tuple like ("right-arm", "fast") or ("left-arm", "off-spin").
    """
    arm = ""
    style = ""

    # First try to find arm
    for pattern, arm_value in BOWLING_ARM_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            arm = arm_value
            break

    # Then try to find style
    for pattern, style_value in BOWLING_STYLE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            style = style_value
            # If we found arm in the pattern, extract it
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                matched_text = match.group(0).lower()
                if "left" in matched_text:
                    arm = "left-arm"
                elif "right" in matched_text:
                    arm = "right-arm"
            break

    return arm, style


def determine_ar_subtype(content: str, batting_hand: str, bowling_arm: str, bowling_style: str) -> str:
    """Determine if an all-rounder is batting or bowling focused.

    Returns "batting-ar" or "bowling-ar" or "" if unknown.
    """
    # Look for explicit mentions
    if re.search(r"batting\s+all[\s-]?rounder", content, re.IGNORECASE):
        return "batting-ar"
    if re.search(r"bowling\s+all[\s-]?rounder", content, re.IGNORECASE):
        return "bowling-ar"

    # Infer from bowling style - if they have a clear bowling style, likely bowling AR
    # If they're a spinner or fast bowler with clear style, more bowling focused
    if bowling_style in ("fast", "medium-fast", "fast-medium", "leg-spin", "off-spin", "slow-left-arm-orthodox", "left-arm-wrist-spin"):
        return "bowling-ar"

    # If they have batting hand but unclear bowling, more batting focused
    if batting_hand and not bowling_style:
        return "batting-ar"

    return ""


def extract_player_details(cache_path: Path) -> dict[str, str]:
    """Extract all player details from cached search results.

    Returns dict with keys: batting_hand, bowling_arm, bowling_style, ar_subtype
    """
    results = _load_search_results(cache_path)
    if not results:
        return {"batting_hand": "", "bowling_arm": "", "bowling_style": "", "ar_subtype": ""}

    content = _get_content(results)

    batting_hand = extract_batting_hand(content)
    bowling_arm, bowling_style = extract_bowling_style(content)
    ar_subtype = determine_ar_subtype(content, batting_hand, bowling_arm, bowling_style)

    return {
        "batting_hand": batting_hand,
        "bowling_arm": bowling_arm,
        "bowling_style": bowling_style,
        "ar_subtype": ar_subtype,
    }


def process_all_players(cache_dir: Path, player_roles_path: Path, output_path: Path) -> dict[str, int]:
    """Process all players and extract details.

    Returns stats dict with counts.
    """
    import csv

    # Load existing player roles
    players = []
    with player_roles_path.open(newline="") as f:
        reader = csv.DictReader(f)
        players = list(reader)

    # Add new columns
    for player in players:
        player["batting_hand"] = ""
        player["bowling_arm"] = ""
        player["bowling_style"] = ""
        player["ar_subtype"] = ""

    # Process each player
    stats = {
        "total": len(players),
        "batting_hand_found": 0,
        "bowling_style_found": 0,
        "ar_subtype_found": 0,
    }

    for player in players:
        name = player["player_name"]
        cache_file = player.get("cache_file", "")

        if cache_file:
            cache_path = Path(cache_file)
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_file

            details = extract_player_details(cache_path)
            player["batting_hand"] = details["batting_hand"]
            player["bowling_arm"] = details["bowling_arm"]
            player["bowling_style"] = details["bowling_style"]

            # Only set AR subtype for all-rounders
            if player.get("role") == "AR":
                player["ar_subtype"] = details["ar_subtype"]

            if details["batting_hand"]:
                stats["batting_hand_found"] += 1
            if details["bowling_style"]:
                stats["bowling_style_found"] += 1
            if details["ar_subtype"] and player.get("role") == "AR":
                stats["ar_subtype_found"] += 1

    # Write output
    fieldnames = [
        "player_name", "match_count", "role", "status",
        "batting_hand", "bowling_arm", "bowling_style", "ar_subtype",
        "source_url", "source_title", "cache_file"
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for player in players:
            writer.writerow({k: player.get(k, "") for k in fieldnames})

    return stats


if __name__ == "__main__":
    import sys
    cache_dir = Path("tmp/player_role_searches")
    player_roles_path = Path("data/player_roles.csv")
    output_path = Path("data/player_roles_detailed.csv")

    stats = process_all_players(cache_dir, player_roles_path, output_path)
    print(f"Processed {stats['total']} players")
    print(f"  Batting hand found: {stats['batting_hand_found']}")
    print(f"  Bowling style found: {stats['bowling_style_found']}")
    print(f"  AR subtype found: {stats['ar_subtype_found']}")
    print(f"Output: {output_path}")
