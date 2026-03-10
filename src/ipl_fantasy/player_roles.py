from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROLE_PRIORITY = ("WK", "AR", "BOWL", "BAT")

DOMAIN_PRIORITY = (
    "icc-cricket.com",
    "espncricinfo.com",
    "cricbuzz.com",
    "wikipedia.org",
)

URL_EXCLUDE_PATTERNS = (
    "match-squads",
    "/series/",
    "/records/",
    "/news/",
)

WK_PATTERNS = (
    r"(playing role|role)\s*[:.\-\n ]+\s*(wicket[\s-]?keeper|keeper[\s-]?(batter|batsman))",
    r"wicket[\s-]?keeper",
    r"keeper[\s-]?batter",
    r"keeper[\s-]?batsman",
)
AR_PATTERNS = (
    r"(playing role|role)\s*[:.\-\n ]+\s*((batting|bowling)\s+)?all[\s-]?rounder",
    r"\|\s*((batting|bowling)\s+)?all[\s-]?rounder",
)
BOWL_PATTERNS = (
    r"playing role\s*\n\s*bowler\b",
    r"\brole\s*\n\s*bowler\b",
    r"playing role[^.\n]*bowler",
    r"\brole[^.\n]*bowler",
    r"\brole[^.\n]*spinner",
    r"\brole[^.\n]*pacer",
    r"\|\s*bowler\b",
    r"\|\s*(left|right)[^|\n]*bowler",
    r"\|\s*spin bowler",
    r"\|\s*fast bowler",
    r"\|\s*pace bowler",
    r"\bfast bowler\b",
    r"\bpace bowler\b",
    r"\bspin bowler\b",
    r"\bleg[\s-]?spinner\b",
    r"\boff[\s-]?spinner\b",
    # Bowling style patterns from player profiles
    r"(left|right)[\s-]arm\s+(fast|medium|slow)[\s-]?(fast|medium)?",
    r"(left|right)[\s-]arm\s+(orthodox|chinaman|wrist[\s-]?spin|off[\s-]?spin|leg[\s-]?spin|spin)",
    r"slow\s+(left|right)[\s-]arm\s+orthodox",
    r"\bslow left[\s-]?arm\b",
    r"\bleft[\s-]?arm wrist[\s-]?spin\b",
    r"\bleg[\s-]?break\b",
    r"\boff[\s-]?break\b",
)
BAT_PATTERNS = (
    r"playing role\s*\n\s*(batter|batsman)\b",
    r"\brole\s*\n\s*(batter|batsman)\b",
    r"playing role[^.\n]*batter",
    r"playing role[^.\n]*batsman",
    r"\brole[^.\n]*batter",
    r"\brole[^.\n]*batsman",
    r"\|\s*(top|middle|opening)[^|\n]*(batter|batsman)",
    r"\|\s*batter\b",
    r"\|\s*batsman\b",
    r"\btop order batter\b",
    r"\bmiddle order batter\b",
    r"\bopening batter\b",
    r"\bbatter\b",
    r"\bbatsman\b",
)


def _truncate_parallel_output(text: str) -> str:
    warning_idx = text.find('\n  "warnings"')
    if warning_idx == -1:
        return text
    prefix = text[:warning_idx]
    if prefix.endswith(","):
        prefix = prefix[:-1]
    if prefix.endswith("\n  ],"):
        prefix = prefix[:-4] + "\n  ]"
    return prefix + "\n}\n"


def load_parallel_search_results(path: Path | str) -> list[dict[str, Any]]:
    raw_text = Path(path).read_text()
    payload = json.loads(_truncate_parallel_output(raw_text))
    results = payload.get("results", [])
    return [result for result in results if isinstance(result, dict)]


def infer_role_from_text(text: str) -> str | None:
    normalized = text.lower()

    explicit_patterns = (
        ("WK", WK_PATTERNS),
        ("AR", AR_PATTERNS),
        ("BAT", BAT_PATTERNS[:8]),
        ("BOWL", BOWL_PATTERNS[:8]),
    )
    for role, patterns in explicit_patterns:
        for pattern in patterns:
            if re.search(pattern, normalized):
                return role

    for pattern in WK_PATTERNS:
        if re.search(pattern, normalized):
            return "WK"
    for pattern in AR_PATTERNS:
        if re.search(pattern, normalized):
            return "AR"
    for pattern in BOWL_PATTERNS:
        if re.search(pattern, normalized):
            return "BOWL"
    for pattern in BAT_PATTERNS[8:]:
        if re.search(pattern, normalized):
            return "BAT"
    return None


def resolve_role_from_search_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked_matches: list[tuple[int, int, str, dict[str, Any]]] = []
    for idx, result in enumerate(results):
        url = str(result.get("url", ""))
        if any(pattern in url for pattern in URL_EXCLUDE_PATTERNS):
            continue
        title = str(result.get("title", ""))
        excerpts = result.get("excerpts", [])
        content_parts = [title]
        if isinstance(excerpts, list):
            content_parts.extend(str(excerpt) for excerpt in excerpts)
        content = "\n".join(content_parts)
        role = infer_role_from_text(content)
        if role is None:
            continue

        domain_rank = len(DOMAIN_PRIORITY)
        for domain_idx, domain in enumerate(DOMAIN_PRIORITY):
            if domain in url:
                domain_rank = domain_idx
                break
        role_rank = ROLE_PRIORITY.index(role)
        ranked_matches.append((domain_rank, role_rank, role, result))

    if not ranked_matches:
        return None

    ranked_matches.sort(key=lambda item: (item[0], item[1]))
    _, _, role, result = ranked_matches[0]
    return {
        "role": role,
        "source_url": str(result.get("url", "")).replace("\n", " ").strip(),
        "source_title": str(result.get("title", "")).replace("\n", " ").strip(),
    }
