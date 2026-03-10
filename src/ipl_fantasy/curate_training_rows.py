from __future__ import annotations

from typing import Iterable


RELEVANT_T20I_TEAMS = {
    "India",
    "England",
    "Australia",
    "New Zealand",
    "South Africa",
    "Pakistan",
    "Sri Lanka",
    "West Indies",
    "Bangladesh",
    "Zimbabwe",
    "Ireland",
    "Scotland",
    "Netherlands",
    "Namibia",
    "Nepal",
    "Oman",
    "United States of America",
    "United Arab Emirates",
    "Canada",
}


def filter_rows_for_ipl_model(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        source_dataset = row.get("source_dataset", "")
        if source_dataset != "t20s_json":
            filtered.append(row)
            continue

        team = row.get("team", "")
        opponent = row.get("opponent", "")
        if team in RELEVANT_T20I_TEAMS and opponent in RELEVANT_T20I_TEAMS:
            filtered.append(row)

    return filtered
