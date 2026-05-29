"""
Approximate park factors (HR and runs) keyed by MLB Stats API venue_id when known,
else by common team abbreviation. Values near 100 = neutral FanGraphs-style index.

For personal use: static table you can edit as seasons change.
Sources approximate league-published indices; refine over time.
"""

from __future__ import annotations

# venue_id -> (hr_factor_100_scale, runs_factor_100_scale). Extend with real MLB venue IDs as you like.
VENUE_FACTORS: dict[int, tuple[float, float]] = {}

# Team abbreviation (FG-style) -> (hr_factor, runs_factor)
TEAM_PARK_DEFAULT: dict[str, tuple[float, float]] = {
    "NYY": (110, 102),
    "COL": (118, 115),
    "BOS": (105, 104),
    "CHC": (103, 102),
    "SDP": (96, 94),
    "SD": (96, 94),
    "SEA": (95, 94),
    "SF": (95, 93),
    "SFG": (95, 93),
    "LAD": (102, 99),
    "ATL": (100, 101),
    "HOU": (103, 102),
    "TEX": (104, 103),
    "PHI": (108, 104),
    "CIN": (118, 106),
    "BAL": (112, 103),
    "TOR": (104, 103),
    "MIN": (99, 98),
    "MIL": (104, 102),
    "STL": (97, 98),
    "CHW": (110, 104),
    "CLE": (101, 99),
    "DET": (100, 99),
    "KC": (96, 97),
    "OAK": (94, 93),
    "LAA": (103, 102),
    "ARI": (104, 103),
    "MIA": (94, 93),
    "NYM": (99, 98),
    "WSN": (102, 101),
    "PIT": (97, 96),
    "TB": (96, 95),
    "TBR": (96, 95),
}


def park_factors_for_team(team_abbr: str) -> tuple[float, float]:
    """Return (hr_factor, runs_factor) on ~100 neutral scale."""
    key = team_abbr.strip().upper()
    return TEAM_PARK_DEFAULT.get(key, (100.0, 100.0))


def park_factors_for_venue(venue_id: int | None, team_abbr_fallback: str) -> tuple[float, float]:
    if venue_id and venue_id in VENUE_FACTORS:
        return VENUE_FACTORS[venue_id]
    return park_factors_for_team(team_abbr_fallback)
