"""
Approximate home ballpark coordinates (WGS84) for Open-Meteo weather lookups.

Keys are FanGraphs-style abbreviations from ``fg_abbr_from_mlb_name`` (plus common aliases).
"""

from __future__ import annotations

# (latitude, longitude) — good enough for daily weather grids, not surveying.
HOME_FG_BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4453, -112.0667),
    "ATL": (33.7357, -84.3899),
    "BAL": (39.2839, -76.6217),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CHW": (41.8299, -87.6338),
    "CIN": (39.0974, -84.5066),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2778),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9265),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9059, -75.1664),
    "PIT": (40.4469, -80.0057),
    "SDP": (32.7073, -117.1570),
    "SD": (32.7073, -117.1570),
    "SF": (37.7786, -122.3893),
    "SFG": (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325),
    "STL": (38.6226, -90.1928),
    "TB": (27.7682, -82.6534),
    "TBR": (27.7682, -82.6534),
    "TEX": (32.7512, -97.0828),
    "TOR": (43.6414, -79.3894),
    "WSN": (38.8730, -77.0074),
}


def coords_for_home_fg(home_fg: str | None) -> tuple[float, float] | None:
    if not home_fg:
        return None
    key = str(home_fg).strip().upper()
    return HOME_FG_BALLPARK_COORDS.get(key)
