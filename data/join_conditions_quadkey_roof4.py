# join_conditions_quadkey_roof4.py
# Join PBP conditions onto player-per-game by (season, week, home_team, away_team)
# and FORCE roof ∈ {"dome","closed","open","outdoors"} with no NULLs.

import argparse
import logging
import re
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("quadkey_join_roof4")

TEAM_NORM = {
    "JAC": "JAX",
    "WSH": "WAS",
    "STL": "LAR",
    "LA":  "LAR",
    "SD":  "LAC",
    "OAK": "LV",
}
def norm_team(x: str) -> str:
    s = (str(x) if pd.notna(x) else "").strip().upper()
    return TEAM_NORM.get(s, s)

def _mode_ignore_na(s: pd.Series):
    vals = [x for x in s.dropna().tolist() if str(x) != ""]
    return Counter(vals).most_common(1)[0][0] if vals else None

def _last_non_null(s: pd.Series):
    for x in reversed(s.tolist()):
        if pd.notna(x) and str(x) != "":
            return x
    return None

def _pick(s: pd.Series):
    return _mode_ignore_na(s) or _last_non_null(s)

# --- Weather parsing for temp/wind fallback ---
def parse_weather(text: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(text, str) or not text.strip():
        return None, None
    s = text.strip()
    mT = re.search(r'(?:temp(?:erature)?[:\s-]*)?(-?\d{1,3})\s*[°º]?\s*[Ff]\b', s)
    # allow "Wind: NE 9 mph" or just "9 mph"
    mW = (re.search(r'wind[:\s-]*[A-Z]{0,3}\s*(\d{1,3})\s*mph', s, flags=re.IGNORECASE)
          or re.search(r'(\d{1,3})\s*mph', s, flags=re.IGNORECASE))
    temp_f = int(mT.group(1)) if mT else None
    wind_mph = int(mW.group(1)) if mW else None
    return temp_f, wind_mph

# --- ROOF to the 4 canonical values ---
ROOF_CATS = ["dome", "closed", "open", "outdoors"]
ROOF_DTYPE = CategoricalDtype(categories=ROOF_CATS, ordered=False)

def roof_to_four(roof_val: Optional[str], weather_val: Optional[str]) -> str:
    """
    Map messy roof strings to one of: dome, closed, open, outdoors.
    Never returns NULL.
    """
    s = (str(roof_val).strip().lower() if roof_val is not None else "")
    if s in {"dome", "indoor"}:
        return "dome"
    if "retract" in s:
        if "open" in s:
            return "open"
        if "clos" in s:
            return "closed"
        # retractable but unspecified -> assume closed to be conservative
        return "closed"
    if s == "closed":
        return "closed"
    if s == "open":
        return "open"
    if s in {"outdoor", "outdoors"}:
        return "outdoors"

    # Fallback via weather text
    w = (str(weather_val).lower() if isinstance(weather_val, str) else "")
    if "indoor" in w or "indoors" in w:
        return "dome"
    # If we can parse a wind speed from weather, assume outdoors/open
    _, wmph = parse_weather(weather_val) if isinstance(weather_val, str) else (None, None)
    if wmph is not None:
        return "outdoors"

    # Final fallback: call it outdoors
    return "outdoors"

def safe_fill_cat_unknown(s: pd.Series) -> pd.Series:
    # we won't use "unknown" for roof, but keep helper for other fields
    if isinstance(s.dtype, CategoricalDtype):
        if "unknown" not in s.cat.categories:
            s = s.cat.add_categories(["unknown"])
        return s.fillna("unknown")
    else:
        return s.fillna("unknown").astype("category")

def build_games_env_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    req = ["season", "week", "home_team", "away_team"]
    for k in req:
        if k not in pbp.columns:
            raise ValueError(f"PBP is missing required column '{k}'")

    env = pbp.copy()
    env["home_team"] = env["home_team"].map(norm_team)
    env["away_team"] = env["away_team"].map(norm_team)
    env["season"]    = pd.to_numeric(env["season"], errors="coerce").astype("Int16")
    env["week"]      = pd.to_numeric(env["week"], errors="coerce").astype("Int16")

    # Prepare weather parse columns
    if "weather" in env.columns:
        temp_w, wind_w = zip(*env["weather"].map(parse_weather))
        env["_temp_from_weather"] = pd.Series(temp_w, index=env.index)
        env["_wind_from_weather"] = pd.Series(wind_w, index=env.index)
    else:
        env["_temp_from_weather"] = np.nan
        env["_wind_from_weather"] = np.nan
        env["weather"] = None

    log.info("Aggregating conditions per (season, week, home_team, away_team)…")
    agg = {
        "roof": _pick if "roof" in env.columns else "first",
        "surface": _pick if "surface" in env.columns else "first",
        "temp": _pick if "temp" in env.columns else "first",
        "wind": _pick if "wind" in env.columns else "first",
        "game_stadium": _pick if "game_stadium" in env.columns else "first",
        "stadium": _pick if "stadium" in env.columns else "first",
        "weather": _pick,
        "_temp_from_weather": _pick,
        "_wind_from_weather": _pick,
    }
    games = (env
             .groupby(["season", "week", "home_team", "away_team"], dropna=False)
             .agg(agg)
             .reset_index())

    # Prefer game_stadium; fallback to stadium
    if "game_stadium" not in games.columns or games["game_stadium"].isna().all():
        games["game_stadium"] = games.get("stadium")
    games["game_stadium"] = games["game_stadium"].fillna("unknown").astype(str)

    # Numeric temp/wind with weather fallback
    games["temp"] = pd.to_numeric(games.get("temp"), errors="coerce")
    games["wind"] = pd.to_numeric(games.get("wind"), errors="coerce")
    games["temp"] = games["temp"].where(games["temp"].notna(), games["_temp_from_weather"])
    games["wind"] = games["wind"].where(games["wind"].notna(), games["_wind_from_weather"])

    # *** FORCE roof to the 4 categories ***
    games["roof"] = games.apply(
        lambda r: roof_to_four(r.get("roof"), r.get("weather")),
        axis=1
    ).astype(ROOF_DTYPE)

    # Surface – leave as-is but normalize a bit
    def norm_surface(x: str) -> str:
        s = (str(x) if pd.notna(x) else "").strip().lower()
        if "turf" in s or "artificial" in s: return "turf"
        if "grass" in s or "natural" in s or "hybrid" in s: return "grass"
        return "unknown"
    games["surface"] = games.get("surface").map(norm_surface).astype("category")

    # Engineered flags & clips
    games["is_dome"] = (games["roof"] == "dome").astype("int8")
    games["is_turf"] = (games["surface"] == "turf").astype("int8")
    games["temp_clip"] = games["temp"].clip(lower=20, upper=95)
    games["wind_clip"] = games["wind"].clip(lower=0, upper=30)
    games["temp_clip"] = games["temp_clip"].fillna(70).astype("float32")
    games["wind_clip"] = games["wind_clip"].fillna(0).astype("float32")

    keep = ["season", "week", "home_team", "away_team",
            "roof", "surface", "temp", "wind",
            "temp_clip", "wind_clip", "is_dome", "is_turf", "game_stadium"]
    return games[keep]

def merge_on_quadkey(players: pd.DataFrame, games_env: pd.DataFrame) -> pd.DataFrame:
    req = ["season", "week", "home_team", "away_team"]
    for k in req:
        if k not in players.columns:
            raise ValueError(f"Player table is missing required column '{k}'")

    out = players.copy()
    out["home_team"] = out["home_team"].map(norm_team)
    out["away_team"] = out["away_team"].map(norm_team)
    out["season"]    = pd.to_numeric(out["season"], errors="coerce").astype("Int16")
    out["week"]      = pd.to_numeric(out["week"], errors="coerce").astype("Int16")

    merged = out.merge(
        games_env,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        suffixes=("", "_env"),
    )

    total = len(merged)
    matched = total - merged["roof"].isna().sum()
    cov = 100.0 * matched / total if total else 0.0
    log.info("Join coverage: %.2f%% (%d / %d rows)", cov, matched, total)

    # Ensure roof has NO NULLs by forcing category and filling with outdoors
    merged["roof"] = merged["roof"].astype(ROOF_DTYPE)
    merged["roof"] = merged["roof"].cat.add_categories(["outdoors"]) if "outdoors" not in merged["roof"].cat.categories else merged["roof"]
    merged["roof"] = merged["roof"].fillna("outdoors").astype(ROOF_DTYPE)

    # Fill other fields safely
    for c in ["surface", "game_stadium"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna("unknown").astype("category")
    for c in ["temp_clip", "wind_clip"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype("float32")
    for c in ["is_dome", "is_turf"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype("int8")

    # Quick distribution print
    dist = merged["roof"].value_counts(dropna=False).to_dict()
    log.info("Roof distribution: %s", dist)

    return merged

def main():
    ap = argparse.ArgumentParser(description="Join conditions with 4-way roof normalization (no NULLs).")
    ap.add_argument("--player", type=str, required=True,
                    help="Path to player-per-game parquet (needs season, week, home_team, away_team)")
    ap.add_argument("--pbp", type=str, required=True,
                    help="Path to PBP parquet/CSV with season/week/home/away + roof/surface/temp/wind/game_stadium/weather")
    ap.add_argument("--out", type=str, required=True,
                    help="Output parquet path")
    args = ap.parse_args()

    log.info("Loading player-per-game: %s", args.player)
    players = pd.read_parquet(args.player)

    log.info("Loading PBP: %s", args.pbp)
    if args.pbp.lower().endswith(".csv"):
        keep = {"season","week","home_team","away_team","roof","surface","temp","wind","game_stadium","stadium","weather"}
        pbp = pd.read_csv(args.pbp, usecols=lambda c: c in keep)
    else:
        try:
            pbp = pd.read_parquet(args.pbp, columns=["season","week","home_team","away_team","roof","surface","temp","wind","game_stadium","stadium","weather"])
        except Exception:
            pbp = pd.read_parquet(args.pbp)

    if "season" in pbp.columns:
        vc = (pd.to_numeric(pbp["season"], errors="coerce").dropna().astype(int).value_counts().sort_index())
        log.info("PBP seasons present: %s", dict(vc))

    games_env = build_games_env_from_pbp(pbp)
    log.info("Built env for %d unique (season,week,home,away) games.", len(games_env))

    merged = merge_on_quadkey(players, games_env)
    merged.to_parquet(args.out, index=False)
    log.info("Wrote parquet: %s", args.out)

if __name__ == "__main__":
    main()
