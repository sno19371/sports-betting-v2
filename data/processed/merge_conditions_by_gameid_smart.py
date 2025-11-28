# join_conditions_quadkey.py
# Add game conditions to a player-per-game table by joining on (season, week, home_team, away_team).

import argparse
import logging
import re
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("quadkey_join")

# ---------- team normalization ----------
# Map legacy/alt codes to a single canonical set used for the join
TEAM_NORM = {
    "JAC": "JAX",
    "WSH": "WAS",
    "STL": "LAR",
    "LA":  "LAR",   # if your data sometimes has 'LA' for the Rams
    "SD":  "LAC",
    "OAK": "LV",
}

def norm_team(x: str) -> str:
    s = (str(x) if pd.notna(x) else "").strip().upper()
    return TEAM_NORM.get(s, s)

# ---------- small reducers ----------
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

# ---------- parsing + normalization ----------
def parse_weather(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Pull temp (F) and wind (mph) out of a free-text 'weather' string."""
    if not isinstance(text, str):
        return None, None
    m_temp = re.search(r'(-?\d+)\s*°?\s*[Ff]\b', text)
    m_wind = re.search(r'Wind:\s*[^,]*?(\d+)\s*mph', text, flags=re.IGNORECASE)
    t = int(m_temp.group(1)) if m_temp else None
    w = int(m_wind.group(1)) if m_wind else None
    return t, w

def norm_roof(x: str) -> str:
    s = (str(x) if pd.notna(x) else "").strip().lower()
    if s in {"indoor", "dome"}: return "indoor"
    if "retract" in s and "closed" in s: return "retractable_closed"
    if "retract" in s: return "retractable_open"
    if s in {"outdoor", "open"}: return "outdoor"
    return "unknown"

def norm_surface(x: str) -> str:
    s = (str(x) if pd.notna(x) else "").strip().lower()
    if "turf" in s or "artificial" in s: return "turf"
    if "grass" in s or "natural" in s or "hybrid" in s: return "grass"
    return "unknown"

def safe_fill_cat_unknown(s: pd.Series) -> pd.Series:
    """Fill NaNs with 'unknown' without crashing if it's already a category."""
    if isinstance(s.dtype, CategoricalDtype):
        cats = list(s.cat.categories)
        if "unknown" not in cats:
            s = s.cat.add_categories(["unknown"])
        return s.fillna("unknown")
    else:
        return s.fillna("unknown").astype("category")

# ---------- build per-game env from PBP ----------
def build_games_env_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    req = ["season", "week", "home_team", "away_team"]
    for k in req:
        if k not in pbp.columns:
            raise ValueError(f"PBP is missing required column '{k}'")

    env = pbp.copy()

    # Normalize join keys on PBP side
    env["home_team"] = env["home_team"].map(norm_team)
    env["away_team"] = env["away_team"].map(norm_team)
    env["season"]    = pd.to_numeric(env["season"], errors="coerce").astype("Int16")
    env["week"]      = pd.to_numeric(env["week"], errors="coerce").astype("Int16")

    # Weather fallback parsing
    if "weather" in env.columns:
        temp_w, wind_w = zip(*env["weather"].map(parse_weather))
        env["_temp_from_weather"] = pd.Series(temp_w, index=env.index)
        env["_wind_from_weather"] = pd.Series(wind_w, index=env.index)
    else:
        env["_temp_from_weather"] = np.nan
        env["_wind_from_weather"] = np.nan

    # Aggregate to one row per (season, week, home_team, away_team)
    log.info("Aggregating conditions per (season, week, home_team, away_team)…")
    agg = {
        "roof": _pick if "roof" in env.columns else "first",
        "surface": _pick if "surface" in env.columns else "first",
        "temp": _pick if "temp" in env.columns else "first",
        "wind": _pick if "wind" in env.columns else "first",
        "game_stadium": _pick if "game_stadium" in env.columns else "first",
        "stadium": _pick if "stadium" in env.columns else "first",
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

    # Fill temp/wind from weather text when missing
    games["temp"] = pd.to_numeric(games.get("temp"), errors="coerce")
    games["wind"] = pd.to_numeric(games.get("wind"), errors="coerce")
    games["temp"] = games["temp"].where(games["temp"].notna(), games["_temp_from_weather"])
    games["wind"] = games["wind"].where(games["wind"].notna(), games["_wind_from_weather"])

    # Normalize categories
    games["roof"]    = games.get("roof").map(norm_roof) if "roof" in games.columns else "unknown"
    games["surface"] = games.get("surface").map(norm_surface) if "surface" in games.columns else "unknown"

    # Engineered flags
    games["is_dome"] = games["roof"].isin(["indoor", "retractable_closed"]).astype("int8")
    games["is_turf"] = (games["surface"] == "turf").astype("int8")
    games["temp_clip"] = games["temp"].clip(lower=20, upper=95)
    games["wind_clip"] = games["wind"].clip(lower=0, upper=30)

    # Dtypes
    for c in ["roof", "surface", "game_stadium", "home_team", "away_team"]:
        if c in games.columns:
            games[c] = games[c].astype("category")
    games["temp_clip"] = games["temp_clip"].fillna(0).astype("float32")
    games["wind_clip"] = games["wind_clip"].fillna(0).astype("float32")

    keep = ["season", "week", "home_team", "away_team",
            "roof", "surface", "temp", "wind",
            "temp_clip", "wind_clip", "is_dome", "is_turf", "game_stadium"]
    return games[keep]

# ---------- merge onto player-per-game ----------
def merge_on_quadkey(players: pd.DataFrame, games_env: pd.DataFrame) -> pd.DataFrame:
    req = ["season", "week", "home_team", "away_team"]
    for k in req:
        if k not in players.columns:
            raise ValueError(f"Player table is missing required column '{k}'")

    out = players.copy()

    # Normalize join keys on player side
    out["home_team"] = out["home_team"].map(norm_team)
    out["away_team"] = out["away_team"].map(norm_team)
    out["season"]    = pd.to_numeric(out["season"], errors="coerce").astype("Int16")
    out["week"]      = pd.to_numeric(out["week"], errors="coerce").astype("Int16")

    # Join
    merged = out.merge(
        games_env,
        on=["season", "week", "home_team", "away_team"],
        how="left",
        suffixes=("", "_env"),
    )

    # Coverage diagnostics
    total = len(merged)
    matched = total - merged["is_dome"].isna().sum() if "is_dome" in merged.columns else 0
    cov = 100.0 * matched / total if total else 0.0
    log.info("Join coverage on (season, week, home_team, away_team): %.2f%% (%d / %d rows)", cov, matched, total)
    if cov < 99.0:
        # show a few unmatched keys
        cols = ["season", "week", "home_team", "away_team"]
        miss = merged.loc[merged["is_dome"].isna(), cols].drop_duplicates().head(10)
        if len(miss):
            log.warning("Example unmatched keys (up to 10):\n%s", miss.to_string(index=False))

    # Safe fills
    for c in ["roof", "surface", "game_stadium"]:
        if c in merged.columns:
            merged[c] = safe_fill_cat_unknown(merged[c])
    for c in ["temp_clip", "wind_clip"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype("float32")
    for c in ["is_dome", "is_turf"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0).astype("int8")

    return merged

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Join conditions onto player-per-game by (season, week, home_team, away_team).")
    ap.add_argument("--player", type=str, required=True,
                    help="Path to player-per-game parquet (must include season, week, home_team, away_team)")
    ap.add_argument("--pbp", type=str, required=True,
                    help="Path to PBP parquet or CSV that includes those keys plus roof/surface/temp/wind/game_stadium/weather")
    ap.add_argument("--out", type=str, required=True,
                    help="Output parquet path (e.g., .../player_games_with_conditions.parquet)")
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

    # Inspect PBP season coverage (helps explain any partial joins)
    if "season" in pbp.columns:
        vc = (pd.to_numeric(pbp["season"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index())
        log.info("PBP seasons present (row counts): %s", dict(vc))

    games_env = build_games_env_from_pbp(pbp)
    log.info("Built env for %d unique (season,week,home,away) games.", len(games_env))

    merged = merge_on_quadkey(players, games_env)
    merged.to_parquet(args.out, index=False)
    log.info("Wrote parquet: %s", args.out)

if __name__ == "__main__":
    main()
