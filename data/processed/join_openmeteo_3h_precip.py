# join_openmeteo_3h_precip.py
#
# Uses Open-Meteo ERA5 hourly archive to fetch *actual* weather around kickoff
# for each game, then joins a small, clean set of features onto your
# player-per-game table:
#
#   roof           (dome/closed/open/outdoors)
#   surface        (grass/turf/unknown)
#   is_dome        (0/1)
#   temp_effective (°F, cleaned)
#   wind_effective (mph, cleaned, 0 indoors)
#   precip_3h_mm   (total mm precip in [kickoff, kickoff+3h) )
#
# Inputs:
#   --locations  locations.csv (homeTeamAbbr,roof,surface,lat,long)
#   --pbp        pbp_combined.parquet
#   --players    full_games_with_tds_ints_fumbles_yards.parquet
#   --out        output parquet path
#   --cache      optional JSON cache for Open-Meteo responses

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dateutil import parser as dtparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("openmeteo_3h")

# ------------ CONFIG ------------ #

# Variables we ask Open-Meteo for (only what we need)
HOURLY_FIELDS = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
]

# Team code normalization so everything agrees
TEAM_NORM = {
    "JAC": "JAX",
    "WSH": "WAS",
    "STL": "LAR",
    "LA":  "LAR",
    "SD":  "LAC",
    "OAK": "LV",
}


def norm_team(x: str) -> str:
    s = (str(x) if x is not None else "").strip().upper()
    return TEAM_NORM.get(s, s)


# ------------ ROOF / SURFACE NORMALIZATION ------------ #

def roof_from_csv(csv_roof: str) -> str:
    """
    Map locations.csv roof values (yes/no/retractable/etc) to:
    dome / closed / open / outdoors
    """
    s = (str(csv_roof) if csv_roof is not None else "").strip().lower()
    if s in {"yes", "dome", "indoor"}:
        return "dome"
    if s == "retractable":
        # Without per-game open/close, treat retractable as closed by default
        return "closed"
    if s in {"no", "outdoor", "outdoors"}:
        return "outdoors"
    return "outdoors"  # safe default


def surface_from_csv(s: str) -> str:
    s = (str(s) if s is not None else "").strip().lower()
    if "turf" in s or "artificial" in s:
        return "turf"
    if "grass" in s or "hybrid" in s or "natural" in s:
        return "grass"
    return "unknown"


# ------------ TIME HANDLING ------------ #

def parse_kick_local(game_date: str, time_of_day: str) -> datetime:
    """
    Build a naive 'local' datetime from game_date + time_of_day.
    If time_of_day is missing/weird, default to 13:00 (1 PM).
    """
    try:
        d = dtparse.parse(str(game_date)).date()
    except Exception:
        d = pd.to_datetime(str(game_date), errors="coerce").date()
    if pd.isna(d):
        raise ValueError(f"Unparseable game_date: {game_date}")

    if time_of_day and str(time_of_day).strip():
        try:
            t = dtparse.parse(str(time_of_day)).time()
        except Exception:
            # Sometimes it's like "9/7/25, 13:02:38"; drop the date part
            tok = str(time_of_day).split(",")[-1].strip()
            try:
                t = dtparse.parse(tok).time()
            except Exception:
                t = datetime.strptime("13:00:00", "%H:%M:%S").time()
    else:
        t = datetime.strptime("13:00:00", "%H:%M:%S").time()

    return datetime.combine(d, t)


def round_to_nearest_hour(dt: datetime) -> datetime:
    """Round naive datetime to nearest hour (ties up)."""
    if dt.minute >= 30:
        dt = dt + timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)


# ------------ OPEN-METEO FETCH ------------ #

def fetch_hourly_one_day(lat: float, lon: float, date_str: str, session: requests.Session) -> dict:
    """
    Call Open-Meteo ERA5 for one day's hourly data at given lat/lon.
    timezone=auto -> times in local stadium time.
    """
    base = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "auto",
        "timeformat": "iso8601",
    }
    for attempt in range(4):
        try:
            r = session.get(base, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("Open-Meteo fetch failed (%s). Retry %d", e, attempt + 1)
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch Open-Meteo for {lat},{lon} {date_str}")


def extract_hour_and_3h(json_obj: dict, local_hour: datetime) -> dict:
    """
    From Open-Meteo JSON and desired local_hour, pick:
      - temp_c, wind_ms at kickoff hour (nearest hour)
      - precip_mm at kickoff hour
      - precip_3h_mm = sum precip over [kickoff, kickoff+3h)
    """
    hourly = json_obj.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {"temp_c": np.nan, "wind_ms": np.nan, "precip_mm": np.nan, "precip_3h_mm": np.nan}

    time_index = pd.to_datetime(times)
    t_local = pd.to_datetime(local_hour)

    # index for kickoff hour: closest time to t_local
    diffs = np.abs((time_index - t_local).total_seconds())
    i0 = int(diffs.argmin())

    # helper to get a field array as float np.array
    def arr(field):
        vals = hourly.get(field, [])
        if not vals:
            return np.full(len(time_index), np.nan, dtype=float)
        return np.array(pd.to_numeric(vals, errors="coerce"), dtype=float)

    temp_arr = arr("temperature_2m")
    wind_arr = arr("wind_speed_10m")
    precip_arr = arr("precipitation")

    temp_c = temp_arr[i0] if not np.isnan(temp_arr[i0]) else np.nan
    wind_ms = wind_arr[i0] if not np.isnan(wind_arr[i0]) else np.nan
    precip_mm = precip_arr[i0] if not np.isnan(precip_arr[i0]) else np.nan

    # 3-hour window: [kickoff, kickoff+3h)
    t_end = t_local + timedelta(hours=3)
    mask = (time_index >= t_local) & (time_index < t_end)
    if mask.any():
        precip_3h = float(np.nansum(precip_arr[mask]))
    else:
        precip_3h = np.nan

    return {
        "temp_c": temp_c,
        "wind_ms": wind_ms,
        "precip_mm": precip_mm,
        "precip_3h_mm": precip_3h,
    }


# ------------ UTILS ------------ #

def pick_mode(series: pd.Series):
    vals = [v for v in series.dropna().tolist() if str(v).strip() != ""]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


# ------------ MAIN ------------ #

def main():
    ap = argparse.ArgumentParser(description="Join Open-Meteo kickoff weather (with 3h precip) to player-per-game.")
    ap.add_argument("--locations", required=True, help="locations.csv with homeTeamAbbr,roof,surface,lat,long")
    ap.add_argument("--pbp", required=True, help="pbp_combined.parquet")
    ap.add_argument("--players", required=True, help="full_games_with_tds_ints_fumbles_yards.parquet")
    ap.add_argument("--out", required=True, help="Output parquet for players+weather")
    ap.add_argument("--cache", default="openmeteo_cache_3h.json", help="JSON cache for Open-Meteo calls")
    args = ap.parse_args()

    # 1) Locations (stadium metadata)
    log.info("Loading locations: %s", args.locations)
    loc = pd.read_csv(args.locations)
    loc["homeTeamAbbr"] = loc["homeTeamAbbr"].map(norm_team)
    loc = loc.rename(columns={"homeTeamAbbr": "home_team", "long": "lon"})

    loc["lat"] = pd.to_numeric(loc["lat"], errors="coerce")
    loc["lon"] = pd.to_numeric(loc["lon"], errors="coerce")
    loc["roof"] = loc["roof"].apply(roof_from_csv)
    loc["surface"] = loc["surface"].apply(surface_from_csv)

    log.info("Locations loaded for %d unique home teams.", loc["home_team"].nunique())

    # 2) PBP -> one row per game with kickoff time
    log.info("Loading PBP: %s", args.pbp)
    pbp = pd.read_parquet(args.pbp)
    need_cols = ["season", "week", "home_team", "away_team", "game_date", "time_of_day"]
    for c in need_cols:
        if c not in pbp.columns:
            raise ValueError(f"PBP missing required column: {c}")

    games = pbp[need_cols].copy()
    games["home_team"] = games["home_team"].map(norm_team)
    games["away_team"] = games["away_team"].map(norm_team)

    games = (
        games.groupby(["season", "week", "home_team", "away_team", "game_date"], dropna=False)
        .agg({"time_of_day": pick_mode})
        .reset_index()
    )

    # Attach stadium lat/lon + roof/surface
    games = games.merge(
        loc[["home_team", "lat", "lon", "roof", "surface"]],
        on="home_team",
        how="left",
    )

    missing_geo = games["lat"].isna().sum()
    if missing_geo:
        log.warning("WARNING: %d games missing lat/lon after location join.", missing_geo)

    # 3) Compute kickoff local time (rounded to hour)
    games["kick_local"] = games.apply(
        lambda r: round_to_nearest_hour(parse_kick_local(r["game_date"], r["time_of_day"])),
        axis=1,
    )
    games["game_date_str"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

    # 4) Fetch Open-Meteo for each (home_team, game_date)
    cache = {}
    if args.cache and os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache, "r"))
        except Exception:
            cache = {}

    session = requests.Session()
    weather_rows = []

    keys = games[["home_team", "game_date_str", "lat", "lon"]].drop_duplicates()
    log.info("Unique (home_team, game_date) keys to fetch: %d", len(keys))

    for _, row in keys.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = float(row["lat"]) if not pd.isna(row["lat"]) else None
        lon = float(row["lon"]) if not pd.isna(row["lon"]) else None

        if lat is None or lon is None:
            weather_rows.append({"home_team": ht, "game_date_str": gd, "wx_json": None})
            continue

        cache_key = f"{ht}:{gd}:{lat:.4f},{lon:.4f}"
        if cache_key in cache:
            wx_json = cache[cache_key]
        else:
            wx_json = fetch_hourly_one_day(lat, lon, gd, session)
            cache[cache_key] = wx_json
            time.sleep(0.15)  # be nice

        weather_rows.append({"home_team": ht, "game_date_str": gd, "wx_json": wx_json})

    if args.cache:
        try:
            json.dump(cache, open(args.cache, "w"))
        except Exception as e:
            log.warning("Could not write cache: %s", e)

    wx_df = pd.DataFrame(weather_rows)

    # Attach wx_json to each game and extract features around kickoff
    games = games.merge(wx_df, on=["home_team", "game_date_str"], how="left")

    def pick_hour_features(row):
        if not isinstance(row["wx_json"], dict) or row["wx_json"] is None:
            return {"temp_c": np.nan, "wind_ms": np.nan, "precip_mm": np.nan, "precip_3h_mm": np.nan}
        return extract_hour_and_3h(row["wx_json"], row["kick_local"])

    hour_df = games.apply(pick_hour_features, axis=1).apply(pd.Series)
    games = pd.concat([games, hour_df], axis=1)

    # 5) Build model-ready features: is_dome, temp_effective, wind_effective, precip_3h_mm
    games["is_dome"] = (games["roof"] == "dome").astype("int8")

    # temp: C -> F
    temp_c = pd.to_numeric(games["temp_c"], errors="coerce")
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    temp_eff = temp_f.astype("float32")

    # If dome and temp missing -> 70°F, any remaining NaN -> 70°F
    temp_eff = temp_eff.where(~games["is_dome"].eq(1) | temp_eff.notna(), 70.0)
    temp_eff = temp_eff.fillna(70.0).astype("float32")
    games["temp_effective"] = temp_eff

    # wind: m/s -> mph, indoors -> 0, NaN -> 0
    wind_ms = pd.to_numeric(games["wind_ms"], errors="coerce")
    wind_mph = wind_ms * 2.236936

    # np.where gives a NumPy array
    wind_eff = np.where(games["is_dome"].eq(1), 0.0, wind_mph)

    # Replace NaN with 0, cast to float32
    wind_eff = np.nan_to_num(wind_eff, nan=0.0).astype("float32")

    games["wind_effective"] = wind_eff

    # precip_3h_mm already computed; just ensure numeric
    games["precip_3h_mm"] = pd.to_numeric(games["precip_3h_mm"], errors="coerce").astype("float32")

    # Keep only what we need per game key
    env_cols = [
        "season",
        "week",
        "home_team",
        "away_team",
        "game_date",
        "roof",
        "surface",
        "is_dome",
        "temp_effective",
        "wind_effective",
        "precip_3h_mm",
    ]
    games_env = games[env_cols].drop_duplicates()
    log.info("Built env rows for %d games.", len(games_env))

    # 6) Join onto player-per-game table
    log.info("Loading players table: %s", args.players)
    players = pd.read_parquet(args.players)
    for c in ["season", "week", "home_team", "away_team", "game_date"]:
        if c not in players.columns:
            raise ValueError(f"Players parquet missing required column '{c}'")

    players["home_team"] = players["home_team"].map(norm_team)
    players["away_team"] = players["away_team"].map(norm_team)

    merged = players.merge(
        games_env,
        on=["season", "week", "home_team", "away_team", "game_date"],
        how="left",
        suffixes=("", "_env"),
    )

    total = len(merged)
    have_temp = merged["temp_effective"].notna().sum()
    log.info(
        "Weather join coverage: %.2f%% rows with env features (%d / %d)",
        100.0 * have_temp / total if total else 0.0,
        have_temp,
        total,
    )

    merged.to_parquet(args.out, index=False)
    log.info("Wrote parquet: %s", args.out)


if __name__ == "__main__":
    main()
