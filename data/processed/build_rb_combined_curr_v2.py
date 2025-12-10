import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser as dtparse

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
PBP_PATH      = "play-by-play/pbp_combined_curr.parquet"
SCHEDULE_PATH = "games_schedule.parquet"
ODDS_PATH     = "nfl_odds_updated.csv"
ROSTER_PATH   = "roster_weekly/roster_weekly_combined_curr.parquet"
DEPTH_PATH    = "depth_charts/depth_charts_combined_curr.parquet"
LOCATIONS_PATH = "locations.csv"              # stadium metadata (homeTeamAbbr, roof, surface, lat, long)
WEATHER_CACHE_PATH = "openmeteo_cache_3h_curr.json"  # your JSON cache

OUTPUT_PATH   = "rb_combined_curr_v2.parquet"

# Suppress chained assignment warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)

# ==========================================
# FIX 1: UNIFIED TEAM MAPPING (Target: nflverse standard)
# ==========================================
# Map Odds CSV Team Names -> PBP/Schedule Abbreviations
# CRITICAL CHANGE: Rams -> "LA", Raiders -> "LV", Washington -> "WAS"
TEAM_MAP = {
    # RAMS (Force everything to 'LA')
    "Los Angeles Rams": "LA", "LA Rams": "LA", "L.A. Rams": "LA", "LAR": "LA", 
    "St. Louis Rams": "LA", "STL": "LA", "LA": "LA",
    
    # CHARGERS (Force everything to 'LAC')
    "Los Angeles Chargers": "LAC", "LA Chargers": "LAC", "L.A. Chargers": "LAC", 
    "LAC": "LAC", "San Diego Chargers": "LAC", "San Diego": "LAC", "SD": "LAC",
    
    # RAIDERS (Force everything to 'LV')
    "Las Vegas Raiders": "LV", "Oakland Raiders": "LV", "Oakland": "LV", "OAK": "LV", 
    "Raiders": "LV", "LV": "LV",
    
    # WASHINGTON (Force everything to 'WAS')
    "Washington Commanders": "WAS", "Washington Football Team": "WAS", 
    "Washington Redskins": "WAS", "Washington": "WAS", "WAS": "WAS", "WSH": "WAS",
    
    # STANDARD TEAMS
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN"
}

# Team normalization for locations / PBP alignment
TEAM_NORM = {
    # CRITICAL FIXES
    "LAR": "LA", "STL": "LA", "LA": "LA",   # Rams -> LA
    "OAK": "LV", "RAI": "LV", "LV": "LV",   # Raiders -> LV
    "SD": "LAC", "LAC": "LAC",              # Chargers -> LAC
    "WSH": "WAS", "WAS": "WAS",             # Washington -> WAS
    
    # Standard Short Codes
    "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"
}


def norm_team(x: str) -> str:
    s = (str(x) if x is not None else "").strip().upper()
    return TEAM_NORM.get(s, s)


# ==========================================
# 2. WEATHER HELPERS (Open-Meteo JSON)
# ==========================================

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


def pick_mode(series: pd.Series):
    vals = [v for v in series.dropna().tolist() if str(v).strip() != ""]
    if not vals:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def parse_kick_local(game_date, time_of_day) -> datetime:
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


def extract_hour_and_3h(json_obj: dict, local_hour: datetime) -> dict:
    """
    From Open-Meteo JSON and desired local_hour, pick:
      - temp_c at kickoff hour
      - wind_kmh at kickoff hour
      - precip_3h_mm = sum precip over [kickoff, kickoff+3h)
    """
    hourly = json_obj.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {"temp_c": np.nan, "wind_kmh": np.nan, "precip_3h_mm": np.nan}

    time_index = pd.to_datetime(times)
    t_local = pd.to_datetime(local_hour)

    # closest hour index
    diffs = np.abs((time_index - t_local).total_seconds())
    i0 = int(diffs.argmin())

    def arr(field):
        vals = hourly.get(field, [])
        if not vals:
            return np.full(len(time_index), np.nan, dtype=float)
        return np.array(pd.to_numeric(vals, errors="coerce"), dtype=float)

    temp_arr = arr("temperature_2m")     # °C
    wind_arr = arr("wind_speed_10m")     # km/h (per your JSON)
    precip_arr = arr("precipitation")    # mm

    temp_c = temp_arr[i0] if not np.isnan(temp_arr[i0]) else np.nan
    wind_kmh = wind_arr[i0] if not np.isnan(wind_arr[i0]) else np.nan

    # 3-hour precip window
    t_end = t_local + timedelta(hours=3)
    mask = (time_index >= t_local) & (time_index < t_end)
    if mask.any():
        precip_3h = float(np.nansum(precip_arr[mask]))
    else:
        precip_3h = np.nan

    return {
        "temp_c": temp_c,
        "wind_kmh": wind_kmh,
        "precip_3h_mm": precip_3h,
    }


def build_weather_from_cache(pbp_path, locations_path, cache_path) -> pd.DataFrame:
    """
    Build per-game weather using:
      - pbp_combined.parquet (season/week/home_team/away_team/game_date/time_of_day/game_id)
      - locations.csv (homeTeamAbbr, roof, surface, lat, long)
      - Open-Meteo JSON cache

    Returns: DataFrame keyed by game_id with:
      game_id, roof, surface, is_dome, temp_effective (F), wind_effective (mph), precip_3h_mm
    """
    print("--- Building Weather from Open-Meteo JSON Cache ---")

    # Locations / stadium metadata
    loc = pd.read_csv(locations_path)
    loc["homeTeamAbbr"] = loc["homeTeamAbbr"].map(norm_team)
    loc = loc.rename(columns={"homeTeamAbbr": "home_team", "long": "lon"})
    loc["lat"] = pd.to_numeric(loc["lat"], errors="coerce")
    loc["lon"] = pd.to_numeric(loc["lon"], errors="coerce")
    loc["roof"] = loc["roof"].apply(roof_from_csv)       # -> dome / closed / outdoors
    loc["surface"] = loc["surface"].apply(surface_from_csv)

    # PBP for game-level info
    pbp = pd.read_parquet(pbp_path)
    need_cols = ["game_id", "season", "week", "home_team", "away_team", "game_date", "time_of_day"]
    for c in need_cols:
        if c not in pbp.columns:
            raise ValueError(f"PBP missing required column: {c}")

    games = pbp[need_cols].copy()
    games["home_team"] = games["home_team"].map(norm_team)
    games["away_team"] = games["away_team"].map(norm_team)

    # One row per game_id
    games = (
        games.groupby("game_id", as_index=False)
        .agg({
            "season": "first",
            "week": "first",
            "home_team": "first",
            "away_team": "first",
            "game_date": "first",
            "time_of_day": pick_mode,
        })
    )

    # Attach lat/lon + roof/surface from locations
    games = games.merge(
        loc[["home_team", "lat", "lon", "roof", "surface"]],
        on="home_team",
        how="left",
    )

    games["kick_local"] = games.apply(
        lambda r: round_to_nearest_hour(parse_kick_local(r["game_date"], r["time_of_day"])),
        axis=1,
    )
    games["game_date_str"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

    # Load cache
    with open(cache_path, "r") as f:
        cache = json.load(f)

    # Extract temp_c, wind_kmh, precip_3h_mm from cache per game
    temp_c_list = []
    wind_kmh_list = []
    precip_list = []

    for idx, row in games.iterrows():
        ht = row["home_team"]
        gd = row["game_date_str"]
        lat = row["lat"]
        lon = row["lon"]

        if pd.isna(lat) or pd.isna(lon):
            temp_c_list.append(np.nan)
            wind_kmh_list.append(np.nan)
            precip_list.append(np.nan)
            continue

        key = f"{ht}:{gd}:{float(lat):.4f},{float(lon):.4f}"
        wx_json = cache.get(key)

        if not isinstance(wx_json, dict):
            temp_c_list.append(np.nan)
            wind_kmh_list.append(np.nan)
            precip_list.append(np.nan)
            continue

        feats = extract_hour_and_3h(wx_json, row["kick_local"])
        temp_c_list.append(feats["temp_c"])
        wind_kmh_list.append(feats["wind_kmh"])
        precip_list.append(feats["precip_3h_mm"])

    games["temp_c"] = temp_c_list
    games["wind_kmh"] = wind_kmh_list
    games["precip_3h_mm"] = precip_list

    # ---------- indoor overrides + final features ----------

    # is_dome = dome OR closed
    indoor_mask = games["roof"].isin(["dome", "closed"])
    games["is_dome"] = indoor_mask.astype("int8")

    # temp: C -> F
    temp_c = pd.to_numeric(games["temp_c"], errors="coerce")
    temp_f = temp_c * 9.0 / 5.0 + 32.0

    # Start with weather temp, then override indoors, then fill missing
    temp_eff = temp_f.astype("float32")

    # Indoors (dome/closed) -> hard 70 F regardless of ERA5
    temp_eff = temp_eff.where(~indoor_mask, 70.0)

    # Any remaining NaN (e.g. missing weather) -> 70
    temp_eff = temp_eff.fillna(70.0).astype("float32")
    games["temp_effective"] = temp_eff

    # wind: km/h -> mph
    wind_kmh = pd.to_numeric(games["wind_kmh"], errors="coerce")
    wind_mph = wind_kmh * 0.621371

    # Indoors -> hard 0 mph
    wind_eff = np.where(indoor_mask, 0.0, wind_mph)

    # Any remaining NaN -> 0
    wind_eff = np.nan_to_num(wind_eff, nan=0.0).astype("float32")
    games["wind_effective"] = wind_eff

    # precip_3h_mm already numeric-ish
    games["precip_3h_mm"] = pd.to_numeric(games["precip_3h_mm"], errors="coerce").astype("float32")

    env_cols = [
        "game_id",
        "roof",
        "surface",
        "is_dome",
        "temp_effective",
        "wind_effective",
        "precip_3h_mm",
    ]
    games_env = games[env_cols].drop_duplicates()
    print(f"   Built env rows for {len(games_env)} games from JSON cache (with 70/0 indoors).")

    return games_env


# ==========================================
# 3. ODDS PROCESSING (ROBUST FIX)
# ==========================================

def process_odds(odds, schedule):
    print("--- Processing Odds (Robust Merge) ---")
    
    # 1. Clean the Odds Data First
    # Force numeric columns to handle "Data supply issue" strings
    odds["Home Line Close"] = pd.to_numeric(odds["Home Line Close"], errors='coerce')
    odds["Total Score Open"] = pd.to_numeric(odds["Total Score Open"], errors='coerce')
    
    # Drop rows that have no valid line data
    odds = odds.dropna(subset=["Home Line Close"])
    
    # 2. Normalize Odds Teams
    odds["home_team_abbr"] = odds["Home Team"].map(TEAM_MAP)
    odds["gameday"] = pd.to_datetime(odds["Date"], format="mixed", errors="coerce").dt.date

    odds_clean = odds[[
        "gameday", "home_team_abbr", "Home Line Close", "Total Score Open"
    ]].rename(columns={
        "Home Line Close": "home_line_close",
        "Total Score Open": "total_score_open",
        "home_team_abbr": "home_team"
    })
    # Remove rows where team mapping failed
    odds_clean = odds_clean.dropna(subset=['home_team'])

    # 3. Normalize Schedule Teams (CRITICAL: Normalize BEFORE Merge)
    sched_key = schedule[["game_id", "gameday", "home_team"]].copy()
    sched_key["home_team"] = sched_key["home_team"].map(norm_team)
    sched_key["gameday"] = pd.to_datetime(sched_key["gameday"]).dt.date

    merged = pd.merge(sched_key, odds_clean, on=["gameday", "home_team"], how="left")
    missing_mask = merged["home_line_close"].isna()

    # 4. Expanded Fuzzy Recovery (+/- 4 Days)
    if missing_mask.sum() > 0:
        print(f"   Refining match for {missing_mask.sum()} games using +/- 4 day logic...")
        from datetime import timedelta

        missing_rows = merged[missing_mask][["game_id", "gameday", "home_team"]]

        # Try loop for days 1 to 4
        for i in range(1, 5):
            # Check Forward
            odds_shifted = odds_clean.copy()
            odds_shifted["gameday"] = odds_shifted["gameday"] - timedelta(days=i)
            rec = pd.merge(missing_rows, odds_shifted, on=["gameday", "home_team"], how="inner")
            if not rec.empty:
                rec = rec.drop_duplicates(subset=["game_id"])
                merged = merged.set_index("game_id")
                rec = rec.set_index("game_id")
                merged.update(rec)
                merged = merged.reset_index()

            # Check Backward
            odds_shifted = odds_clean.copy()
            odds_shifted["gameday"] = odds_shifted["gameday"] + timedelta(days=i)
            rec = pd.merge(missing_rows, odds_shifted, on=["gameday", "home_team"], how="inner")
            if not rec.empty:
                rec = rec.drop_duplicates(subset=["game_id"])
                merged = merged.set_index("game_id")
                rec = rec.set_index("game_id")
                merged.update(rec)
                merged = merged.reset_index()
            
            # Check remaining
            missing_mask = merged["home_line_close"].isna()
            missing_rows = merged[missing_mask][["game_id", "gameday", "home_team"]]
            if missing_rows.empty: break

    final_odds = merged[["game_id", "home_line_close", "total_score_open"]]
    
    return final_odds.drop_duplicates(subset=['game_id'])


def process_pbp_stats(pbp_path):
    print(f"--- Loading PBP from {pbp_path} ---")
    df = pd.read_parquet(pbp_path)
    df = df[df["play_type"].isin(["run", "pass", "no_play"])].copy()

    print("--- Engineering Contextual Flags ---")
    df["is_early_down"] = df["down"].isin([1, 2]).astype(int)
    df["is_neutral_script"] = df["wp"].fillna(0).between(0.20, 0.80).astype(int)
    df["is_rz"] = (df["yardline_100"] <= 20).astype(int)
    df["is_gtg"] = (df["goal_to_go"] == 1).astype(int)
    df["is_short_yds"] = (df["ydstogo"] <= 2).astype(int)
    df["is_two_minute"] = (df["half_seconds_remaining"] <= 120).astype(int)
    df["is_four_minute"] = ((df["qtr"] == 4) & (df["quarter_seconds_remaining"] < 240) & (df["score_differential"] > 0)).astype(int)
    df["is_under_center"] = (df["shotgun"] == 0).astype(int)

    print("--- Aggregating RB Stats ---")
    runs = df[(df["rush_attempt"] == 1) & (df["play_type"] == "run") & (df["rusher_player_id"].notna())].copy()
    runs["is_explosive10"] = (runs["yards_gained"] >= 10).astype(int)
    runs["is_stuffed"] = (runs["yards_gained"] <= 0).astype(int)

    rb_stats = runs.groupby(
        ["season", "week", "game_id", "game_date", "posteam", "defteam", "rusher_player_id", "rusher_player_name"]
    ).agg(
        rb_carries=("play_id", "count"),
        rb_rush_yards=("yards_gained", "sum"),
        rb_epa_sum=("epa", "sum"),
        rb_explosive10_count=("is_explosive10", "sum"),
        rb_stuffed_count=("is_stuffed", "sum"),
        rb_rz_carries=("is_rz", "sum"),
        rb_gtg_carries=("is_gtg", "sum"),
        rb_short_yds_carries=("is_short_yds", "sum"),
        rb_third_down_carries=("down", lambda x: (x == 3).sum()),
        rb_two_minute_carries=("is_two_minute", "sum"),
        rb_four_minute_carries=("is_four_minute", "sum"),
        rb_undercenter_carries=("is_under_center", "sum"),
        rb_nohuddle_share_carries=("no_huddle", "sum"),
        rb_neutral_carries=("play_id", lambda x: x[runs.loc[x.index, "is_neutral_script"] == 1].count()),
        rb_ed_neutral_carries=("play_id", lambda x: x[(runs.loc[x.index, "is_neutral_script"] == 1) &
                                                      (runs.loc[x.index, "is_early_down"] == 1)].count()),
    ).reset_index()

    rb_stats.rename(columns={
        "rusher_player_id": "player_id",
        "rusher_player_name": "full_name",
        "defteam": "opponent",
    }, inplace=True)

    pass_plays = df[(df["pass_attempt"] == 1) & (df["receiver_player_id"].notna())].copy()
    rec_stats = pass_plays.groupby(["game_id", "receiver_player_id"]).agg(
        rec_targets=("play_id", "count"),
        rec_receptions=("complete_pass", "sum"),
        rec_rec_yards=("yards_gained", lambda x: x[pass_plays.loc[x.index, "complete_pass"] == 1].sum()),
    ).reset_index()

    rb_stats = pd.merge(
        rb_stats,
        rec_stats,
        left_on=["game_id", "player_id"],
        right_on=["game_id", "receiver_player_id"],
        how="left",
    )
    rb_stats[["rec_targets", "rec_receptions", "rec_rec_yards"]] = rb_stats[
        ["rec_targets", "rec_receptions", "rec_rec_yards"]
    ].fillna(0)

    print("--- Aggregating Team & QB Stats ---")
    team_rush_stats = runs.groupby(["game_id", "posteam"]).agg(
        team_total_carries=("play_id", "count"),
        team_neutral_carries=("play_id", lambda x: x[runs.loc[x.index, "is_neutral_script"] == 1].count()),
        team_ed_neutral_carries=("play_id", lambda x: x[(runs.loc[x.index, "is_neutral_script"] == 1) &
                                                        (runs.loc[x.index, "is_early_down"] == 1)].count()),
    ).reset_index()

    dropbacks = df[df["qb_dropback"] == 1]
    qb_stats = dropbacks.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        dropbacks=("play_id", "count"),
        qb_pass_yards=("passing_yards", "sum"),
        qb_epa_sum=("epa", "sum"),
        qb_scrambles=("qb_scramble", "sum"),
    ).reset_index()
    qb_stats["rank"] = qb_stats.groupby(["game_id", "posteam"])["dropbacks"].rank(method="first", ascending=False)
    starter_stats = qb_stats[qb_stats["rank"] == 1].copy()
    starter_stats.rename(columns={
        "qb_pass_yards": "starter_pass_yards",
        "qb_epa_sum": "starter_epa_sum",
        "dropbacks": "starter_dropbacks",
        "qb_scrambles": "starter_scrambles",
    }, inplace=True)

    team_passing = df[df["pass_attempt"] == 1].groupby(["game_id", "posteam"])["yards_gained"].sum().reset_index(
        name="team_wr_yards"
    )

    ed_neutral_plays = df[
        (df["is_early_down"] == 1) &
        (df["is_neutral_script"] == 1) &
        (df["play_type"].isin(["run", "pass"]))
    ]
    team_rates = ed_neutral_plays.groupby(["game_id", "posteam"]).agg(
        ed_neutral_pass=("pass_attempt", "sum"),
        ed_neutral_total=("play_id", "count"),
    ).reset_index()
    team_rates["qb_ed_neutral_pass_rate"] = team_rates["ed_neutral_pass"] / team_rates["ed_neutral_total"]

    tr_ed = df[(df["is_early_down"] == 1) & (df["play_type"].isin(["run", "pass"]))].groupby(
        ["game_id", "posteam"]
    )["pass_attempt"].mean().reset_index(name="qb_ed_pass_rate_all")
    tr_h1 = df[(df["game_half"] == "Half1") & (df["play_type"].isin(["run", "pass"]))].groupby(
        ["game_id", "posteam"]
    )["pass_attempt"].mean().reset_index(name="qb_pass_rate_h1_all")
    tr_h2 = df[(df["game_half"] == "Half2") & (df["play_type"].isin(["run", "pass"]))].groupby(
        ["game_id", "posteam"]
    )["pass_attempt"].mean().reset_index(name="qb_pass_rate_h2_all")

    def_stats = runs.groupby(["game_id", "defteam"]).agg(
        opp_rush_epa_sum=("epa", "sum"),
        opp_rush_attempts=("play_id", "count"),
        opp_rush_yards=("yards_gained", "sum"),
    ).reset_index()
    def_stats.rename(columns={"defteam": "opponent"}, inplace=True)

    merged = pd.merge(rb_stats, team_rush_stats, on=["game_id", "posteam"], how="left")
    merged = pd.merge(
        merged,
        starter_stats[
            ["game_id", "posteam", "starter_pass_yards", "starter_epa_sum", "starter_dropbacks", "starter_scrambles"]
        ],
        on=["game_id", "posteam"],
        how="left",
    )
    merged = pd.merge(merged, team_rates[["game_id", "posteam", "qb_ed_neutral_pass_rate"]],
                      on=["game_id", "posteam"], how="left")
    merged = pd.merge(merged, tr_ed, on=["game_id", "posteam"], how="left")
    merged = pd.merge(merged, tr_h1, on=["game_id", "posteam"], how="left")
    merged = pd.merge(merged, tr_h2, on=["game_id", "posteam"], how="left")
    merged = pd.merge(merged, team_passing, on=["game_id", "posteam"], how="left")
    merged = pd.merge(merged, def_stats, on=["game_id", "opponent"], how="left")

    return merged


def process_roles(depth_path, roster_path):
    print("--- Loading Depth Charts & Rosters for Role Context ---")
    df_depth = pd.read_parquet(depth_path)
    df_depth["depth_team"] = pd.to_numeric(df_depth["depth_team"], errors="coerce")
    df_depth = df_depth[df_depth["position"] == "RB"][["gsis_id", "season", "week", "depth_team", "club_code"]].copy()
    df_depth = (
        df_depth
        .sort_values(["gsis_id", "season", "week", "depth_team"])
        .drop_duplicates(subset=["gsis_id", "season", "week"], keep="first")
    )
    df_depth.rename(columns={"depth_team": "depth_rank", "club_code": "team"}, inplace=True)

    df_roster = pd.read_parquet(roster_path)
    df_roster = df_roster[["gsis_id", "season", "week", "status"]].copy()
    df_roster = (
        df_roster
        .sort_values(["gsis_id", "season", "week"])
        .drop_duplicates(subset=["gsis_id", "season", "week"], keep="last")
    )

    df_role = pd.merge(df_depth, df_roster, on=["gsis_id", "season", "week"], how="left")
    df_role["status"] = df_role["status"].fillna("ACT")

    active_mask = df_role["status"] == "ACT"
    df_role["depth_rank_active"] = np.where(active_mask, df_role["depth_rank"], np.nan)
    min_depth_active = df_role.groupby(["season", "week", "team"])["depth_rank_active"].transform("min")

    df_role["is_projected_starter"] = (
        active_mask &
        (df_role["depth_rank_active"] == min_depth_active)
    ).astype(int)

    df_role.rename(columns={"gsis_id": "player_id"}, inplace=True)
    df_role = df_role[["player_id", "season", "week", "team", "depth_rank", "status", "is_projected_starter"]]

    df_role = (
        df_role
        .sort_values(["player_id", "season", "week", "depth_rank"])
        .drop_duplicates(subset=["player_id", "season", "week"], keep="first")
    )

    print("Role rows after clean:", len(df_role))
    print(
        "Unique (player_id, season, week):",
        df_role[["player_id", "season", "week"]].drop_duplicates().shape[0],
    )

    return df_role


def get_rb_player_ids(roster_path):
    """Get set of player IDs that are RBs from roster data."""
    print("--- Loading RB player IDs from roster ---")
    df_roster = pd.read_parquet(roster_path)
    
    # Filter to RBs only
    rb_roster = df_roster[df_roster["position"] == "RB"]
    rb_ids = set(rb_roster["gsis_id"].dropna().unique())
    
    print(f"   Found {len(rb_ids)} unique RB player IDs")
    return rb_ids


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    # 1) PBP stats
    main_df = process_pbp_stats(PBP_PATH)

    # 1b) Filter to RBs only using roster position data
    rb_ids = get_rb_player_ids(ROSTER_PATH)
    before_filter = len(main_df)
    main_df = main_df[main_df["player_id"].isin(rb_ids)]
    print(f"   Filtered to RBs: {before_filter} -> {len(main_df)} rows")

    # 2) Schedule & odds
    print("--- Loading Schedule & Odds ---")
    schedule = pd.read_parquet(SCHEDULE_PATH)
    odds = pd.read_csv(ODDS_PATH)

    odds_df = process_odds(odds, schedule)

    # 3) Weather from Open-Meteo JSON cache
    weather_df = build_weather_from_cache(PBP_PATH, LOCATIONS_PATH, WEATHER_CACHE_PATH)

    # 4) Merge context
    print("--- Merging Context Data ---")
    main_df = pd.merge(main_df, weather_df, on="game_id", how="left")
    main_df = pd.merge(main_df, odds_df, on="game_id", how="left")

    # === DATA RECOVERY: Fill Remaining Missing Odds with League Averages ===
    # This prevents dropping rows where odds are still missing due to bad CSV data
    missing_odds = main_df["home_line_close"].isna()
    if missing_odds.sum() > 0:
        print(f"   NOTICE: {missing_odds.sum()} rows still missing odds after fuzzy match.")
        print("           Filling with League Averages (Spread: -2.5, Total: 44.5).")
        main_df["home_line_close"] = main_df["home_line_close"].fillna(-2.5)
        main_df["total_score_open"] = main_df["total_score_open"].fillna(44.5)

    # 5) Home/Away & betting lines
    print("--- Finalizing Home/Away & Betting Lines ---")
    sched_info = schedule[["game_id", "home_team"]].drop_duplicates()
    if "home_team" in main_df.columns:
        main_df = pd.merge(main_df, sched_info, on="game_id", suffixes=("", "_sched"), how="left")
        main_df["home_team"] = main_df["home_team"].fillna(main_df["home_team_sched"])
        if "home_team_sched" in main_df.columns:
            main_df.drop(columns=["home_team_sched"], inplace=True)
    else:
        main_df = pd.merge(main_df, sched_info, on="game_id", how="left")

    main_df["is_home"] = (main_df["posteam"] == main_df["home_team"]).astype(int)

    main_df["team_spread"] = np.where(
        main_df["is_home"] == 1,
        main_df["home_line_close"],
        -1 * main_df["home_line_close"],
    )

    # If you want implied script = Vegas expectation (could tweak)
    main_df["implied_game_script"] = main_df["team_spread"]

    # 6) RB / QB / DEF features
    print("--- Finalizing Columns ---")
    main_df["rb_ypc"] = main_df["rb_rush_yards"] / main_df["rb_carries"]
    main_df["rb_epa_per_rush"] = main_df["rb_epa_sum"] / main_df["rb_carries"]
    main_df["rb_explosive10_rate"] = main_df["rb_explosive10_count"] / main_df["rb_carries"]
    main_df["rb_stuffed_rate"] = main_df["rb_stuffed_count"] / main_df["rb_carries"]

    main_df["rb_ed_carry_share_all"] = main_df["rb_ed_neutral_carries"] / main_df["team_ed_neutral_carries"]
    main_df["rb_ed_carry_share_all"] = main_df["rb_ed_carry_share_all"].fillna(0)

    main_df["rb_ed_neutral_carry_rate"] = main_df["rb_neutral_carries"] / main_df["team_neutral_carries"]
    main_df["rb_neutral_carries_per_game"] = main_df["rb_neutral_carries"]
    main_df["rb_nohuddle_share_carries"] = main_df["rb_nohuddle_share_carries"] / main_df["rb_carries"]
    main_df["rb_undercenter_share_carries"] = main_df["rb_undercenter_carries"] / main_df["rb_carries"]

    main_df["starter_qb_epa_per_db"] = main_df["starter_epa_sum"] / main_df["starter_dropbacks"]
    main_df["starter_qb_scramble_rate_db"] = main_df["starter_scrambles"] / main_df["starter_dropbacks"]

    main_df["opponent_rush_epa_allowed"] = main_df["opp_rush_epa_sum"] / main_df["opp_rush_attempts"]
    main_df["opponent_rush_yards_per_game_allowed"] = main_df["opp_rush_yards"]

    # 7) Roles
    roles_df = process_roles(DEPTH_PATH, ROSTER_PATH)
    main_df = pd.merge(main_df, roles_df, on=["player_id", "season", "week"], how="left")

    main_df["depth_rank"] = main_df["depth_rank"].fillna(4).astype(int)
    main_df["is_projected_starter"] = main_df["is_projected_starter"].fillna(0).astype(int)
    main_df["status"] = main_df["status"].fillna("ACT")
    main_df["is_active"] = np.where(main_df["status"] == "ACT", 1, 0)

    fill_zeros = [
        "rb_ed_neutral_carry_rate",
        "rb_nohuddle_share_carries",
        "rb_undercenter_share_carries",
        "qb_ed_neutral_pass_rate",
        "qb_ed_pass_rate_all",
    ]
    for c in fill_zeros:
        if c in main_df.columns:
            main_df[c] = main_df[c].fillna(0)

    # 8) Final column selection (home_team REMOVED here)
    final_cols = [
        "player_id", "full_name", "season", "week", "game_date", "game_id",
        "posteam", "opponent", "is_home",
        "rb_rush_yards", "rb_carries", "rb_ed_carry_share_all", "rb_rz_carries",
        "rb_epa_per_rush", "rb_ypc", "rb_explosive10_rate", "rb_stuffed_rate",
        "rec_targets", "rec_receptions", "rec_rec_yards",
        "starter_pass_yards", "starter_qb_epa_per_db", "starter_qb_scramble_rate_db",
        "team_wr_yards",
        "opponent_rush_epa_allowed", "opponent_rush_yards_per_game_allowed",
        "home_line_close", "total_score_open", "team_spread", "implied_game_script",
        "temp_effective", "wind_effective", "is_dome",
        "depth_rank", "is_projected_starter", "is_active",
        "qb_ed_neutral_pass_rate", "qb_ed_pass_rate_all", "qb_pass_rate_h1_all", "qb_pass_rate_h2_all",
        "rb_ed_neutral_carry_rate", "rb_neutral_carries_per_game",
        "rb_nohuddle_share_carries", "rb_four_minute_carries", "rb_gtg_carries",
        "rb_undercenter_carries", "rb_undercenter_share_carries",
        "rb_short_yds_carries", "rb_third_down_carries", "rb_two_minute_carries",
    ]
    safe_cols = [c for c in final_cols if c in main_df.columns]
    main_df = main_df[safe_cols]

    # Explicitly drop home_team if it somehow stuck around
    if "home_team" in main_df.columns:
        main_df = main_df.drop(columns=["home_team"])

    # 9) Enforce unique grain: (player, season, week, date, game_id)
    key_cols = ["player_id", "season", "week", "game_date", "game_id"]
    main_df = main_df.sort_values(key_cols + ["rb_carries"], ascending=[True, True, True, True, True, False])
    main_df = main_df.drop_duplicates(subset=key_cols, keep="first")

    print(f"--- Saving {main_df.shape} to {OUTPUT_PATH} ---")
    main_df.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()