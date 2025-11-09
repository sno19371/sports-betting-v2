# build_rb_game.py
# One row per rusher per game with modeling features.
# Neutral = (qtr in {1,2,3}) AND (score_differential in [-7, +7]).
# Adds home_team, away_team, final home/away score, and game_date.
# Includes non-neutral counterparts for early-down/half splits and 3rd-down distance shares.

import argparse
import numpy as np
import pandas as pd


def main(pbp_path: str, out_path: str):
    df = pd.read_parquet(pbp_path)

    # ---- ID / name columns (support both nflverse variants) ----
    rid_col = "rusher_id" if "rusher_id" in df.columns else "rusher_player_id"
    rname_col = "rusher" if "rusher" in df.columns else "rusher_player_name"

    # ---- numeric flags ----
    for c in [
        "rush_attempt","qb_kneel","aborted_play","play",
        "first_down_rush","rush_touchdown","tackled_for_loss",
        "shotgun","no_huddle","penalty","fumble","fumble_lost"
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    for c in ["epa","wpa","rushing_yards","penalty_yards","air_yards"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ---- canonical rush filter (exclude kneels, aborted, and no-plays) ----
    df["is_rush"] = (
        (df.get("rush_attempt", 0) == 1)
        & (df.get("qb_kneel", 0) == 0)
        & (df.get("aborted_play", 0) == 0)
        & (df.get("play", 1) == 1)  # keep if column missing
    ).astype(int)

    # ---- rush-only EPA/WPA (prevents kneel EPA leaking into sums) ----
    df["epa_rush"] = df["epa"] * df["is_rush"]
    df["wpa_rush"] = df["wpa"] * df["is_rush"]

    # ---- Situational flags ----
    df["neutral"] = (
        df["qtr"].between(1, 3, inclusive="both")
        & df["score_differential"].between(-7, 7, inclusive="both")
    ).astype(int)
    df["early_down"] = df["down"].isin([1, 2]).astype(int)
    df["half_1"] = df["qtr"].isin([1, 2]).astype(int)
    df["half_2"] = df["qtr"].isin([3, 4, 5]).astype(int)  # OT -> 2H
    df["rz"] = (df["yardline_100"] <= 20).astype(int)
    df["i10"] = (df["yardline_100"] <= 10).astype(int)
    df["i5"] = (df["yardline_100"] <= 5).astype(int)
    df["gtg"] = df["goal_to_go"].fillna(0).astype(int)
    df["third_down"] = (df["down"] == 3).astype(int)
    ytg = df["ydstogo"].astype(float)
    df["td_1_3"] = ((df["third_down"] == 1) & (ytg.between(1, 3))).astype(int)
    df["td_4_6"] = ((df["third_down"] == 1) & (ytg.between(4, 6))).astype(int)
    df["td_7_9"] = ((df["third_down"] == 1) & (ytg.between(7, 9))).astype(int)
    df["td_10p"] = ((df["third_down"] == 1) & (ytg >= 10)).astype(int)
    df["short_yds"] = (ytg <= 2).astype(int)
    df["four_minute"] = ((df["qtr"] == 4) & (df["score_differential"] > 0)).astype(int)  # proxy
    df["two_minute"] = (df["quarter_seconds_remaining"] <= 120) & (df["qtr"].isin([2, 4]))
    df["two_minute"] = df["two_minute"].astype(int)

    # ---- Direction buckets (combine run_location + run_gap into 7-lane taxonomy) ----
    if ("run_location" in df.columns) and ("run_gap" in df.columns):
        def _dir7(row):
            rl = row["run_location"]
            rg = row["run_gap"]
            if rl == "middle":
                return "middle"
            if rl in ("left", "right"):
                if rg in ("end", "tackle", "guard"):
                    return f"{rl}_{rg}"
            return None
        df["dir7"] = df.apply(_dir7, axis=1)
    else:
        df["dir7"] = None

    dir_bins = ["left_end","left_tackle","left_guard","middle","right_guard","right_tackle","right_end"]
    for bin_ in dir_bins:
        df[f"dir_{bin_}"] = ((df["dir7"] == bin_) & (df["is_rush"] == 1)).astype(int)

    # ---- Formation flags on rushes ----
    df["shotgun_rush"] = (df.get("shotgun", 0) * df["is_rush"]).astype(int)
    df["undercenter_rush"] = (((1 - df.get("shotgun", 0)) * df["is_rush"]).astype(int))
    df["nohuddle_rush"] = (df.get("no_huddle", 0) * df["is_rush"]).astype(int)

    # ---- Explosives & stuffs ----
    ry = df["rushing_yards"]
    df["explosive10"] = ((df["is_rush"] == 1) & (ry >= 10)).astype(int)
    df["explosive15"] = ((df["is_rush"] == 1) & (ry >= 15)).astype(int)
    df["explosive20"] = ((df["is_rush"] == 1) & (ry >= 20)).astype(int)
    df["stuffed_le0"] = ((df["is_rush"] == 1) & (ry <= 0)).astype(int)

    # ---- Penalties drawn on runs (defensive) ----
    df["rush_def_pen"] = ((df.get("penalty", 0) == 1) & (df.get("penalty_team") != df.get("posteam")) & (df["is_rush"] == 1)).astype(int)
    df["rush_def_pen_yds"] = np.where(df["rush_def_pen"] == 1, df.get("penalty_yards", 0), 0)

    # ---- Game-level metadata (final score & teams) ----
    game_fields = (
        df.groupby("game_id", as_index=False)
          .agg(
              home_team=("home_team", "first"),
              away_team=("away_team", "first"),
              game_date=("game_date", "first"),
              home_score=("home_score", "max"),
              away_score=("away_score", "max"),
          )
    )

    # ---- grouping keys ----
    keys = ["season", "week", "game_id", "posteam", rid_col, rname_col]

    # ---- Per-game rusher aggregates ----
    g = df[df[rid_col].notna()].groupby(keys, dropna=False)

    per_game = g.agg(
        # Volume
        carries=("is_rush", "sum"),
        neutral_carries=("is_rush", lambda s: int((df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),
        ed_neutral_carries=("is_rush", lambda s: int((df.loc[s.index, "early_down"] * df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),
        ed_carries_all=("is_rush", lambda s: int((df.loc[s.index, "early_down"] * df.loc[s.index, "is_rush"]).sum())),
        h1_carries_all=("is_rush", lambda s: int((df.loc[s.index, "half_1"] * df.loc[s.index, "is_rush"]).sum())),
        h2_carries_all=("is_rush", lambda s: int((df.loc[s.index, "half_2"] * df.loc[s.index, "is_rush"]).sum())),
        third_down_carries=("third_down", lambda s: int((df.loc[s.index, "third_down"] * df.loc[s.index, "is_rush"]).sum())),
        fourth_down_carries=("down", lambda s: int(((df.loc[s.index, "down"] == 4) * df.loc[s.index, "is_rush"]).sum())),
        short_yds_carries=("short_yds", lambda s: int((df.loc[s.index, "short_yds"] * df.loc[s.index, "is_rush"]).sum())),
        four_minute_carries=("four_minute", lambda s: int((df.loc[s.index, "four_minute"] * df.loc[s.index, "is_rush"]).sum())),
        two_minute_carries=("two_minute", lambda s: int((df.loc[s.index, "two_minute"] * df.loc[s.index, "is_rush"]).sum())),
        rz_carries=("rz", lambda s: int((df.loc[s.index, "rz"] * df.loc[s.index, "is_rush"]).sum())),
        i10_carries=("i10", lambda s: int((df.loc[s.index, "i10"] * df.loc[s.index, "is_rush"]).sum())),
        i5_carries=("i5", lambda s: int((df.loc[s.index, "i5"] * df.loc[s.index, "is_rush"]).sum())),
        gtg_carries=("gtg", lambda s: int((df.loc[s.index, "gtg"] * df.loc[s.index, "is_rush"]).sum())),

        # 3rd-down distance carries (overall & neutral)
        td_1_3_carries=("td_1_3", lambda s: int((df.loc[s.index, "td_1_3"] * df.loc[s.index, "is_rush"]).sum())),
        td_4_6_carries=("td_4_6", lambda s: int((df.loc[s.index, "td_4_6"] * df.loc[s.index, "is_rush"]).sum())),
        td_7_9_carries=("td_7_9", lambda s: int((df.loc[s.index, "td_7_9"] * df.loc[s.index, "is_rush"]).sum())),
        td_10p_carries=("td_10p", lambda s: int((df.loc[s.index, "td_10p"] * df.loc[s.index, "is_rush"]).sum())),
        td_1_3_neutral=("td_1_3", lambda s: int((df.loc[s.index, "td_1_3"] * df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),
        td_4_6_neutral=("td_4_6", lambda s: int((df.loc[s.index, "td_4_6"] * df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),
        td_7_9_neutral=("td_7_9", lambda s: int((df.loc[s.index, "td_7_9"] * df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),
        td_10p_neutral=("td_10p", lambda s: int((df.loc[s.index, "td_10p"] * df.loc[s.index, "neutral"] * df.loc[s.index, "is_rush"]).sum())),

        # Production
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rush_touchdown", "sum"),
        rush_1st_downs=("first_down_rush", "sum"),
        explosive10=("explosive10", "sum"),
        explosive15=("explosive15", "sum"),
        explosive20=("explosive20", "sum"),
        stuffed_le0=("stuffed_le0", "sum"),
        tfl=("tackled_for_loss", "sum"),
        epa_sum=("epa_rush", "sum"),   # rush-only
        wpa_sum=("wpa_rush", "sum"),   # rush-only

        rz_rush_yards=("rushing_yards", lambda s: float((df.loc[s.index, "rushing_yards"] * df.loc[s.index, "rz"]).sum())),
        i10_rush_yards=("rushing_yards", lambda s: float((df.loc[s.index, "rushing_yards"] * df.loc[s.index, "i10"]).sum())),
        i5_rush_tds=("rush_touchdown", lambda s: int((df.loc[s.index, "rush_touchdown"] * df.loc[s.index, "i5"]).sum())),

        # Direction counts
        dir_left_end=("dir_left_end", "sum"),
        dir_left_tackle=("dir_left_tackle", "sum"),
        dir_left_guard=("dir_left_guard", "sum"),
        dir_middle=("dir_middle", "sum"),
        dir_right_guard=("dir_right_guard", "sum"),
        dir_right_tackle=("dir_right_tackle", "sum"),
        dir_right_end=("dir_right_end", "sum"),

        # Formation counts
        shotgun_carries=("shotgun_rush", "sum"),
        undercenter_carries=("undercenter_rush", "sum"),
        nohuddle_carries=("nohuddle_rush", "sum"),

        # Ball security & penalties
        fumbles=("fumble", lambda s: int((df.loc[s.index, "fumble"] * df.loc[s.index, "is_rush"]).sum())),
        fumbles_lost=("fumble_lost", lambda s: int((df.loc[s.index, "fumble_lost"] * df.loc[s.index, "is_rush"]).sum())),
        rush_def_penalties=("rush_def_pen", "sum"),
        rush_def_penalty_yards=("rush_def_pen_yds", "sum"),
    ).reset_index()

    # ---- Merge game metadata ----
    pg = per_game.merge(game_fields, on="game_id", how="left")

    # ---- Derived metrics ----
    int_cols = [
        "carries","neutral_carries","ed_neutral_carries","ed_carries_all",
        "h1_carries_all","h2_carries_all","third_down_carries","fourth_down_carries",
        "short_yds_carries","four_minute_carries","two_minute_carries",
        "rz_carries","i10_carries","i5_carries","gtg_carries",
        "td_1_3_carries","td_4_6_carries","td_7_9_carries","td_10p_carries",
        "td_1_3_neutral","td_4_6_neutral","td_7_9_neutral","td_10p_neutral",
        "rush_tds","rush_1st_downs","explosive10","explosive15","explosive20",
        "stuffed_le0","tfl","i5_rush_tds",
        "dir_left_end","dir_left_tackle","dir_left_guard","dir_middle",
        "dir_right_guard","dir_right_tackle","dir_right_end",
        "shotgun_carries","undercenter_carries","nohuddle_carries",
        "fumbles","fumbles_lost","rush_def_penalties"
    ]
    for c in int_cols:
        if c in pg.columns:
            pg[c] = pg[c].fillna(0).astype(int)

    # Helper for safe division (zero-with-indicator approach)
    def _safe_div(num, den, fill=0.0):
        return np.where(den > 0, num / den, fill)

    # Efficiencies / rates
    pg["ypc"] = pg["rush_yards"] / pg["carries"].replace(0, np.nan)
    pg["epa_per_rush"] = pg["epa_sum"] / pg["carries"].replace(0, np.nan)
    pg["wpa_per_rush"] = pg["wpa_sum"] / pg["carries"].replace(0, np.nan)

    # Neutral rates / shares (zero-with-indicator)
    pg["neutral_carries_per_game"] = pg["neutral_carries"]
    pg["has_neutral"] = (pg["neutral_carries"] > 0).astype(int)
    pg["ed_neutral_carry_rate"] = _safe_div(pg["ed_neutral_carries"], pg["neutral_carries"], 0.0)

    # Non-neutral counterparts
    pg["ed_carry_share_all"] = pg["ed_carries_all"] / pg["carries"].replace(0, np.nan)
    pg["carry_share_h1_all"] = pg["h1_carries_all"] / pg["carries"].replace(0, np.nan)
    pg["carry_share_h2_all"] = pg["h2_carries_all"] / pg["carries"].replace(0, np.nan)

    # 3rd-down distance shares (overall + neutral; neutral uses safe division)
    for b in ["td_1_3","td_4_6","td_7_9","td_10p"]:
        pg[f"{b}_share"] = pg[f"{b}_carries"] / pg["carries"].replace(0, np.nan)
        pg[f"{b}_neutral_share"] = _safe_div(pg[f"{b}_neutral"], pg["neutral_carries"], 0.0)

    # Direction shares
    dir_bins = ["left_end","left_tackle","left_guard","middle","right_guard","right_tackle","right_end"]
    for bin_ in dir_bins:
        col = f"dir_{bin_}"
        if col in pg.columns:
            pg[f"{col}_share"] = pg[col] / pg["carries"].replace(0, np.nan)

    # Formation shares
    pg["shotgun_share_carries"] = pg["shotgun_carries"] / pg["carries"].replace(0, np.nan)
    pg["undercenter_share_carries"] = pg["undercenter_carries"] / pg["carries"].replace(0, np.nan)
    pg["nohuddle_share_carries"] = pg["nohuddle_carries"] / pg["carries"].replace(0, np.nan)

    # Explosive & negative play rates
    pg["explosive10_rate"] = pg["explosive10"] / pg["carries"].replace(0, np.nan)
    pg["explosive15_rate"] = pg["explosive15"] / pg["carries"].replace(0, np.nan)
    pg["explosive20_rate"] = pg["explosive20"] / pg["carries"].replace(0, np.nan)
    pg["stuffed_rate"] = pg["stuffed_le0"] / pg["carries"].replace(0, np.nan)
    pg["tfl_rate"] = pg["tfl"] / pg["carries"].replace(0, np.nan)
    pg["fumble_rate"] = pg["fumbles"] / pg["carries"].replace(0, np.nan)
    pg["fumble_lost_rate"] = pg["fumbles_lost"] / pg["carries"].replace(0, np.nan)

    # ---- Remove kneel-only (zero-carry) rows ----
    if "carries" in pg.columns:
        pg = pg[pg["carries"] > 0].copy()

    # ---- Final select (unchanged except new has_neutral) ----
    keep = [
        "season","week","game_date","game_id","home_team","away_team","home_score","away_score",
        "posteam", rid_col, rname_col,

        # Volume
        "carries","neutral_carries","neutral_carries_per_game",
        "ed_neutral_carries","ed_carries_all","ed_carry_share_all",
        "h1_carries_all","h2_carries_all","carry_share_h1_all","carry_share_h2_all",
        "third_down_carries","fourth_down_carries","short_yds_carries",
        "four_minute_carries","two_minute_carries",
        "rz_carries","i10_carries","i5_carries","gtg_carries",

        # Production
        "rush_yards","rush_tds","rush_1st_downs",
        "explosive10","explosive15","explosive20",
        "stuffed_le0","tfl","epa_sum","wpa_sum",
        "rz_rush_yards","i10_rush_yards","i5_rush_tds",

        # Efficiency / rates
        "ypc","epa_per_rush","wpa_per_rush",
        "ed_neutral_carry_rate",
        "explosive10_rate","explosive15_rate","explosive20_rate",
        "stuffed_rate","tfl_rate","fumble_rate","fumble_lost_rate",

        # 3rd down distance shares
        "td_1_3_share","td_4_6_share","td_7_9_share","td_10p_share",
        "td_1_3_neutral_share","td_4_6_neutral_share","td_7_9_neutral_share","td_10p_neutral_share",

        # Direction shares
        "dir_left_end","dir_left_tackle","dir_left_guard","dir_middle",
        "dir_right_guard","dir_right_tackle","dir_right_end",
        "dir_left_end_share","dir_left_tackle_share","dir_left_guard_share","dir_middle_share",
        "dir_right_guard_share","dir_right_tackle_share","dir_right_end_share",

        # Formation
        "shotgun_carries","undercenter_carries","nohuddle_carries",
        "shotgun_share_carries","undercenter_share_carries","nohuddle_share_carries",

        # Ball security / penalties
        "fumbles","fumbles_lost","rush_def_penalties","rush_def_penalty_yards",

        # Indicators
        "has_neutral",
    ]
    keep = [c for c in keep if c in pg.columns]
    pg = pg[keep].reset_index(drop=True)

    pg.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(pg):,} rows and {pg.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", required=True, help="Path to combined PBP parquet")
    ap.add_argument("--out", required=True, help="Path to write rb_game.parquet")
    args = ap.parse_args()
    main(args.pbp, args.out)
