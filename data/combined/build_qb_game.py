# build_qb_game.py
# One row per passer per game with modeling features.
# Neutral = (qtr in {1,2,3}) AND (score_differential in [-7, +7])  (no WP).
# Adds home_team, away_team, final home/away score, and game_date.
# Includes non-neutral counterparts for all neutral stats (early-down pass rate, half split pass rates,
# and 3rd-down distance shares).

import argparse
import numpy as np
import pandas as pd


def main(pbp_path: str, out_path: str):
    df = pd.read_parquet(pbp_path)

    # ---- ID / name columns (support both nflverse variants) ----
    pid_col = "passer_id" if "passer_id" in df.columns else "passer_player_id"
    pname_col = "passer" if "passer" in df.columns else "passer_player_name"

    # ---- numeric flags ----
    for c in [
        "qb_dropback", "qb_scramble", "pass_attempt", "sack", "interception",
        "rush_attempt", "complete_pass", "pass_touchdown"
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # ---- dropbacks ----
    if "qb_dropback" in df.columns:
        df["dropback"] = df["qb_dropback"].astype(int)
    else:
        df["dropback"] = (
            df.get("pass_attempt", 0).astype(int)
            | df.get("sack", 0).astype(int)
            | df.get("qb_scramble", 0).astype(int)
        ).astype(int)

    # Sack yards (positive)
    df["sack_yards"] = np.where(df.get("sack", 0) == 1, -df.get("yards_gained", 0).fillna(0), 0)

    # Convenience
    df["is_pass_play"] = (df.get("pass_attempt", 0) == 1)

    # ---- Neutral / situational flags ----
    df["neutral"] = (
        df["qtr"].between(1, 3, inclusive="both")
        & df["score_differential"].between(-7, 7, inclusive="both")
    ).astype(int)
    df["early_down"] = df["down"].isin([1, 2]).astype(int)
    df["half_1"] = df["qtr"].isin([1, 2]).astype(int)
    df["half_2"] = df["qtr"].isin([3, 4, 5]).astype(int)  # OT -> 2H

    # ---- RZ / i10 / GTG ----
    df["rz"] = (df["yardline_100"] <= 20).astype(int)
    df["i10"] = (df["yardline_100"] <= 10).astype(int)
    df["gtg"] = df["goal_to_go"].fillna(0).astype(int)

    # ---- 3rd-down distance bins ----
    df["third_down"] = (df["down"] == 3).astype(int)
    ytg = df["ydstogo"].astype(float)
    df["td_1_3"] = ((df["third_down"] == 1) & (ytg.between(1, 3))).astype(int)
    df["td_4_6"] = ((df["third_down"] == 1) & (ytg.between(4, 6))).astype(int)
    df["td_7_9"] = ((df["third_down"] == 1) & (ytg.between(7, 9))).astype(int)
    df["td_10p"] = ((df["third_down"] == 1) & (ytg >= 10)).astype(int)

    # ---- Depth buckets (air_yards) ----
    ay = df["air_yards"]
    df["depth_blos"]   = ((df["is_pass_play"]) & (ay < 0)).astype(int)
    df["depth_1_9"]    = ((df["is_pass_play"]) & (ay.between(0, 9))).astype(int)
    df["depth_10_19"]  = ((df["is_pass_play"]) & (ay.between(10, 19))).astype(int)
    df["depth_20p"]    = ((df["is_pass_play"]) & (ay >= 20)).astype(int)

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
    keys = ["season", "week", "game_id", "posteam", pid_col, pname_col]

    # ---- Target concentration (WR1/TE1 proxy) ----
    pass_rows = df[df["is_pass_play"] & df[pid_col].notna()].copy()
    rc_counts = (
        pass_rows.groupby(keys + ["receiver_player_id"], dropna=False)
                 .size().rename("rc_targets").reset_index()
    )
    att_per_game = (
        pass_rows.groupby(keys)
                 .size().rename("att_total").reset_index()
    )
    top_share = (
        rc_counts.sort_values(keys + ["rc_targets"], ascending=[True]*len(keys) + [False])
                 .groupby(keys).agg(top_rc_targets=("rc_targets", "max")).reset_index()
    )
    tgt_conc = att_per_game.merge(top_share, on=keys, how="left")
    tgt_conc["top_target_share"] = (tgt_conc["top_rc_targets"] / tgt_conc["att_total"]).fillna(0)

    # ---- Per-game passer aggregates ----
    g = df[df[pid_col].notna()].groupby(keys, dropna=False)
    per_game = g.agg(
        dropbacks=("dropback", "sum"),
        pass_att=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        sacks=("sack", "sum"),
        scrambles=("qb_scramble", "sum"),
        ints=("interception", "sum"),
        pass_td=("pass_touchdown", "sum"),
        epa_sum=("epa", "sum"),
        pass_yards=("passing_yards", "sum"),
        air_yards_sum=("air_yards", "sum"),
        cpoe_mean=("cpoe", "mean"),

        shotgun_dropbacks=("shotgun",
            lambda s: (df.loc[s.index, "shotgun"].fillna(0) * df.loc[s.index, "dropback"]).sum()),
        undercen_dropbacks=("shotgun",
            lambda s: ((1 - df.loc[s.index, "shotgun"].fillna(0)) * df.loc[s.index, "dropback"]).sum()),

        # NEUTRAL subsets
        neutral_dropbacks=("dropback",
            lambda s: int(df.loc[s.index, "neutral"].dot(df.loc[s.index, "dropback"]))),

        ed_neutral_pass_att=("pass_attempt",
            lambda s: int((df.loc[s.index, "early_down"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "pass_attempt"]))),

        h1_neutral_db=("dropback",
            lambda s: int((df.loc[s.index, "half_1"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),
        h1_neutral_att=("pass_attempt",
            lambda s: int((df.loc[s.index, "half_1"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "pass_attempt"]))),
        h2_neutral_db=("dropback",
            lambda s: int((df.loc[s.index, "half_2"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),
        h2_neutral_att=("pass_attempt",
            lambda s: int((df.loc[s.index, "half_2"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "pass_attempt"]))),

        # OVERALL (non-neutral) counterparts
        ed_att_all=("pass_attempt",
            lambda s: int(df.loc[s.index, "early_down"].dot(df.loc[s.index, "pass_attempt"]))),
        ed_db_all=("dropback",
            lambda s: int(df.loc[s.index, "early_down"].dot(df.loc[s.index, "dropback"]))),

        h1_db_all=("dropback",
            lambda s: int(df.loc[s.index, "half_1"].dot(df.loc[s.index, "dropback"]))),
        h1_att_all=("pass_attempt",
            lambda s: int(df.loc[s.index, "half_1"].dot(df.loc[s.index, "pass_attempt"]))),
        h2_db_all=("dropback",
            lambda s: int(df.loc[s.index, "half_2"].dot(df.loc[s.index, "dropback"]))),
        h2_att_all=("pass_attempt",
            lambda s: int(df.loc[s.index, "half_2"].dot(df.loc[s.index, "pass_attempt"]))),

        # 3rd down distance dropbacks (overall & neutral counts)
        db_td_1_3=("dropback", lambda s: int(df.loc[s.index, "td_1_3"].dot(df.loc[s.index, "dropback"]))),
        db_td_4_6=("dropback", lambda s: int(df.loc[s.index, "td_4_6"].dot(df.loc[s.index, "dropback"]))),
        db_td_7_9=("dropback", lambda s: int(df.loc[s.index, "td_7_9"].dot(df.loc[s.index, "dropback"]))),
        db_td_10p=("dropback", lambda s: int(df.loc[s.index, "td_10p"].dot(df.loc[s.index, "dropback"]))),

        db_td_1_3_neu=("dropback",
            lambda s: int((df.loc[s.index, "td_1_3"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),
        db_td_4_6_neu=("dropback",
            lambda s: int((df.loc[s.index, "td_4_6"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),
        db_td_7_9_neu=("dropback",
            lambda s: int((df.loc[s.index, "td_7_9"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),
        db_td_10p_neu=("dropback",
            lambda s: int((df.loc[s.index, "td_10p"] * df.loc[s.index, "neutral"])
                          .dot(df.loc[s.index, "dropback"]))),

        # Scoring context (overall counts)
        rz_att=("pass_attempt", lambda s: int(df.loc[s.index, "rz"].dot(df.loc[s.index, "pass_attempt"]))),
        i10_att=("pass_attempt", lambda s: int(df.loc[s.index, "i10"].dot(df.loc[s.index, "pass_attempt"]))),
        gtg_db=("dropback",     lambda s: int(df.loc[s.index, "gtg"].dot(df.loc[s.index, "dropback"]))),

        # Depth counts (overall)
        depth_blos=("depth_blos", "sum"),
        depth_1_9=("depth_1_9", "sum"),
        depth_10_19=("depth_10_19", "sum"),
        depth_20p=("depth_20p", "sum"),
        sack_yards=("sack_yards", "sum"),
    ).reset_index()

    # Merge target concentration
    pg = per_game.merge(tgt_conc[keys + ["top_target_share"]], on=keys, how="left")

    # Merge game metadata
    pg = pg.merge(game_fields, on="game_id", how="left")

    # ---- Derived metrics ----
    for c in ["dropbacks", "pass_att", "sacks", "scrambles",
              "neutral_dropbacks", "ed_neutral_pass_att",
              "h1_neutral_db", "h1_neutral_att", "h2_neutral_db", "h2_neutral_att",
              "ed_att_all", "ed_db_all", "h1_db_all", "h1_att_all", "h2_db_all", "h2_att_all"]:
        if c in pg.columns:
            pg[c] = pg[c].fillna(0).astype(int)

    # Efficiencies
    denom_db = (pg["pass_att"] + pg["sacks"]).replace(0, np.nan)
    pg["ypa"] = pg["pass_yards"] / pg["pass_att"].replace(0, np.nan)
    pg["anya"] = (pg["pass_yards"] + 20 * pg["pass_td"] - 45 * pg["ints"] - pg["sack_yards"]) / denom_db
    pg["epa_per_db"] = pg["epa_sum"] / pg["dropbacks"].replace(0, np.nan)
    pg["ay_per_att"] = pg["air_yards_sum"] / pg["pass_att"].replace(0, np.nan)
    if "cpoe_mean" in pg.columns:
        pg["cpoe_mean"] = pg["cpoe_mean"].astype(float)

    # Negative volume / mix
    pg["sack_rate_db"] = pg["sacks"] / pg["dropbacks"].replace(0, np.nan)
    pg["int_rate_db"] = pg["ints"] / pg["dropbacks"].replace(0, np.nan)
    pg["scramble_rate_db"] = pg["scrambles"] / pg["dropbacks"].replace(0, np.nan)
    pg["shotgun_share_db"] = pg["shotgun_dropbacks"] / pg["dropbacks"].replace(0, np.nan)
    pg["undercenter_share_db"] = pg["undercen_dropbacks"] / pg["dropbacks"].replace(0, np.nan)

    # Neutral rates
    pg["neutral_db_per_game"] = pg["neutral_dropbacks"]
    pg["ed_neutral_pass_rate"] = pg["ed_neutral_pass_att"] / pg["neutral_dropbacks"].replace(0, np.nan)
    pg["pass_rate_neutral_h1"] = pg["h1_neutral_att"] / pg["h1_neutral_db"].replace(0, np.nan)
    pg["pass_rate_neutral_h2"] = pg["h2_neutral_att"] / pg["h2_neutral_db"].replace(0, np.nan)

    # Non-neutral counterparts
    pg["ed_pass_rate_all"] = pg["ed_att_all"] / pg["ed_db_all"].replace(0, np.nan)
    pg["pass_rate_h1_all"] = pg["h1_att_all"] / pg["h1_db_all"].replace(0, np.nan)
    pg["pass_rate_h2_all"] = pg["h2_att_all"] / pg["h2_db_all"].replace(0, np.nan)

    # 3rd-down shares (overall + neutral)
    for b in ["db_td_1_3", "db_td_4_6", "db_td_7_9", "db_td_10p"]:
        pg[f"{b}_share"] = pg[b] / pg["dropbacks"].replace(0, np.nan)
        pg[f"{b}_neutral_share"] = pg[f"{b}_neu"] / pg["neutral_dropbacks"].replace(0, np.nan)

    # Depth shares (overall)
    for b in ["depth_blos", "depth_1_9", "depth_10_19", "depth_20p"]:
        pg[f"{b}_share"] = pg[b] / pg["pass_att"].replace(0, np.nan)

    # ---- Rolling within (season, passer) by week ----
    pg = pg.sort_values(["season", pid_col, "week"])

    def _rolling(group, cols, windows=(3, 5)):
        for c in cols:
            for w in windows:
                group[f"{c}_roll{w}"] = group[c].rolling(w, min_periods=1).mean()
            group[f"{c}_season_avg_to_date"] = group[c].expanding().apply(
                lambda s: np.nan if len(s) <= 1 else np.nanmean(s[:-1]), raw=False
            )
        return group

    roll_cols = ["pass_att", "ypa", "anya", "epa_per_db", "ay_per_att"]
    pg = pg.groupby(["season", pid_col], group_keys=False).apply(_rolling, cols=roll_cols)

    # ---- Final select ----
    keep = [
        "season","week","game_date","game_id","home_team","away_team","home_score","away_score",
        "posteam", pid_col, pname_col,

        # Volume
        "dropbacks","pass_att","completions","sacks","scrambles",

        # Neutral & non-neutral rates
        "neutral_db_per_game",
        "ed_neutral_pass_rate","ed_pass_rate_all",
        "pass_rate_neutral_h1","pass_rate_neutral_h2",
        "pass_rate_h1_all","pass_rate_h2_all",

        # 3rd down shares (overall + neutral)
        "db_td_1_3_share","db_td_4_6_share","db_td_7_9_share","db_td_10p_share",
        "db_td_1_3_neutral_share","db_td_4_6_neutral_share","db_td_7_9_neutral_share","db_td_10p_neutral_share",

        # Efficiency
        "ypa","anya","epa_per_db","ay_per_att","cpoe_mean",

        # Negative volume / mix
        "sack_rate_db","int_rate_db","scramble_rate_db",
        "shotgun_share_db","undercenter_share_db",

        # Scoring context & target map
        "rz_att","i10_att","gtg_db","top_target_share",

        # Depth mix (overall shares)
        "depth_blos_share","depth_1_9_share","depth_10_19_share","depth_20p_share",

        # Rolling windows
        "pass_att_roll3","pass_att_roll5","pass_att_season_avg_to_date",
        "ypa_roll3","ypa_roll5","ypa_season_avg_to_date",
        "anya_roll3","anya_roll5","anya_season_avg_to_date",
        "epa_per_db_roll3","epa_per_db_roll5","epa_per_db_season_avg_to_date",
        "ay_per_att_roll3","ay_per_att_roll5","ay_per_att_season_avg_to_date",
    ]
    keep = [c for c in keep if c in pg.columns]
    pg = pg[keep].reset_index(drop=True)

    pg.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(pg):,} rows and {pg.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", required=True, help="Path to combined PBP parquet")
    ap.add_argument("--out", required=True, help="Path to write qb_game.parquet")
    args = ap.parse_args()
    main(args.pbp, args.out)
