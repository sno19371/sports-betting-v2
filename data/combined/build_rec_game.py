# build_rec_game.py
# One row per receiver per game with modeling features (concise set).
# Targets = pass_attempt=1 & receiver_id present & play=1 & aborted_play=0 & qb_spike=0
# Neutral = (qtr in {1,2,3}) AND (score_differential in [-7, +7]).

import argparse
import numpy as np
import pandas as pd


def main(pbp_path: str, out_path: str):
    df = pd.read_parquet(pbp_path)

    # ---- Column aliases (support nflverse variants) ----
    rid_col = (
        "receiver_id"
        if "receiver_id" in df.columns
        else ("receiver_player_id" if "receiver_player_id" in df.columns else "receiver")
    )
    rname_col = (
        "receiver"
        if "receiver" in df.columns
        else ("receiver_player_name" if "receiver_player_name" in df.columns else "receiver")
    )

    # ---- Numeric/bool flags; fill + cast ----
    for c in [
        "pass_attempt","complete_pass","qb_spike","aborted_play","play",
        "shotgun","no_huddle","pass_touchdown","first_down_pass",
        "penalty","fumble","fumble_lost","sack"
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    for c in ["epa","wpa","receiving_yards","yards_after_catch","air_yards","yardline_100","ydstogo"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # ---- Situational flags ----
    df["neutral"] = (
        df["qtr"].between(1, 3, inclusive="both")
        & df["score_differential"].between(-7, 7, inclusive="both")
    ).astype(int)
    df["early_down"] = df["down"].isin([1, 2]).astype(int)
    df["half_1"] = df["qtr"].isin([1, 2]).astype(int)
    df["half_2"] = df["qtr"].isin([3, 4, 5]).astype(int)  # OT -> 2H
    df["third_down"] = (df["down"] == 3).astype(int)
    df["fourth_down"] = (df["down"] == 4).astype(int)
    ytg = df["ydstogo"].astype(float)
    df["td_1_3"] = ((df["third_down"] == 1) & (ytg.between(1, 3))).astype(int)
    df["td_4_6"] = ((df["third_down"] == 1) & (ytg.between(4, 6))).astype(int)
    df["td_7_9"] = ((df["third_down"] == 1) & (ytg.between(7, 9))).astype(int)
    df["td_10p"] = ((df["third_down"] == 1) & (ytg >= 10)).astype(int)
    df["two_minute"] = (df["quarter_seconds_remaining"] <= 120) & (df["qtr"].isin([2, 4]))
    df["two_minute"] = df["two_minute"].astype(int)
    df["rz"] = (df["yardline_100"] <= 20).astype(int)
    df["i10"] = (df["yardline_100"] <= 10).astype(int)
    df["gtg"] = df["goal_to_go"].fillna(0).astype(int)

    # ---- Target definition (exclude spikes/aborts/no-plays) ----
    df["is_target"] = (
        (df.get("pass_attempt", 0) == 1)
        & df[rid_col].notna()
        & (df.get("play", 1) == 1)
        & (df.get("aborted_play", 0) == 0)
        & (df.get("qb_spike", 0) == 0)
    ).astype(int)

    # ---- Receptions only on targeted plays ----
    df["is_catch"] = (df["is_target"] == 1) & (df.get("complete_pass", 0) == 1)

    # ---- End-zone target/TD (requires both values present) ----
    ay = df["air_yards"]
    yl = df["yardline_100"]
    df["endzone_target"] = ((df["is_target"] == 1) & ay.notna() & yl.notna() & (ay >= yl)).astype(int)
    df["endzone_td"] = ((df.get("pass_touchdown", 0) == 1) & ay.notna() & yl.notna() & (ay >= yl)).astype(int)

    # ---- Formation / tempo on targets ----
    df["shotgun_tgt"] = (df.get("shotgun", 0) * df["is_target"]).astype(int)
    df["undercenter_tgt"] = (((1 - df.get("shotgun", 0)) * df["is_target"]).astype(int))
    df["nohuddle_tgt"] = (df.get("no_huddle", 0) * df["is_target"]).astype(int)

    # ---- Pass location on targets ----
    loc = df.get("pass_location")
    df["loc_left_tgt"] = ((loc == "left") & (df["is_target"] == 1)).astype(int)
    df["loc_middle_tgt"] = ((loc == "middle") & (df["is_target"] == 1)).astype(int)
    df["loc_right_tgt"] = ((loc == "right") & (df["is_target"] == 1)).astype(int)

    # ---- Depth buckets on targets (by air_yards) ----
    df["blos_tgt"] = ((df["is_target"] == 1) & ay.notna() & (ay < 0)).astype(int)
    df["short_tgt"] = ((df["is_target"] == 1) & ay.notna() & (ay >= 0) & (ay <= 9)).astype(int)
    df["intermediate_tgt"] = ((df["is_target"] == 1) & ay.notna() & (ay >= 10) & (ay <= 19)).astype(int)
    df["deep_tgt"] = ((df["is_target"] == 1) & ay.notna() & (ay >= 20)).astype(int)

    # ---- Explosive receptions ----
    ry = df.get("receiving_yards", pd.Series(0.0, index=df.index))
    df["explosive15_rec"] = ((df["is_catch"] == 1) & (ry >= 15)).astype(int)
    df["explosive20_rec"] = ((df["is_catch"] == 1) & (ry >= 20)).astype(int)

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

    # ---- Team denominators per game/team ----
    team_keys = ["season","week","game_id","posteam"]
    team = (
        df.groupby(team_keys, dropna=False)
          .agg(
              team_att_ns=("pass_attempt", lambda s: int((df.loc[s.index, "pass_attempt"] * (1 - df.loc[s.index, "qb_spike"]) * (1 - df.loc[s.index, "aborted_play"]) * (df.loc[s.index, "play"])).sum())),
              team_air_on_tgts=("air_yards", lambda s: float((df.loc[s.index, "air_yards"].fillna(0) * df.loc[s.index, "is_target"]).sum())),
          )
          .reset_index()
    )

    # ---- Grouping keys ----
    keys = ["season","week","game_id","posteam", rid_col, rname_col]

    # ---- Per-game receiver aggregates ----
    g = df[df[rid_col].notna()].groupby(keys, dropna=False)

    per_game = g.agg(
        # Volume
        targets=("is_target", "sum"),
        receptions=("is_catch", "sum"),
        ed_targets=("early_down", lambda s: int((df.loc[s.index, "early_down"] * df.loc[s.index, "is_target"]).sum())),
        h1_targets=("half_1", lambda s: int((df.loc[s.index, "half_1"] * df.loc[s.index, "is_target"]).sum())),
        h2_targets=("half_2", lambda s: int((df.loc[s.index, "half_2"] * df.loc[s.index, "is_target"]).sum())),
        neutral_targets=("neutral", lambda s: int((df.loc[s.index, "neutral"] * df.loc[s.index, "is_target"]).sum())),
        third_down_targets=("third_down", lambda s: int((df.loc[s.index, "third_down"] * df.loc[s.index, "is_target"]).sum())),
        td_1_3_targets=("td_1_3", lambda s: int((df.loc[s.index, "td_1_3"] * df.loc[s.index, "is_target"]).sum())),
        td_4_6_targets=("td_4_6", lambda s: int((df.loc[s.index, "td_4_6"] * df.loc[s.index, "is_target"]).sum())),
        td_7_9_targets=("td_7_9", lambda s: int((df.loc[s.index, "td_7_9"] * df.loc[s.index, "is_target"]).sum())),
        td_10p_targets=("td_10p", lambda s: int((df.loc[s.index, "td_10p"] * df.loc[s.index, "is_target"]).sum())),
        fourth_down_targets=("fourth_down", lambda s: int((df.loc[s.index, "fourth_down"] * df.loc[s.index, "is_target"]).sum())),
        two_minute_targets=("two_minute", lambda s: int((df.loc[s.index, "two_minute"] * df.loc[s.index, "is_target"]).sum())),
        rz_targets=("rz", lambda s: int((df.loc[s.index, "rz"] * df.loc[s.index, "is_target"]).sum())),
        i10_targets=("i10", lambda s: int((df.loc[s.index, "i10"] * df.loc[s.index, "is_target"]).sum())),
        gtg_targets=("gtg", lambda s: int((df.loc[s.index, "gtg"] * df.loc[s.index, "is_target"]).sum())),
        endzone_targets=("endzone_target", "sum"),

        # Location / depth / formation on targets
        left_targets=("loc_left_tgt", "sum"),
        middle_targets=("loc_middle_tgt", "sum"),
        right_targets=("loc_right_tgt", "sum"),
        blos_targets=("blos_tgt", "sum"),
        short_targets=("short_tgt", "sum"),
        intermediate_targets=("intermediate_tgt", "sum"),
        deep_targets=("deep_tgt", "sum"),
        shotgun_targets=("shotgun_tgt", "sum"),
        undercenter_targets=("undercenter_tgt", "sum"),
        nohuddle_targets=("nohuddle_tgt", "sum"),

        # Production (on targets/receptions)
        rec_yards=("receiving_yards", lambda s: float((df.loc[s.index, "receiving_yards"].fillna(0) * df.loc[s.index, "is_catch"]).sum())),
        yac=("yards_after_catch", lambda s: float((df.loc[s.index, "yards_after_catch"].fillna(0) * df.loc[s.index, "is_catch"]).sum())),
        air_yards=("air_yards", lambda s: float((df.loc[s.index, "air_yards"].fillna(0) * df.loc[s.index, "is_target"]).sum())),
        rec_tds=("pass_touchdown", lambda s: int((df.loc[s.index, "pass_touchdown"] * df.loc[s.index, "is_target"]).sum())),
        fd_receptions=("first_down_pass", lambda s: int((df.loc[s.index, "first_down_pass"] * df.loc[s.index, "is_catch"]).sum())),
        explosive15_receptions=("explosive15_rec", "sum"),
        explosive20_receptions=("explosive20_rec", "sum"),
        rz_rec_yards=("receiving_yards", lambda s: float((df.loc[s.index, "receiving_yards"].fillna(0) * df.loc[s.index, "is_catch"] * df.loc[s.index, "rz"]).sum())),
        i10_rec_yards=("receiving_yards", lambda s: float((df.loc[s.index, "receiving_yards"].fillna(0) * df.loc[s.index, "is_catch"] * df.loc[s.index, "i10"]).sum())),
        endzone_tds=("endzone_td", "sum"),
        epa_sum=("epa", lambda s: float((df.loc[s.index, "epa"].fillna(0) * df.loc[s.index, "is_target"]).sum())),
        wpa_sum=("wpa", lambda s: float((df.loc[s.index, "wpa"].fillna(0) * df.loc[s.index, "is_target"]).sum())),
    ).reset_index()

    # ---- Merge game metadata + team denoms ----
    pg = (
        per_game.merge(game_fields, on="game_id", how="left")
                .merge(team, on=team_keys, how="left")
    )

    # ---- Cast integer count cols ----
    int_cols = [
        "targets","receptions","ed_targets","h1_targets","h2_targets","neutral_targets",
        "third_down_targets","td_1_3_targets","td_4_6_targets","td_7_9_targets","td_10p_targets",
        "fourth_down_targets","two_minute_targets","rz_targets","i10_targets","gtg_targets","endzone_targets",
        "left_targets","middle_targets","right_targets",
        "blos_targets","short_targets","intermediate_targets","deep_targets",
        "shotgun_targets","undercenter_targets","nohuddle_targets",
        "rec_tds","fd_receptions","explosive15_receptions","explosive20_receptions",
        "endzone_tds"
    ]
    for c in int_cols:
        if c in pg.columns:
            pg[c] = pg[c].fillna(0).astype(int)

    # ---- Zero-with-indicator shares/rates ----
    # Denominator flags
    pg["has_targets"] = (pg["targets"] > 0).astype(int)
    pg["has_receptions"] = (pg["receptions"] > 0).astype(int)
    pg["has_team_att"] = (pg["team_att_ns"].fillna(0) > 0).astype(int)
    pg["has_team_air"] = (pg["team_air_on_tgts"].fillna(0) > 0).astype(int)
    pg["has_air"] = (pg["air_yards"].fillna(0) > 0).astype(int)

    # Team-level shares
    pg["target_share"] = np.where(pg["has_team_att"] == 1, pg["targets"] / pg["team_att_ns"].replace(0, np.nan), 0.0)
    pg["air_yards_share"] = np.where(pg["has_team_air"] == 1, pg["air_yards"] / pg["team_air_on_tgts"].replace(0, np.nan), 0.0)
    pg["wopr"] = 0.7 * pg["target_share"] + 0.3 * pg["air_yards_share"]

    # Efficiencies
    pg["catch_rate"] = np.where(pg["has_targets"] == 1, pg["receptions"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["ypt"] = np.where(pg["has_targets"] == 1, pg["rec_yards"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["ypr"] = np.where(pg["has_receptions"] == 1, pg["rec_yards"] / pg["receptions"].replace(0, np.nan), 0.0)
    pg["adot"] = np.where(pg["has_targets"] == 1, pg["air_yards"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["yac_per_target"] = np.where(pg["has_targets"] == 1, pg["yac"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["yac_per_rec"] = np.where(pg["has_receptions"] == 1, pg["yac"] / pg["receptions"].replace(0, np.nan), 0.0)
    pg["racr"] = np.where(pg["has_air"] == 1, pg["rec_yards"] / pg["air_yards"].replace(0, np.nan), 0.0)
    pg["td_rate"] = np.where(pg["has_targets"] == 1, pg["rec_tds"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["fd_rate"] = np.where(pg["has_targets"] == 1, pg["fd_receptions"] / pg["targets"].replace(0, np.nan), 0.0)
    pg["explosive15_rate"] = np.where(pg["has_receptions"] == 1, pg["explosive15_receptions"] / pg["receptions"].replace(0, np.nan), 0.0)
    pg["explosive20_rate"] = np.where(pg["has_receptions"] == 1, pg["explosive20_receptions"] / pg["receptions"].replace(0, np.nan), 0.0)

    # Formation / location / depth shares (vs targets)
    for base in [
        "shotgun","undercenter","nohuddle",
        "left","middle","right",
        "blos","short","intermediate","deep"
    ]:
        num = f"{base}_targets"
        den = "targets"
        out = f"{base}_share_targets"
        if num in pg.columns:
            pg[out] = np.where(pg["has_targets"] == 1, pg[num] / pg[den].replace(0, np.nan), 0.0)

    # ---- Keep only rows with at least one target ----
    pg = pg[pg["targets"] > 0].copy()

    # ---- Final select (concise; no rolling) ----
    keep = [
        "season","week","game_date","game_id","home_team","away_team","home_score","away_score",
        "posteam", rid_col, rname_col,

        # Volume
        "targets","receptions","target_share","air_yards_share","wopr",
        "ed_targets","h1_targets","h2_targets","neutral_targets",
        "third_down_targets","td_1_3_targets","td_4_6_targets","td_7_9_targets","td_10p_targets",
        "fourth_down_targets","two_minute_targets",
        "rz_targets","i10_targets","gtg_targets","endzone_targets",

        # Location / depth / formation
        "left_targets","middle_targets","right_targets",
        "blos_targets","short_targets","intermediate_targets","deep_targets",
        "shotgun_targets","undercenter_targets","nohuddle_targets",
        "left_share_targets","middle_share_targets","right_share_targets",
        "blos_share_targets","short_share_targets","intermediate_share_targets","deep_share_targets",
        "shotgun_share_targets","undercenter_share_targets","nohuddle_share_targets",

        # Production
        "rec_yards","yac","air_yards",
        "rec_tds","fd_receptions","explosive15_receptions","explosive20_receptions",
        "rz_rec_yards","i10_rec_yards","endzone_tds",
        "epa_sum","wpa_sum",

        # Efficiency
        "catch_rate","ypt","ypr","adot","yac_per_target","yac_per_rec","racr","td_rate","fd_rate",
        "explosive15_rate","explosive20_rate",

        # Team denoms + indicators (useful for modeling)
        "team_att_ns","team_air_on_tgts",
        "has_targets","has_receptions","has_team_att","has_team_air","has_air",
    ]
    keep = [c for c in keep if c in pg.columns]
    pg = pg[keep].reset_index(drop=True)

    pg.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(pg):,} rows and {pg.shape[1]} columns")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", required=True, help="Path to combined PBP parquet")
    ap.add_argument("--out", required=True, help="Path to write rec_game.parquet")
    args = ap.parse_args()
    main(args.pbp, args.out)
