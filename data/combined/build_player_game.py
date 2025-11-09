# build_player_game.py
"""
Join QB/RB/REC per-player game stats into one wide table.

- Keys: (game_id, posteam, player_id)
- For each role, we prefix all role-specific metrics with qb_/rb_/rec_
  and keep just: game_id, posteam, player_id, <role>_player for names.
- Base game columns (season, week, game_date, home/away teams/scores, posteam)
  are taken once from any source table and merged in.
- We zero-fill numeric role metrics and False-fill boolean flags so there are no NaNs
  in metrics/flags. Name columns become "" when missing.

Usage (PowerShell):
  python build_player_game.py `
    --qb  "path\to\qb_game.parquet" `
    --rb  "path\to\rb_game.parquet" `
    --rec "path\to\rec_game.parquet" `
    --out "path\to\joined_games.parquet"
"""

import argparse
import pandas as pd


BASE = [
    "season", "week", "game_date", "game_id",
    "home_team", "away_team", "home_score", "away_score", "posteam",
]
KEYS = ["game_id", "posteam", "player_id"]  # join keys


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _any_positive(df: pd.DataFrame, cols) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].fillna(0).astype(float).sum(axis=1).gt(0)


def prep_role(
    df: pd.DataFrame,
    role: str,
    id_col: str,
    name_col: str,
) -> pd.DataFrame:
    """
    Prepare a role-specific frame with:
      - columns: game_id, posteam, player_id, <role>_player, and prefixed role metrics
      - base columns are NOT carried here (added later from a base lookup)
    """
    out = pd.DataFrame(index=df.index)

    # keys
    out["game_id"] = df["game_id"]
    out["posteam"] = df["posteam"]
    out["player_id"] = df[id_col]
    out[f"{role}_player"] = df[name_col]

    # role metrics = everything except base + id/name + keys
    exclude = set(BASE) | {id_col, name_col, "player_id", f"{role}_player", "game_id", "posteam"}
    metric_cols = [c for c in df.columns if c not in exclude]

    # prefix metrics
    for c in metric_cols:
        out[f"{role}_{c}"] = df[c]

    # role-level "has" flags (added here for convenience)
    if role == "qb":
        # QB did a dropback if any of these > 0
        out["qb_has_db"] = _any_positive(df, ["dropbacks", "pass_att", "sacks", "scrambles"])
    elif role == "rb":
        # RB recorded a carry if any carry-like fields > 0
        out["rb_has_carries"] = _any_positive(df, ["carries", "neutral_carries", "h1_carries_all", "h2_carries_all"])
        # normalize has_neutral -> rb_has_neutral (if it existed, it was prefixed already)
        if "rb_has_neutral" not in out.columns and "has_neutral" in df.columns:
            out["rb_has_neutral"] = df["has_neutral"].fillna(False)
    elif role == "rec":
        # REC recorded a target if any target-like fields > 0
        out["rec_has_targets_flag"] = _any_positive(df, ["targets", "neutral_targets", "h1_targets", "h2_targets"])

    # ensure unique column names (just in case)
    out = out.loc[:, ~out.columns.duplicated()]

    return out


def build_base_lookup(qb: pd.DataFrame, rb: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """
    Collect BASE game columns from any source table (qb/rb/rec) and drop
    duplicates by (game_id, posteam). Only BASE columns are returned.
    """
    frames = []
    for src in (qb, rb, rec):
        have = [c for c in BASE if c in src.columns]
        if have:
            frames.append(src[have].copy())

    if not frames:
        raise RuntimeError("No base columns found in any input tables.")

    base = pd.concat(frames, ignore_index=True)
    # Keep only BASE columns (and ensure uniqueness/order)
    base = base.loc[:, dict.fromkeys([c for c in BASE if c in base.columns]).keys()]
    # Unique rows per (game_id, posteam)
    base = base.drop_duplicates(subset=["game_id", "posteam"])

    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qb", required=True, help="Path to qb_game.parquet")
    ap.add_argument("--rb", required=True, help="Path to rb_game.parquet")
    ap.add_argument("--rec", required=True, help="Path to rec_game.parquet")
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    # Read raw tables
    qb_raw = read_parquet(args.qb).copy()
    rb_raw = read_parquet(args.rb).copy()
    rec_raw = read_parquet(args.rec).copy()

    # Prepare role frames (no base columns in these)
    qb_df = prep_role(qb_raw, "qb", id_col="passer_id",   name_col="passer")
    rb_df = prep_role(rb_raw, "rb", id_col="rusher_id",   name_col="rusher")
    rec_df = prep_role(rec_raw, "rec", id_col="receiver_id", name_col="receiver")

    # Merge by keys, outer (players may appear in only one role)
    merged = qb_df.merge(rb_df, on=KEYS, how="outer")
    merged = merged.merge(rec_df, on=KEYS, how="outer")

    # Add unified player display name + role presence flags
    merged["player"] = (
        merged.get("qb_player", pd.Series(index=merged.index, dtype=object))
        .fillna("")
        .where(lambda s: s != "", merged.get("rb_player", "").fillna(""))
        .where(lambda s: s != "", merged.get("rec_player", "").fillna(""))
    )

    merged["has_qb"] = merged.get("qb_player", "").fillna("") != ""
    merged["has_rb"] = merged.get("rb_player", "").fillna("") != ""
    merged["has_rec"] = merged.get("rec_player", "").fillna("") != ""

    # Bring in base game columns once
    base_lookup = build_base_lookup(qb_raw, rb_raw, rec_raw)
    merged = merged.merge(base_lookup, on=["game_id", "posteam"], how="left")

    # === Fill missing values ===
    # 1) Flags we explicitly manage (fill False)
    flag_candidates = [
        "has_qb", "has_rb", "has_rec",
        "qb_has_db", "rb_has_carries", "rb_has_neutral",
        "rec_has_targets_flag", "rec_has_targets",  # rec_has_targets comes from rec metrics (if present)
    ]
    for col in flag_candidates:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    # 2) Names (fill empty string)
    for col in ["player", "qb_player", "rb_player", "rec_player"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(object)

    # 3) Numeric metrics (fill 0)
    num_cols = merged.select_dtypes(include=["number", "boolean"]).columns.tolist()
    # Exclude key columns from numeric fill (not needed but harmless); still fill NaNs in ints/floats
    merged[num_cols] = merged[num_cols].fillna(0)

    # 4) Object/string columns except the name columns already handled (fill "")
    obj_cols = merged.select_dtypes(include=["object", "string"]).columns.tolist()
    name_cols = {"player", "qb_player", "rb_player", "rec_player"}
    other_obj = [c for c in obj_cols if c not in name_cols]
    if other_obj:
        merged[other_obj] = merged[other_obj].fillna("")

    # Ensure unique column labels before ordering/writing
    if merged.columns.duplicated().any():
        dups = merged.columns[merged.columns.duplicated()].tolist()
        print(f"[warn] Dropping duplicate labels: {dups}")
        merged = merged.loc[:, ~merged.columns.duplicated()]

    # === Order columns ===
    front = [
        "season", "week", "game_date", "game_id",
        "home_team", "away_team", "home_score", "away_score",
        "posteam", "player_id", "player",
        "has_qb", "has_rb", "has_rec",
        "qb_player", "rb_player", "rec_player",
        "qb_has_db", "rb_has_carries", "rb_has_neutral",
        "rec_has_targets", "rec_has_targets_flag",
    ]
    front = [c for c in front if c in merged.columns]
    front_set = set(front)

    qb_cols = sorted([c for c in merged.columns if c.startswith("qb_") and c not in front_set])
    rb_cols = sorted([c for c in merged.columns if c.startswith("rb_") and c not in front_set])
    rec_cols = sorted([c for c in merged.columns if c.startswith("rec_") and c not in front_set])

    ordered = front + qb_cols + rb_cols + rec_cols
    # add leftovers once
    ordered += [c for c in merged.columns if c not in set(ordered)]

    # final de-dup on the selection list
    seen, ordered_unique = set(), []
    for c in ordered:
        if c in merged.columns and c not in seen:
            ordered_unique.append(c)
            seen.add(c)

    merged = merged[ordered_unique]

    # Final safety: ensure unique labels for Parquet
    if merged.columns.duplicated().any():
        dups = merged.columns[merged.columns.duplicated()].tolist()
        print(f"[warn] Dropping duplicate labels prior to write: {dups}")
        merged = merged.loc[:, ~merged.columns.duplicated()]

    # Write
    merged.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with shape {merged.shape}")


if __name__ == "__main__":
    main()
