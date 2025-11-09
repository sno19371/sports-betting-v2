#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from config import PROCESSED_DATA_DIR

def main(pbp_path: str, out_path: str) -> None:
	p = Path(pbp_path)
	if not p.exists():
		raise SystemExit(f"PBP parquet not found: {p}")
	df = pd.read_parquet(p)

	# Required
	for k in ["season","week","game_id","game_date","posteam"]:
		if k not in df.columns:
			raise SystemExit(f"PBP missing required column '{k}'")

	# Canonical rush filter (exclude kneels/aborted/no-plays)
	rush_attempt = df.get("rush_attempt", 0).fillna(0).astype(int)
	qb_kneel = df.get("qb_kneel", 0).fillna(0).astype(int)
	aborted = df.get("aborted_play", 0).fillna(0).astype(int)
	is_play = df.get("play", 1).fillna(1).astype(int)

	is_rush = ((rush_attempt == 1) & (qb_kneel == 0) & (aborted == 0) & (is_play == 1)).astype(int)
	df["is_rush"] = is_rush
	df["yards_gained"] = df.get("yards_gained", 0).fillna(0)

	# Filter rows with a rusher id
	dfr = df[df.get("rusher_player_id").notna()].copy()

	# Group & aggregate
	group_cols = ["season","week","game_id","game_date","posteam","rusher_player_id","rusher_player_name"]
	if "rusher_player_name" not in dfr.columns and "rusher" in dfr.columns:
		dfr["rusher_player_name"] = dfr["rusher"]

	agg = dfr.groupby(group_cols, as_index=False).agg(
		rb_carries=("is_rush","sum"),
		rb_yards=("yards_gained", lambda s: float((s * dfr.loc[s.index, "is_rush"]).sum())),
	)

	out = agg.rename(columns={"rusher_player_id":"player_id", "rusher_player_name":"player"})
	out = out.sort_values(["season","week","game_id","player_id"])

	op = Path(out_path)
	op.parent.mkdir(parents=True, exist_ok=True)
	out.to_parquet(op, index=False)
	print(f"✓ Wrote {op} with {len(out):,} rows and {out.shape[1]} columns")

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--pbp", default=f"{PROCESSED_DATA_DIR}/pbp_combined.parquet")
	ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/rb_game.parquet")
	args = ap.parse_args()
	main(args.pbp, args.out)


