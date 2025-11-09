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

	# Required keys expected in pbp (from nfl_data_py)
	required = ["season", "week", "game_id", "game_date", "posteam"]
	for k in required:
		if k not in df.columns:
			raise SystemExit(f"PBP missing required column '{k}'")

	# Filter passer rows
	dfp = df[df.get("passer_player_id").notna()].copy()
	dfp["pass_attempt"] = dfp.get("pass_attempt", 0).fillna(0).astype(int)
	# Yards gained on plays (for pass plays)
	dfp["yards_gained"] = dfp.get("yards_gained", 0).fillna(0)

	group_cols = ["season", "week", "game_id", "game_date", "posteam", "passer_player_id", "passer_player_name"]
	for c in group_cols:
		if c not in dfp.columns:
			# fallback for name column variations
			if c == "passer_player_name" and "passer" in dfp.columns:
				dfp["passer_player_name"] = dfp["passer"]
			else:
				raise SystemExit(f"PBP for QB build missing column '{c}'")

	agg = dfp.groupby(group_cols, as_index=False).agg(
		qb_pass_att=("pass_attempt", "sum"),
		qb_yards=("yards_gained", "sum"),
	)

	out = agg.rename(columns={"passer_player_id": "player_id", "passer_player_name": "player"})
	out = out.sort_values(["season", "week", "game_id", "player_id"])

	op = Path(out_path)
	op.parent.mkdir(parents=True, exist_ok=True)
	out.to_parquet(op, index=False)
	print(f"✓ Wrote {op} with {len(out):,} rows and {out.shape[1]} columns")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--pbp", default=f"{PROCESSED_DATA_DIR}/pbp_combined.parquet")
	ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/qb_game.parquet")
	args = ap.parse_args()
	main(args.pbp, args.out)


