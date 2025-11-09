#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import PROCESSED_DATA_DIR


def main(pbp_path: str, out_path: str) -> None:
	p = Path(pbp_path)
	if not p.exists():
		raise SystemExit(f"PBP parquet not found: {p}")
	df = pd.read_parquet(p)

	required = ["season", "week", "game_id", "game_date", "posteam"]
	for k in required:
		if k not in df.columns:
			raise SystemExit(f"PBP missing required column '{k}'")

	# Receiving viewpoints: rows with a receiver id on pass plays
	dfr = df[df.get("receiver_player_id").notna()].copy()
	# Targets ≈ complete_pass + incomplete_pass for the receiver
	dfr["complete_pass"] = dfr.get("complete_pass", 0).fillna(0).astype(int)
	dfr["incomplete_pass"] = dfr.get("incomplete_pass", 0).fillna(0).astype(int)
	dfr["rec_targets"] = dfr["complete_pass"] + dfr["incomplete_pass"]
	dfr["receptions"] = dfr["complete_pass"]
	dfr["yards_gained"] = dfr.get("yards_gained", 0).fillna(0)

	group_cols = ["season", "week", "game_id", "game_date", "posteam", "receiver_player_id", "receiver_player_name"]
	for c in group_cols:
		if c not in dfr.columns:
			if c == "receiver_player_name" and "receiver" in dfr.columns:
				dfr["receiver_player_name"] = dfr["receiver"]
			else:
				raise SystemExit(f"PBP for REC build missing column '{c}'")

	agg = dfr.groupby(group_cols, as_index=False).agg(
		rec_targets=("rec_targets", "sum"),
		receptions=("receptions", "sum"),
		rec_yards=("yards_gained", "sum"),
	)

	out = agg.rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player"})
	out = out.sort_values(["season", "week", "game_id", "player_id"])

	op = Path(out_path)
	op.parent.mkdir(parents=True, exist_ok=True)
	out.to_parquet(op, index=False)
	print(f"✓ Wrote {op} with {len(out):,} rows and {out.shape[1]} columns")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--pbp", default=f"{PROCESSED_DATA_DIR}/pbp_combined.parquet")
	ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/rec_game.parquet")
	args = ap.parse_args()
	main(args.pbp, args.out)


