#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR

KEYS: List[str] = ["season", "week", "game_id", "game_date", "posteam", "player_id", "player"]


def _standardize_ids(df: pd.DataFrame, id_candidates: List[str], name_candidates: List[str]) -> pd.DataFrame:
	"""Rename player id/name columns to canonical names."""
	# If already standardized, return copy
	if "player_id" in df.columns and "player" in df.columns:
		return df.copy()
	pid = next((c for c in id_candidates if c in df.columns), None)
	pname = next((c for c in name_candidates if c in df.columns), None)
	if pid is None or pname is None:
		raise ValueError(f"Could not find id/name columns in {id_candidates} / {name_candidates}")
	out = df.copy()
	out = out.rename(columns={pid: "player_id", pname: "player"})
	return out


def _prefix_nonkeys(df: pd.DataFrame, prefix: str, keys: List[str]) -> pd.DataFrame:
	"""Prefix all columns not in KEYS with a given prefix to avoid collisions."""
	df = df.copy()
	rename_map = {c: f"{prefix}{c}" for c in df.columns if c not in keys and not c.startswith(prefix)}
	return df.rename(columns=rename_map)


def _select_keys(df: pd.DataFrame) -> pd.DataFrame:
	missing = [k for k in KEYS if k not in df.columns]
	if missing:
		raise ValueError(f"Missing required keys: {missing}")
	return df[KEYS].drop_duplicates()


def _has_block(df: pd.DataFrame, prefix: str) -> np.ndarray:
	cols = [c for c in df.columns if c.startswith(prefix)]
	if not cols:
		return np.zeros(len(df), dtype=int)
	m = df[cols]
	return (m.notna().any(axis=1)).astype(int)


def _require_files(paths: List[Path]) -> Tuple[bool, List[Path]]:
	missing = [p for p in paths if not p.exists()]
	return (len(missing) == 0, missing)


def main(qb_parquet: str, rb_parquet: str, rec_parquet: str, out_parquet: str) -> None:
	qb_path = Path(qb_parquet)
	rb_path = Path(rb_parquet)
	rec_path = Path(rec_parquet)

	ok, missing = _require_files([qb_path, rb_path, rec_path])
	if not ok:
		raise SystemExit(
			"Missing required parquet files:\n  - " + "\n  - ".join(str(p) for p in missing) +
			"\nBuild these position files first, then re-run this script."
		)

	qb = pd.read_parquet(qb_path)
	rb = pd.read_parquet(rb_path)
	rec = pd.read_parquet(rec_path)

	# Standardize id/name per position builders
	qb = _standardize_ids(qb, ["passer_id", "passer_player_id"], ["passer", "passer_player_name"])
	rb = _standardize_ids(rb, ["rusher_id", "rusher_player_id"], ["rusher", "rusher_player_name"])
	rec = _standardize_ids(rec, ["receiver_id", "receiver_player_id"], ["receiver", "receiver_player_name"])

	# Ensure required keys exist
	for tag, df in [("QB", qb), ("RB", rb), ("REC", rec)]:
		for k in KEYS:
			if k not in df.columns:
				raise SystemExit(f"{tag} parquet missing key column '{k}'")

	# Prefix non-key metrics to avoid name collisions
	qb_pref = _prefix_nonkeys(qb, "qb_", KEYS)
	rb_pref = _prefix_nonkeys(rb, "rb_", KEYS)
	rec_pref = _prefix_nonkeys(rec, "rec_", KEYS)

	# Keys universe
	keys_union = (
		pd.concat([_select_keys(qb_pref), _select_keys(rb_pref), _select_keys(rec_pref)], ignore_index=True)
		.drop_duplicates()
	)

	# Outer-join on keys
	out = (
		keys_union.merge(qb_pref, on=KEYS, how="left")
		          .merge(rb_pref, on=KEYS, how="left")
		          .merge(rec_pref, on=KEYS, how="left")
	)

	# Role flags
	out["has_qb"] = _has_block(out, "qb_")
	out["has_rb"] = _has_block(out, "rb_")
	out["has_rec"] = _has_block(out, "rec_")

	# Persist parquet
	out_path = Path(out_parquet)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out.to_parquet(out_path, index=False)
	print(f"✓ Wrote {out_path} with {len(out):,} rows and {out.shape[1]} columns")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--qb", default=f"{PROCESSED_DATA_DIR}/qb_game.parquet")
	ap.add_argument("--rb", default=f"{PROCESSED_DATA_DIR}/rb_game.parquet")
	ap.add_argument("--rec", default=f"{PROCESSED_DATA_DIR}/rec_game.parquet")
	ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/joined_games.parquet")
	args = ap.parse_args()
	main(args.qb, args.rb, args.rec, args.out)


