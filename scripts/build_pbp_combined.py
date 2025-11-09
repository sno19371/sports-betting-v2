#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import HISTORICAL_YEARS, PROCESSED_DATA_DIR

LOCAL_PATTERNS = [
	str(Path(PROCESSED_DATA_DIR) / "pbp_data_{y}.parquet"),
	str(Path(PROCESSED_DATA_DIR) / "pbp_{y}.parquet"),
	str(Path(PROCESSED_DATA_DIR) / "pbp{y}.parquet"),
]


def try_load_local(year: int) -> pd.DataFrame | None:
	for pat in LOCAL_PATTERNS:
		p = Path(pat.format(y=year))
		if p.exists():
			print(f"• Using local PBP: {p}")
			return pd.read_parquet(p)
	return None


def try_fetch(year: int) -> pd.DataFrame | None:
	try:
		import nfl_data_py as nfl
		if hasattr(nfl, "import_pbp_data"):
			print(f"• Fetching via nfl_data_py.import_pbp_data({year})")
			return nfl.import_pbp_data([year])
		elif hasattr(nfl, "import_pbp"):
			print(f"• Fetching via nfl_data_py.import_pbp({year})")
			return nfl.import_pbp([year])
	except Exception as e:
		print(f"⚠️ Could not fetch year {year}: {e}")
	return None


def main(out_path: str, years_csv: str | None) -> None:
	years = [int(x) for x in years_csv.split(",")] if years_csv else HISTORICAL_YEARS
	parts: list[pd.DataFrame] = []
	for y in years:
		df = try_load_local(y)
		if df is None:
			df = try_fetch(y)
		if df is None:
			print(f"⚠️ Skipping {y} (no local file & fetch failed)")
			continue
		parts.append(df)

	if not parts:
		raise SystemExit("No PBP found. Put yearly PBP parquet(s) under data/processed or enable nfl_data_py fetch.")

	df = pd.concat(parts, ignore_index=True)
	op = Path(out_path)
	op.parent.mkdir(parents=True, exist_ok=True)
	df.to_parquet(op, index=False)
	print(f"✓ Wrote {op} with {len(df):,} rows and {df.shape[1]} columns")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/pbp_combined.parquet")
	ap.add_argument("--years", help="Comma-separated years (e.g., 2023,2024). Default = HISTORICAL_YEARS")
	args = ap.parse_args()
	main(args.out, args.years)


