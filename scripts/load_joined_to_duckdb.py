#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from config import DUCKDB_PATH, DUCKDB_JOINED_TABLE, PROCESSED_DATA_DIR


def main(parquet_path: str, table: str | None) -> None:
	table = table or DUCKDB_JOINED_TABLE
	path = Path(parquet_path)
	if not path.exists():
		raise SystemExit(f"Parquet not found: {path}")

	DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
	con = duckdb.connect(str(DUCKDB_PATH), read_only=False)

	# Create or replace table from parquet
	con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?);", [str(path)])
	n = con.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
	print(f"✓ Created table {table} from {path.name} with {n:,} rows")

	# Optional indexes (DuckDB adaptive indexing helps; these are hints)
	for col in ["season", "week", "game_id", "player_id", "posteam", "game_date"]:
		try:
			con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col});")
		except Exception:
			pass

	con.close()
	print("✓ Done.")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--parquet", default=f"{PROCESSED_DATA_DIR}/joined_games.parquet")
	ap.add_argument("--table", default=None)
	args = ap.parse_args()
	main(args.parquet, args.table)


