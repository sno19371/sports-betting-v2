#!/usr/bin/env python3
import os
import sys
# Ensure repository root is on sys.path so 'db' package is importable from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.duck import connect, table_exists, list_tables, rowcount, head, sample_columns
from config import DUCKDB_JOINED_TABLE


def main() -> None:
	con = connect(read_only=True)
	print(f"Connected to DuckDB. Looking for table/view: {DUCKDB_JOINED_TABLE}")
	print(list_tables(con).head(20))

	if not table_exists(con, DUCKDB_JOINED_TABLE):
		raise SystemExit(
			f"❌ Table/view '{DUCKDB_JOINED_TABLE}' not found. "
			f"Set DUCKDB_JOINED_TABLE in .env or create the table."
		)

	print(f"\n✓ Found {DUCKDB_JOINED_TABLE}")
	print(f"Rowcount: {rowcount(con, DUCKDB_JOINED_TABLE):,}")

	print("\nSchema (first 30 cols):")
	info = sample_columns(con, DUCKDB_JOINED_TABLE)
	print(info[['name', 'type']].head(30).to_string(index=False))

	print("\nHead:")
	print(head(con, DUCKDB_JOINED_TABLE, 5).head())


if __name__ == "__main__":
	main()


