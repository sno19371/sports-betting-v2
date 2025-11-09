from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd

from config import DUCKDB_PATH, DUCKDB_JOINED_TABLE


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
	DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
	return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
	q = """
    SELECT CAST(COUNT(*) AS INTEGER)
    FROM information_schema.tables
    WHERE lower(table_name) = lower(?)
    """
	return con.execute(q, [table]).fetchone()[0] > 0


def list_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
	return con.execute(
		"""
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        ORDER BY 1,2
        """
	).fetchdf()


def rowcount(con: duckdb.DuckDBPyConnection, table: Optional[str] = None) -> int:
	t = table or DUCKDB_JOINED_TABLE
	return con.execute(f"SELECT CAST(COUNT(*) AS BIGINT) FROM {t}").fetchone()[0]


def head(con: duckdb.DuckDBPyConnection, table: Optional[str] = None, n: int = 5) -> pd.DataFrame:
	t = table or DUCKDB_JOINED_TABLE
	return con.execute(f"SELECT * FROM {t} LIMIT {int(n)}").fetchdf()


def sample_columns(con: duckdb.DuckDBPyConnection, table: Optional[str] = None, like: Optional[str] = None) -> pd.DataFrame:
	t = table or DUCKDB_JOINED_TABLE
	df = con.execute(f"PRAGMA table_info({t})").fetchdf()
	if like:
		# Use pandas engine for contains to avoid warning
		return df.query("name.str.contains(@like)", engine="python")
	return df


def query(con: duckdb.DuckDBPyConnection, sql: str, params: Optional[Sequence] = None) -> pd.DataFrame:
	return con.execute(sql, params or []).fetchdf()


