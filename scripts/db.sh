#!/usr/bin/env bash
set -euo pipefail

# Load .env if present
if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  . ".env"
fi

DUCKDB_PATH="${DUCKDB_PATH:-databases/seamus.db}"

cmd="${1:-open}"
case "$cmd" in
  open)
    echo "Opening DuckDB at ${DUCKDB_PATH}"
    if ! command -v duckdb >/dev/null 2>&1; then
      echo "duckdb CLI not found on PATH. Install DuckDB CLI or use Python: python -c 'import duckdb,sys; con=duckdb.connect(\"${DUCKDB_PATH}\"); print(\"Connected\"); con.close()'"
      exit 1
    fi
    duckdb "${DUCKDB_PATH}"
    ;;
  check)
    if command -v duckdb >/dev/null 2>&1; then
      duckdb "${DUCKDB_PATH}" -c "SELECT 1;"
    else
      python - <<'PY'
import os, sys
try:
    import duckdb
except Exception as e:
    print("duckdb python package not available:", e)
    sys.exit(1)
path = os.environ.get("DUCKDB_PATH", "databases/seamus.db")
con = duckdb.connect(path)
print(con.execute("SELECT 1").fetchall())
con.close()
print("OK")
PY
    fi
    ;;
  create-views)
    if command -v duckdb >/dev/null 2>&1; then
      # run games views first (always available)
      if [ -f "sql/create_views_games.sql" ]; then
        echo "Applying sql/create_views_games.sql"
        duckdb "${DUCKDB_PATH}" -c "$(cat sql/create_views_games.sql)"
      else
        echo "sql/create_views_games.sql not found (skipping)."
      fi
      # run player views next (requires joined_games table to exist)
      if [ -f "sql/create_views_players.sql" ]; then
        echo "Applying sql/create_views_players.sql"
        duckdb "${DUCKDB_PATH}" -c "$(cat sql/create_views_players.sql)" || {
          echo "⚠️  Player views failed (likely missing joined_games). Create the table first, then re-run."
        }
      else
        echo "sql/create_views_players.sql not found (skipping)."
      fi
    else
      echo "duckdb CLI not found; using Python fallback."
      PY_BIN="python3"
      if [ -x "./nfl/bin/python" ]; then
        PY_BIN="./nfl/bin/python"
      fi
      "$PY_BIN" - <<'PY'
import os
import duckdb

db_path = os.environ.get("DUCKDB_PATH", "databases/seamus.db")
con = duckdb.connect(db_path)

def apply_sql(path: str) -> None:
    with open(path, "r") as f:
        sql = f.read()
    con.execute(sql)
    print(f"Applied {path}")

games_sql = "sql/create_views_games.sql"
players_sql = "sql/create_views_players.sql"

if os.path.exists(games_sql):
    try:
        print(f"Applying {games_sql}")
        apply_sql(games_sql)
    except Exception as e:
        print(f"Failed to apply {games_sql}: {e}")
else:
    print(f"{games_sql} not found (skipping).")

if os.path.exists(players_sql):
    try:
        print(f"Applying {players_sql}")
        apply_sql(players_sql)
    except Exception as e:
        print(f"⚠️  Player views failed (likely missing joined_games): {e}")
else:
    print(f"{players_sql} not found (skipping).")

con.close()
print("Done.")
PY
    fi
    ;;
  *)
    echo "usage: scripts/db.sh {open|check|create-views}"
    exit 2
    ;;
esac


