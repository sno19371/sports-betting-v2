# build_rosters_dedup.py
# Merges yearly roster parquet files and de-dups to one row per (season, gsis_id)
# Usage:
#   python build_rosters_dedup.py
# Optional args:
#   python build_rosters_dedup.py --in ./rosters_years --out rosters_2019_2023.parquet --dedup rosters_2019_2023_dedup.parquet

import argparse
import glob
import os
from pathlib import Path

import duckdb


def main(input_dir: Path, out_all: Path, out_dedup: Path):
    # Gather input files
    paths = sorted(glob.glob(str(input_dir / "roster_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")

    print(f"Found {len(paths)} files:")
    for p in paths:
        print("  -", p)

    con = duckdb.connect()

    # 1) Merge all rosters into a single parquet
    print(f"\nMerging → {out_all}")
    con.execute(f"""
      COPY (
        SELECT * FROM read_parquet({paths})
      ) TO '{out_all.as_posix()}' (FORMAT PARQUET);
    """)
    print("Wrote:", out_all)

    # 2) De-dup to one row per (season, gsis_id)
    # You can adjust the ORDER BY to prefer certain rows (e.g., active roster over PS),
    # or stable columns like team, position, or latest 'full_name'. Current order is arbitrary but deterministic.
    print(f"\nDe-duplicating → one row per (season, gsis_id) → {out_dedup}")
    con.execute(f"""
      CREATE OR REPLACE VIEW roster_all AS
      SELECT * FROM read_parquet('{out_all.as_posix()}');

      COPY (
        WITH ranked AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY season, gsis_id
              ORDER BY
                -- tweak preference order here if desired:
                -- prefer non-null team, then position, then name, then gsis_id as a stable tie-breaker
                (team IS NULL) ASC,
                (position IS NULL) ASC,
                team,
                position,
                full_name,
                gsis_id
            ) AS rn
          FROM roster_all
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
      ) TO '{out_dedup.as_posix()}' (FORMAT PARQUET);
    """)
    print("Wrote:", out_dedup)

    # Quick sanity prints
    rows_all = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_all.as_posix()}')").fetchone()[0]
    rows_dedup = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_dedup.as_posix()}')").fetchone()[0]
    print(f"\nCounts → merged: {rows_all:,} | dedup: {rows_dedup:,}")

    # Optional: show a few distinct seasons and sample rows
    print("\nSample seasons in dedup file:")
    print(con.execute(f"""
      SELECT season, COUNT(*) AS n
      FROM read_parquet('{out_dedup.as_posix()}')
      GROUP BY 1 ORDER BY 1
    """).df())

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_dir", default="./rosters_years", help="Folder with roster_*.parquet")
    parser.add_argument("--out", dest="out_all", default="rosters_2019_2023.parquet", help="Merged output parquet")
    parser.add_argument("--dedup", dest="out_dedup", default="rosters_2019_2023_dedup.parquet", help="Deduped output parquet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_all = Path(args.out_all).resolve()
    out_dedup = Path(args.out_dedup).resolve()
    out_all.parent.mkdir(parents=True, exist_ok=True)
    out_dedup.parent.mkdir(parents=True, exist_ok=True)

    main(input_dir, out_all, out_dedup)
