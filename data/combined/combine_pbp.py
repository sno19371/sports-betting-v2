import duckdb, glob, os

DATA_DIR = "./pbp_years"
paths = sorted(glob.glob(os.path.join(DATA_DIR, "play_by_play_*.parquet")))
assert paths, "No parquet files found"

con = duckdb.connect()
con.execute(f"""
  COPY (
    SELECT * FROM read_parquet({paths})
  ) TO 'pbp_2019_2023.parquet' (FORMAT PARQUET);
""")
con.close()
print("Wrote pbp_2019_2023.parquet")