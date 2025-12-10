import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Replace these with your actual file paths
INPUT_CSV_FILE = 'madden_ol_ratings.csv'
OUTPUT_PARQUET_FILE = 'madden_ol_ratings.parquet'

def main():
    print(f"--- CONVERTING CSV TO PARQUET ---")
    
    # 1. Check if file exists
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"❌ Error: Input file not found at {INPUT_CSV_FILE}")
        return

    # 2. Read CSV
    # low_memory=False helps pandas guess data types correctly for large files
    print(f"Reading CSV: {INPUT_CSV_FILE}...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE, low_memory=False)
        print(f"  > Loaded {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 3. Save to Parquet
    # index=False ensures we don't save the row numbers as a separate column
    print(f"Saving to Parquet: {OUTPUT_PARQUET_FILE}...")
    try:
        df.to_parquet(OUTPUT_PARQUET_FILE, index=False)
        print("✅ Success! Conversion complete.")
    except Exception as e:
        print(f"❌ Error saving Parquet: {e}")

if __name__ == "__main__":
    main()