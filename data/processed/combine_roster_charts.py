import pandas as pd
import os

# 1. Define the years and file naming convention
start_year = 2019
end_year = 2024
years = range(start_year, end_year + 1)

# 2. Define the exact columns you want to keep
desired_columns = [
    "season", "team", "position", "depth_chart_position", "jersey_number", 
    "status", "full_name", "first_name", "last_name", "birth_date", 
    "height", "weight", "college", "gsis_id", "espn_id", 
    "sportradar_id", "yahoo_id", "rotowire_id", "pff_id", "pfr_id", 
    "fantasy_data_id", "sleeper_id", "years_exp", "headshot_url", 
    "ngs_position", "week", "game_type", "status_description_abbr", 
    "football_name", "esb_id", "gsis_it_id", "smart_id", "entry_year", 
    "rookie_year", "draft_club", "draft_number"
]

dataframes = []

print("Starting file processing...")

# 3. Iterate through the years
for year in years:
    filename = f"roster_weekly_{year}.parquet"
    
    if os.path.exists(filename):
        print(f"Reading {filename}...")
        try:
            # Read the parquet file
            df = pd.read_parquet(filename)
            
            # 4. Filter for only the requested columns
            # We use intersection to avoid errors if a specific year is missing a column
            # (Optional: remove 'intersect' logic if you want it to fail on missing cols)
            available_cols = [c for c in desired_columns if c in df.columns]
            
            # Warn if columns are missing
            missing_cols = set(desired_columns) - set(available_cols)
            if missing_cols:
                print(f"  Warning: {filename} is missing columns: {missing_cols}")
            
            # Select columns and add to list
            df_subset = df[available_cols]
            dataframes.append(df_subset)
            
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
    else:
        print(f"  Warning: {filename} not found. Skipping.")

# 5. Combine all dataframes
if dataframes:
    print("Concatenating files...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # 6. Save the result
    output_filename = "roster_weekly_combined.parquet"
    combined_df.to_parquet(output_filename, index=False)
    print(f"Success! Combined data saved to '{output_filename}'")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns included: {list(combined_df.columns)}")
else:
    print("No data was combined. Please check your files.")