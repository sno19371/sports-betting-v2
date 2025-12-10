import pandas as pd
import os

# Define the years you want to combine
years = range(2019, 2025)  # This creates a range: 2019, 2020, ..., 2024

# List to hold all the individual dataframes
all_dataframes = []

print("Starting to load files...")

for year in years:
    # Construct the filename (e.g., "depth_charts_2019.parquet")
    filename = f"depth_charts_{year}.parquet"
    
    if os.path.exists(filename):
        print(f"Loading {filename}...")
        # Read the parquet file
        df = pd.read_parquet(filename)
        all_dataframes.append(df)
    else:
        print(f"Warning: {filename} not found in the current directory.")

if all_dataframes:
    # Combine all dataframes into one
    print("Combining files...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Optional: Sort by season, week, and team for cleaner data
    combined_df = combined_df.sort_values(by=['season', 'week', 'club_code'])
    
    # Save the combined file
    output_filename = "depth_charts_combined.parquet"
    combined_df.to_parquet(output_filename, index=False)
    
    print(f"Success! Saved combined data to '{output_filename}'")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
else:
    print("No files were loaded. Please check your file names and directory.")