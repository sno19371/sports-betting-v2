import requests
import pandas as pd
import time
import os

# --- CONFIGURATION ---
API_KEY = "c04285f88b4f476b8b887f32ceb1321a"  # Paste your key here
OUTPUT_FILE = "NFL_Player_Props_2019_2024.csv"
BASE_URL = "https://api.sportsdata.io/v3/nfl/odds/json/PlayerPropsByWeek"

# 2019 to 2024 (inclusive)
YEARS = range(2020, 2025)

# Headers required by the API
headers = {
    "Ocp-Apim-Subscription-Key": API_KEY
}

def fetch_and_append():
    # 1. Create the file and write headers if it doesn't exist
    # We do this by checking if the file exists. If not, we set a flag to write headers later.
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    total_records = 0
    
    for year in YEARS:
        # Loop through Season Types: Regular (REG) and Playoffs (POST)
        for season_type in ["REG", "POST"]:
            
            # Define week range based on season type
            if season_type == "REG":
                season_str = str(year)
                # 18 weeks (covering modern seasons; older seasons will just be empty for week 18)
                week_range = range(1, 19) 
            else:
                season_str = f"{year}POST"
                # 4 weeks of playoffs
                week_range = range(1, 5)

            print(f"\n--- Processing {season_str} ---")

            for week in week_range:
                endpoint = f"{BASE_URL}/{season_str}/{week}"
                
                print(f"Fetching Week {week}...", end=" ")
                
                try:
                    response = requests.get(endpoint, headers=headers)
                    
                    # Handle Rate Limiting
                    if response.status_code == 429:
                        print("❌ RATE LIMIT. Sleeping 10s...")
                        time.sleep(10)
                        # Ideally, you would retry here, but we'll skip to keep logic simple
                        continue 

                    if response.status_code == 200:
                        data = response.json()
                        
                        if len(data) > 0:
                            # Convert to DataFrame
                            df = pd.DataFrame(data)
                            
                            # Add Context Columns (So you know which year/week this row belongs to)
                            df['Season'] = year
                            df['SeasonType'] = season_type
                            df['Week'] = week
                            
                            # Append to CSV
                            # mode='a' means append. header=not file_exists writes headers only once.
                            df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
                            
                            # After the first write, file definitely exists, so turn off headers
                            file_exists = True
                            
                            count = len(df)
                            total_records += count
                            print(f"✅ Appended {count} rows")
                        else:
                            print("⚠️ Empty (No data)")
                    else:
                        print(f"❌ Error {response.status_code}")

                except Exception as e:
                    print(f"❌ Error: {e}")

                # IMPORTANT: Sleep to prevent crashing the Free Trial API
                time.sleep(1.2)

    print(f"\n🎉 DONE! Total records saved: {total_records}")
    print(f"File saved as: {OUTPUT_FILE}")

    # Also export to Excel for easier analysis
    try:
        excel_file = OUTPUT_FILE.rsplit('.', 1)[0] + ".xlsx"
        # Read the accumulated CSV once and write to a single Excel sheet
        df_all = pd.read_csv(OUTPUT_FILE)
        try:
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                df_all.to_excel(writer, index=False, sheet_name="PlayerProps")
        except ImportError:
            # Fallback to xlsxwriter if openpyxl isn't installed
            with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                df_all.to_excel(writer, index=False, sheet_name="PlayerProps")
        print(f"📘 Also wrote Excel file: {excel_file}")
    except ImportError:
        print("⚠️ Excel export skipped. Install 'openpyxl' or 'xlsxwriter' to enable Excel output.")
    except Exception as e:
        print(f"⚠️ Excel export failed: {e}")

if __name__ == "__main__":
    fetch_and_append()