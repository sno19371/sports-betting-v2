# test_setup.py
import os
from dotenv import load_dotenv
import requests
import nfl_data_py as nfl

# Load environment variables
load_dotenv()

# API key constants (fallback if env vars are not set)
OPENWEATHER_API_KEY = "a4d68442a7189737322d02b91534ff31"
ODDS_API_KEY = "0ab1bb2fa85b362c9fc45b18d477bd33"

def test_everything():
    print("🧪 Testing NFL Props Setup...\n")
    
    # Test 1: NFL Data
    try:
        print("📊 Testing NFL data...")
        weekly_data = nfl.import_weekly_data(years=[2024], columns=['player_name', 'receiving_yards'])
        print(f"✅ NFL Data: {len(weekly_data):,} records loaded")
    except Exception as e:
        print(f"❌ NFL Data failed: {e}")
    
    # Test 2: Weather API
    weather_key = os.getenv('OPENWEATHER_API_KEY') or OPENWEATHER_API_KEY
    if weather_key:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q=Kansas City&appid={weather_key}&units=imperial"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                print(f"✅ Weather API: {temp}°F in Kansas City")
            else:
                print(f"❌ Weather API: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Weather API failed: {e}")
    else:
        print("❌ No weather API key found")
    
    # Test 3: Odds API
    odds_key = os.getenv('ODDS_API_KEY') or ODDS_API_KEY
    if odds_key:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={odds_key}&regions=us&markets=h2h"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                remaining = response.headers.get('x-requests-remaining', 'Unknown')
                print(f"✅ Odds API: {len(data)} games found")
                print(f"📊 Requests remaining: {remaining}")
            else:
                print(f"❌ Odds API: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Odds API failed: {e}")
    else:
        print("❌ No odds API key found")
    
    # Test 4: ESPN API (no key needed)
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ ESPN API: Connected successfully")
        else:
            print(f"❌ ESPN API: Status {response.status_code}")
    except Exception as e:
        print(f"❌ ESPN API failed: {e}")
    
    print("\n🎉 Setup test complete!")

if __name__ == "__main__":
    test_everything()


