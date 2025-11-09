# complete_stadium_data.py
"""
Complete NFL Stadium Reference Data (2024-2025 Season)
Run this to create your stadiums.parquet file with all 32 teams
"""

import pandas as pd
import duckdb

# All 32 NFL stadiums with accurate data as of 2024-2025 season
stadiums_data = [
    # AFC East
    {
        'stadium_id': 1,
        'stadium_name': 'Highmark Stadium',
        'team': 'BUF',
        'city': 'Orchard Park',
        'state': 'NY',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 42.7738,
        'longitude': -78.7870,
        'elevation_feet': 644,
        'capacity': 71608,
        'effective_from': '2021-07-01',  # Renamed from New Era Field
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 2,
        'stadium_name': 'Hard Rock Stadium',
        'team': 'MIA',
        'city': 'Miami Gardens',
        'state': 'FL',
        'roof_type': 'outdoors',  # Technically has a canopy but plays outdoors
        'surface': 'grass',
        'latitude': 25.9580,
        'longitude': -80.2389,
        'elevation_feet': 10,
        'capacity': 65326,
        'effective_from': '2010-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 3,
        'stadium_name': 'Gillette Stadium',
        'team': 'NE',
        'city': 'Foxborough',
        'state': 'MA',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 42.0909,
        'longitude': -71.2643,
        'elevation_feet': 151,
        'capacity': 65878,
        'effective_from': '2002-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 4,
        'stadium_name': 'MetLife Stadium',
        'team': 'NYJ',
        'city': 'East Rutherford',
        'state': 'NJ',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 40.8135,
        'longitude': -74.0745,
        'elevation_feet': 10,
        'capacity': 82500,
        'effective_from': '2010-01-01',
        'effective_to': None,
        'is_current': True
    },
    
    # AFC North
    {
        'stadium_id': 5,
        'stadium_name': 'M&T Bank Stadium',
        'team': 'BAL',
        'city': 'Baltimore',
        'state': 'MD',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 39.2780,
        'longitude': -76.6227,
        'elevation_feet': 50,
        'capacity': 70745,
        'effective_from': '1998-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 6,
        'stadium_name': 'Paycor Stadium',
        'team': 'CIN',
        'city': 'Cincinnati',
        'state': 'OH',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 39.0954,
        'longitude': -84.5160,
        'elevation_feet': 482,
        'capacity': 65515,
        'effective_from': '2022-01-01',  # Renamed from Paul Brown Stadium
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 7,
        'stadium_name': 'Cleveland Browns Stadium',
        'team': 'CLE',
        'city': 'Cleveland',
        'state': 'OH',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 41.5061,
        'longitude': -81.6995,
        'elevation_feet': 584,
        'capacity': 67431,
        'effective_from': '2023-01-01',  # Renamed from FirstEnergy Stadium
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 8,
        'stadium_name': 'Acrisure Stadium',
        'team': 'PIT',
        'city': 'Pittsburgh',
        'state': 'PA',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 40.4468,
        'longitude': -80.0158,
        'elevation_feet': 738,
        'capacity': 68400,
        'effective_from': '2022-07-01',  # Renamed from Heinz Field
        'effective_to': None,
        'is_current': True
    },
    
    # AFC South
    {
        'stadium_id': 9,
        'stadium_name': 'NRG Stadium',
        'team': 'HOU',
        'city': 'Houston',
        'state': 'TX',
        'roof_type': 'retractable',
        'surface': 'fieldturf',
        'latitude': 29.6847,
        'longitude': -95.4107,
        'elevation_feet': 55,
        'capacity': 72220,
        'effective_from': '2002-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 10,
        'stadium_name': 'Lucas Oil Stadium',
        'team': 'IND',
        'city': 'Indianapolis',
        'state': 'IN',
        'roof_type': 'retractable',
        'surface': 'fieldturf',
        'latitude': 39.7601,
        'longitude': -86.1639,
        'elevation_feet': 715,
        'capacity': 67000,
        'effective_from': '2008-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 11,
        'stadium_name': 'EverBank Stadium',
        'team': 'JAX',
        'city': 'Jacksonville',
        'state': 'FL',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 30.3239,
        'longitude': -81.6373,
        'elevation_feet': 16,
        'capacity': 67814,
        'effective_from': '2023-01-01',  # Renamed from TIAA Bank Field
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 12,
        'stadium_name': 'Nissan Stadium',
        'team': 'TEN',
        'city': 'Nashville',
        'state': 'TN',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 36.1665,
        'longitude': -86.7713,
        'elevation_feet': 440,
        'capacity': 69143,
        'effective_from': '1999-01-01',
        'effective_to': None,
        'is_current': True
    },
    
    # AFC West
    {
        'stadium_id': 13,
        'stadium_name': 'Empower Field at Mile High',
        'team': 'DEN',
        'city': 'Denver',
        'state': 'CO',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 39.7439,
        'longitude': -105.0201,
        'elevation_feet': 5280,  # Mile high!
        'capacity': 76125,
        'effective_from': '2019-01-01',  # Renamed from Sports Authority Field
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 14,
        'stadium_name': 'Arrowhead Stadium',
        'team': 'KC',
        'city': 'Kansas City',
        'state': 'MO',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 39.0489,
        'longitude': -94.4839,
        'elevation_feet': 750,
        'capacity': 76416,
        'effective_from': '1972-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 15,
        'stadium_name': 'Allegiant Stadium',
        'team': 'LV',
        'city': 'Las Vegas',
        'state': 'NV',
        'roof_type': 'dome',
        'surface': 'fieldturf',
        'latitude': 36.0909,
        'longitude': -115.1833,
        'elevation_feet': 2000,
        'capacity': 65000,
        'effective_from': '2020-07-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 16,
        'stadium_name': 'SoFi Stadium',
        'team': 'LAC',
        'city': 'Inglewood',
        'state': 'CA',
        'roof_type': 'outdoors',  # Has a canopy but is technically open-air
        'surface': 'fieldturf',
        'latitude': 33.9535,
        'longitude': -118.3392,
        'elevation_feet': 100,
        'capacity': 70240,
        'effective_from': '2020-09-01',
        'effective_to': None,
        'is_current': True
    },
    
    # NFC East
    {
        'stadium_id': 17,
        'stadium_name': 'AT&T Stadium',
        'team': 'DAL',
        'city': 'Arlington',
        'state': 'TX',
        'roof_type': 'retractable',
        'surface': 'fieldturf',
        'latitude': 32.7473,
        'longitude': -97.0945,
        'elevation_feet': 550,
        'capacity': 80000,
        'effective_from': '2009-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 18,
        'stadium_name': 'MetLife Stadium',  # Giants share with Jets
        'team': 'NYG',
        'city': 'East Rutherford',
        'state': 'NJ',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 40.8135,
        'longitude': -74.0745,
        'elevation_feet': 10,
        'capacity': 82500,
        'effective_from': '2010-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 19,
        'stadium_name': 'Lincoln Financial Field',
        'team': 'PHI',
        'city': 'Philadelphia',
        'state': 'PA',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 39.9008,
        'longitude': -75.1675,
        'elevation_feet': 23,
        'capacity': 69796,
        'effective_from': '2003-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 20,
        'stadium_name': 'Northwest Stadium',
        'team': 'WAS',
        'city': 'Landover',
        'state': 'MD',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 38.9076,
        'longitude': -76.8645,
        'elevation_feet': 200,
        'capacity': 62000,
        'effective_from': '2024-07-01',  # Renamed from FedEx Field
        'effective_to': None,
        'is_current': True
    },
    
    # NFC North
    {
        'stadium_id': 21,
        'stadium_name': 'Soldier Field',
        'team': 'CHI',
        'city': 'Chicago',
        'state': 'IL',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 41.8623,
        'longitude': -87.6167,
        'elevation_feet': 594,
        'capacity': 61500,
        'effective_from': '2003-01-01',  # Major renovation
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 22,
        'stadium_name': 'Ford Field',
        'team': 'DET',
        'city': 'Detroit',
        'state': 'MI',
        'roof_type': 'dome',
        'surface': 'fieldturf',
        'latitude': 42.3400,
        'longitude': -83.0456,
        'elevation_feet': 585,
        'capacity': 65000,
        'effective_from': '2002-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 23,
        'stadium_name': 'Lambeau Field',
        'team': 'GB',
        'city': 'Green Bay',
        'state': 'WI',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 44.5013,
        'longitude': -88.0622,
        'elevation_feet': 640,
        'capacity': 81441,
        'effective_from': '1957-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 24,
        'stadium_name': 'U.S. Bank Stadium',
        'team': 'MIN',
        'city': 'Minneapolis',
        'state': 'MN',
        'roof_type': 'dome',
        'surface': 'fieldturf',
        'latitude': 44.9738,
        'longitude': -93.2577,
        'elevation_feet': 830,
        'capacity': 66860,
        'effective_from': '2016-07-01',
        'effective_to': None,
        'is_current': True
    },
    
    # NFC South
    {
        'stadium_id': 25,
        'stadium_name': 'Mercedes-Benz Stadium',
        'team': 'ATL',
        'city': 'Atlanta',
        'state': 'GA',
        'roof_type': 'retractable',
        'surface': 'fieldturf',
        'latitude': 33.7553,
        'longitude': -84.4006,
        'elevation_feet': 1050,
        'capacity': 71000,
        'effective_from': '2017-08-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 26,
        'stadium_name': 'Bank of America Stadium',
        'team': 'CAR',
        'city': 'Charlotte',
        'state': 'NC',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 35.2258,
        'longitude': -80.8528,
        'elevation_feet': 700,
        'capacity': 75523,
        'effective_from': '1996-01-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 27,
        'stadium_name': 'Caesars Superdome',
        'team': 'NO',
        'city': 'New Orleans',
        'state': 'LA',
        'roof_type': 'dome',
        'surface': 'fieldturf',
        'latitude': 29.9511,
        'longitude': -90.0812,
        'elevation_feet': 10,
        'capacity': 73208,
        'effective_from': '2021-07-01',  # Renamed from Mercedes-Benz Superdome
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 28,
        'stadium_name': 'Raymond James Stadium',
        'team': 'TB',
        'city': 'Tampa',
        'state': 'FL',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 27.9759,
        'longitude': -82.5033,
        'elevation_feet': 15,
        'capacity': 65890,
        'effective_from': '1998-01-01',
        'effective_to': None,
        'is_current': True
    },
    
    # NFC West
    {
        'stadium_id': 29,
        'stadium_name': 'State Farm Stadium',
        'team': 'ARI',
        'city': 'Glendale',
        'state': 'AZ',
        'roof_type': 'retractable',
        'surface': 'grass',
        'latitude': 33.5276,
        'longitude': -112.2626,
        'elevation_feet': 1150,
        'capacity': 63400,
        'effective_from': '2018-01-01',  # Renamed from University of Phoenix Stadium
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 30,
        'stadium_name': 'SoFi Stadium',  # Rams share with Chargers
        'team': 'LAR',
        'city': 'Inglewood',
        'state': 'CA',
        'roof_type': 'outdoors',
        'surface': 'fieldturf',
        'latitude': 33.9535,
        'longitude': -118.3392,
        'elevation_feet': 100,
        'capacity': 70240,
        'effective_from': '2020-09-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 31,
        'stadium_name': "Levi's Stadium",
        'team': 'SF',
        'city': 'Santa Clara',
        'state': 'CA',
        'roof_type': 'outdoors',
        'surface': 'grass',
        'latitude': 37.4032,
        'longitude': -121.9698,
        'elevation_feet': 45,
        'capacity': 68500,
        'effective_from': '2014-07-01',
        'effective_to': None,
        'is_current': True
    },
    {
        'stadium_id': 32,
        'stadium_name': 'Lumen Field',
        'team': 'SEA',
        'city': 'Seattle',
        'state': 'WA',
        'roof_type': 'outdoors',  # Has a partial roof but field is open
        'surface': 'fieldturf',
        'latitude': 47.5952,
        'longitude': -122.3316,
        'elevation_feet': 10,
        'capacity': 68740,
        'effective_from': '2020-01-01',  # Renamed from CenturyLink Field
        'effective_to': None,
        'is_current': True
    }
]

# City coordinates for weather API
cities_data = [
    {'city': 'Orchard Park', 'state': 'NY', 'lat': 42.7642, 'lon': -78.7992, 'timezone': 'America/New_York'},
    {'city': 'Miami Gardens', 'state': 'FL', 'lat': 25.9420, 'lon': -80.2456, 'timezone': 'America/New_York'},
    {'city': 'Foxborough', 'state': 'MA', 'lat': 42.0654, 'lon': -71.2489, 'timezone': 'America/New_York'},
    {'city': 'East Rutherford', 'state': 'NJ', 'lat': 40.8139, 'lon': -74.0778, 'timezone': 'America/New_York'},
    {'city': 'Baltimore', 'state': 'MD', 'lat': 39.2904, 'lon': -76.6122, 'timezone': 'America/New_York'},
    {'city': 'Cincinnati', 'state': 'OH', 'lat': 39.1031, 'lon': -84.5120, 'timezone': 'America/New_York'},
    {'city': 'Cleveland', 'state': 'OH', 'lat': 41.4993, 'lon': -81.6944, 'timezone': 'America/New_York'},
    {'city': 'Pittsburgh', 'state': 'PA', 'lat': 40.4406, 'lon': -79.9959, 'timezone': 'America/New_York'},
    {'city': 'Houston', 'state': 'TX', 'lat': 29.7604, 'lon': -95.3698, 'timezone': 'America/Chicago'},
    {'city': 'Indianapolis', 'state': 'IN', 'lat': 39.7684, 'lon': -86.1581, 'timezone': 'America/New_York'},
    {'city': 'Jacksonville', 'state': 'FL', 'lat': 30.3322, 'lon': -81.6557, 'timezone': 'America/New_York'},
    {'city': 'Nashville', 'state': 'TN', 'lat': 36.1627, 'lon': -86.7816, 'timezone': 'America/Chicago'},
    {'city': 'Denver', 'state': 'CO', 'lat': 39.7392, 'lon': -104.9903, 'timezone': 'America/Denver'},
    {'city': 'Kansas City', 'state': 'MO', 'lat': 39.0997, 'lon': -94.5786, 'timezone': 'America/Chicago'},
    {'city': 'Las Vegas', 'state': 'NV', 'lat': 36.1699, 'lon': -115.1398, 'timezone': 'America/Los_Angeles'},
    {'city': 'Inglewood', 'state': 'CA', 'lat': 33.9617, 'lon': -118.3531, 'timezone': 'America/Los_Angeles'},
    {'city': 'Arlington', 'state': 'TX', 'lat': 32.7357, 'lon': -97.1081, 'timezone': 'America/Chicago'},
    {'city': 'Philadelphia', 'state': 'PA', 'lat': 39.9526, 'lon': -75.1652, 'timezone': 'America/New_York'},
    {'city': 'Landover', 'state': 'MD', 'lat': 38.9340, 'lon': -76.8653, 'timezone': 'America/New_York'},
    {'city': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'lon': -87.6298, 'timezone': 'America/Chicago'},
    {'city': 'Detroit', 'state': 'MI', 'lat': 42.3314, 'lon': -83.0458, 'timezone': 'America/New_York'},
    {'city': 'Green Bay', 'state': 'WI', 'lat': 44.5133, 'lon': -88.0133, 'timezone': 'America/Chicago'},
    {'city': 'Minneapolis', 'state': 'MN', 'lat': 44.9778, 'lon': -93.2650, 'timezone': 'America/Chicago'},
    {'city': 'Atlanta', 'state': 'GA', 'lat': 33.7490, 'lon': -84.3880, 'timezone': 'America/New_York'},
    {'city': 'Charlotte', 'state': 'NC', 'lat': 35.2271, 'lon': -80.8431, 'timezone': 'America/New_York'},
    {'city': 'New Orleans', 'state': 'LA', 'lat': 29.9511, 'lon': -90.0715, 'timezone': 'America/Chicago'},
    {'city': 'Tampa', 'state': 'FL', 'lat': 27.9506, 'lon': -82.4572, 'timezone': 'America/New_York'},
    {'city': 'Glendale', 'state': 'AZ', 'lat': 33.5387, 'lon': -112.1859, 'timezone': 'America/Phoenix'},
    {'city': 'Santa Clara', 'state': 'CA', 'lat': 37.3541, 'lon': -121.9552, 'timezone': 'America/Los_Angeles'},
    {'city': 'Seattle', 'state': 'WA', 'lat': 47.6062, 'lon': -122.3321, 'timezone': 'America/Los_Angeles'}
]

if __name__ == "__main__":
    # Create DuckDB connection
    conn = duckdb.connect('databases/seamus.db')
    
    # Convert to DataFrames
    stadiums_df = pd.DataFrame(stadiums_data)
    cities_df = pd.DataFrame(cities_data)
    
    # Create tables in DuckDB
    conn.execute("DROP TABLE IF EXISTS stadiums")
    conn.execute("DROP TABLE IF EXISTS cities")
    
    conn.execute("""
        CREATE TABLE stadiums (
            stadium_id INTEGER,
            stadium_name VARCHAR,
            team VARCHAR,
            city VARCHAR,
            state VARCHAR,
            roof_type VARCHAR,
            surface VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            elevation_feet INTEGER,
            capacity INTEGER,
            effective_from DATE,
            effective_to DATE,
            is_current BOOLEAN,
            PRIMARY KEY (stadium_id, effective_from)
        )
    """)
    
    conn.execute("""
        CREATE TABLE cities (
            city VARCHAR,
            state VARCHAR,
            lat DOUBLE,
            lon DOUBLE,
            timezone VARCHAR,
            PRIMARY KEY (city, state)
        )
    """)
    
    # Insert data
    conn.register('stadiums_temp', stadiums_df)
    conn.register('cities_temp', cities_df)
    
    conn.execute("INSERT INTO stadiums SELECT * FROM stadiums_temp")
    conn.execute("INSERT INTO cities SELECT * FROM cities_temp")
    
    # Export to Parquet
    conn.execute("COPY stadiums TO 'data/static/stadiums.parquet' (FORMAT PARQUET)")
    conn.execute("COPY cities TO 'data/static/cities.parquet' (FORMAT PARQUET)")
    
    # Verify
    print("✅ Stadium data created")
    print(f"   - {len(stadiums_df)} stadiums")
    print(f"   - {len(cities_df)} cities")
    
    # Show summary
    summary = conn.execute("""
        SELECT 
            roof_type,
            COUNT(*) as count,
            STRING_AGG(team, ', ' ORDER BY team) as teams
        FROM stadiums
        WHERE is_current = true
        GROUP BY roof_type
        ORDER BY count DESC
    """).df()
    
    print("\n📊 Stadium breakdown:")
    print(summary.to_string(index=False))
    
    # Show which teams need weather
    outdoor_teams = conn.execute("""
        SELECT team, stadium_name, city, state
        FROM stadiums
        WHERE roof_type = 'outdoors' AND is_current = true
        ORDER BY team
    """).df()
    
    print(f"\n🌤️  {len(outdoor_teams)} teams need weather data:")
    print(outdoor_teams.to_string(index=False))
    
    conn.close()
    print("\n✅ Complete! Files created:")
    print("   - data/static/stadiums.parquet")
    print("   - data/static/cities.parquet")
    print("   - databases/seamus.db")