# sports-
to bet on a new week:


UPDATE OUR PARQUET FILE WITH DATA FROM PREVIOUS WEEK:
    Get the odds from the last week by grabbing the new nfL_odds xlsx from https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
    run xlsx_to_csv.py to confert to csv


    get the updates to play_by_play_2025 parquet file from https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv

    re-run the pbp combine script to generate a new pbp_combined_curr.parquet file

    get the updates to depth_charts parquet file from https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2025.parquet

    run the depth_charts combine script to generate a new depth_charts_combined_curr.parquet

    run the build_weather_cache_curr_v1.py script to update the historical weather caches with their correct data for the most recent data

    get the updates to roster_weekly parquet file from https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2025.parquet

    run the combine_roster_weekly script to generate a new roster_weekly_combined_curr.parquet file

GET CURRENT DATA FOR PLACING BETS:
    run get_rush_attempts_lines_from_api_curr_v1 in scripts to generate json of odds for the rush attempts and the over/under and spread datapoints

    run get_all_rbs_injury_status_curr_v1 to get current rb injuries

    run build_game_weather_curr.py to get the current weather forecasts

    run build_rb_depth_chart_curr.py to get the current depth charts

    run generate_player_model_rows_curr.py to get the current rows for our model

    run generate_predictions.py to generate predictions

    
    MONEY IN ACCOUNTS:
    BetMGM $50
    