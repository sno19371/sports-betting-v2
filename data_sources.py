# data_sources.py
import nfl_data_py as nfl
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
from config import *
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLDataSources:
    def __init__(self):
        self.current_season = CURRENT_SEASON
        self.rate_limits = API_RATE_LIMITS
        
        # Validate API keys
        self._validate_api_keys()
        
    def _validate_api_keys(self):
        """Validate that required API keys are present"""
        required_keys = {
            'OpenWeather': OPENWEATHER_API_KEY,
            'Odds API': ODDS_API_KEY
        }
        
        missing_keys = [name for name, key in required_keys.items() if not key]
        
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        else:
            logger.info("✅ All required API keys found")
    
    def _rate_limit(self, api_name):
        """Apply rate limiting between API calls"""
        if api_name in self.rate_limits:
            time.sleep(self.rate_limits[api_name])
    
    # =============================================================================
    # NFL DATA (nfl-data-py) - FREE
    # =============================================================================
    
    def get_weekly_stats(self, years=None, save=True):
        """Get weekly player statistics"""
        if years is None:
            years = HISTORICAL_YEARS
            
        logger.info(f"📊 Loading weekly stats for years: {years}")
        
        try:
            weekly_data = nfl.import_weekly_data(years=years)
            
            if save:
                filename = DATA_FILES['weekly_stats'].format(
                    start_year=min(years), 
                    end_year=max(years)
                )
                weekly_data.to_csv(filename, index=False)
                logger.info(f"💾 Saved to {filename}")
                
            return weekly_data
            
        except Exception as e:
            logger.error(f"❌ Error loading weekly stats: {e}")
            return None
    
    def get_pbp_data(self, years=None, save=True):
        """Get play-by-play data"""
        if years is None:
            years = [self.current_season]
            
        logger.info(f"🏈 Loading play-by-play for years: {years}")
        
        try:
            pbp_data = nfl.import_pbp_data(years=years)
            
            if save:
                for year in years:
                    year_data = pbp_data[pbp_data['season'] == year]
                    filename = DATA_FILES['pbp_data'].format(year=year)
                    year_data.to_parquet(filename)
                    logger.info(f"💾 Saved {year} PBP data to {filename}")
                    
            return pbp_data
            
        except Exception as e:
            logger.error(f"❌ Error loading PBP data: {e}")
            return None
    
    def get_rosters(self, years=None):
        """Get team rosters"""
        if years is None:
            years = [self.current_season]
            
        try:
            return nfl.import_rosters(years=years)
        except Exception as e:
            logger.error(f"❌ Error loading rosters: {e}")
            return None
    
    def get_injury_reports_nfl(self, years=None):
        """Get injury reports from nfl-data-py"""
        if years is None:
            years = [self.current_season]
            
        try:
            return nfl.import_injuries(years=years)
        except Exception as e:
            logger.error(f"❌ Error loading injury reports: {e}")
            return None
    
    # =============================================================================
    # ESPN API - FREE
    # =============================================================================
    
    def get_espn_scoreboard(self, week=None, season=None):
        """Get games and scores from ESPN"""
        if season is None:
            season = self.current_season
            
        try:
            if week:
                url = f"{API_URLS['espn_scoreboard']}?dates={season}&seasontype=2&week={week}"
            else:
                url = API_URLS['espn_scoreboard']
                
            response = requests.get(url, timeout=10)
            self._rate_limit('espn')
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"ESPN scoreboard returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ ESPN scoreboard error: {e}")
            return None
    
    def get_espn_team_injuries(self, team_abbr):
        """Get team injury report from ESPN"""
        espn_id = get_team_espn_id(team_abbr)
        if not espn_id:
            logger.warning(f"No ESPN ID found for team {team_abbr}")
            return None
            
        try:
            url = API_URLS['espn_team_injuries'].format(team_id=espn_id)
            response = requests.get(url, timeout=10)
            self._rate_limit('espn')
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"ESPN injuries returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ ESPN injuries error for {team_abbr}: {e}")
            return None
    
    def _get_espn_roster(self, team_abbr: str) -> Optional[pd.DataFrame]:
        """
        Fetch current roster for a single NFL team from ESPN API.
        
        This is a private helper function used by get_live_week_data_espn() to
        build the complete roster dataset for upcoming games.
        
        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF', 'SF')
            
        Returns:
            DataFrame with columns: player_id, player_display_name, position, team
            Returns None if API call fails or team is invalid
            
        Example:
            >>> roster_df = self._get_espn_roster('KC')
            >>> print(roster_df.head())
               player_id  player_display_name  position  team
            0  3139477    Patrick Mahomes      QB        KC
            1  4040715    Travis Kelce         TE        KC
        """
        # Normalize team abbreviation (handles ESPN aliases like WSH -> WAS)
        from config import normalize_team_abbr
        team_abbr = normalize_team_abbr(team_abbr)
        
        # Validate team abbreviation
        if team_abbr not in NFL_TEAMS:
            logger.error(f"Invalid team abbreviation: {team_abbr}")
            return None
        
        # Get ESPN team ID
        espn_id = get_team_espn_id(team_abbr)
        if not espn_id:
            logger.error(f"No ESPN ID found for team {team_abbr}")
            return None
        
        try:
            # Construct roster URL
            url = API_URLS['espn_roster'].format(team_id=espn_id)
            
            # Make API request
            response = requests.get(url, timeout=10)
            self._rate_limit('espn')
            
            if response.status_code != 200:
                logger.warning(f"ESPN roster API returned status {response.status_code} for {team_abbr}")
                return None
            
            # Parse JSON response
            data = response.json()
            
            # Navigate to athletes list
            if 'athletes' not in data:
                logger.warning(f"No 'athletes' key in ESPN roster response for {team_abbr}")
                return None
            
            athletes_groups = data['athletes']
            
            if not athletes_groups:
                logger.warning(f"Empty roster returned for {team_abbr}")
                return None
            
            # ESPN returns grouped data: offense, defense, specialTeam, etc.
            # Each group has an 'items' array with the actual players
            all_players = []
            
            for group in athletes_groups:
                # Get the position group name (offense, defense, specialTeam)
                group_name = group.get('position', 'unknown')
                
                # Get the items array for this group
                items = group.get('items', [])
                
                logger.debug(f"Processing {group_name} group: {len(items)} players")
                
                # Extract player information from items
                for athlete in items:
                    try:
                        # Extract required fields with fallbacks
                        player_id = athlete.get('id')
                        full_name = athlete.get('fullName') or athlete.get('displayName')
                        
                        # Get position - handle nested structure
                        position_obj = athlete.get('position', {})
                        if isinstance(position_obj, dict):
                            position = position_obj.get('abbreviation') or position_obj.get('name')
                        else:
                            position = None
                        
                        # Skip if missing critical data
                        if not all([player_id, full_name, position]):
                            logger.debug(f"Skipping player with incomplete data: {athlete.get('id')}")
                            continue
                        
                        # Only include relevant positions for betting
                        if position not in RELEVANT_POSITIONS:
                            continue
                        
                        all_players.append({
                            'player_id': str(player_id),
                            'player_display_name': full_name,
                            'position': position,
                            'team': team_abbr
                        })
                        
                    except Exception as e:
                        logger.debug(f"Error parsing player in {team_abbr} roster: {e}")
                        continue
            
            if not all_players:
                logger.warning(f"No valid players found in {team_abbr} roster after filtering")
                return None
            
            # Create DataFrame
            roster_df = pd.DataFrame(all_players)
            
            logger.info(f"✓ Fetched {len(roster_df)} players for {team_abbr} ({roster_df['position'].value_counts().to_dict()})")
            
            return roster_df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching roster for {team_abbr}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching roster for {team_abbr}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching roster for {team_abbr}: {e}")
            return None
    
    def get_live_week_data_espn(self, season: int, week: int) -> Optional[pd.DataFrame]:
        """
        Construct complete live week roster dataset for all teams playing in specified week.
        
        This is the "newspaper" function - it fetches who is playing THIS WEEK using
        real-time ESPN data, without needing historical performance stats for the
        upcoming games (which don't exist yet).
        
        The function:
        1. Fetches the scoreboard to see which teams are playing
        2. Fetches rosters for all teams in those games
        3. Builds a DataFrame with player metadata and opponent matchups
        4. Returns data structured to match historical data format
        
        Args:
            season: NFL season year (e.g., 2025)
            week: Week number (1-18)
            
        Returns:
            DataFrame with columns:
                - player_id: ESPN player ID
                - player_display_name: Full player name
                - position: Position abbreviation (QB, RB, WR, TE, K)
                - team: Team abbreviation
                - opponent: Opponent abbreviation
                - season: Season year
                - week: Week number
            Returns None if unable to construct dataset
            
        Example:
            >>> live_df = data_api.get_live_week_data_espn(season=2025, week=5)
            >>> print(live_df.head())
               player_id  player_display_name  position  team  opponent  season  week
            0  3139477    Patrick Mahomes      QB        KC    BUF       2025    5
            1  4040715    Travis Kelce         TE        KC    BUF       2025    5
        """
        logger.info(f"🗞️ Fetching live week data for Season {season}, Week {week}")
        
        # Step 1: Fetch scoreboard to get games
        scoreboard = self.get_espn_scoreboard(week=week, season=season)
        
        if not scoreboard:
            logger.error("Failed to fetch ESPN scoreboard - cannot build live week data")
            return None
        
        # Step 2: Parse games from scoreboard
        if 'events' not in scoreboard:
            logger.error("No 'events' key in scoreboard response")
            return None
        
        events = scoreboard['events']
        
        if not events:
            logger.warning(f"No games found for Season {season}, Week {week}")
            return None
        
        logger.info(f"Found {len(events)} game(s) scheduled for Week {week}")
        
        # Extract team matchups from games
        games = []
        for event in events:
            try:
                # Navigate to competitions (usually just one per event)
                competitions = event.get('competitions', [])
                if not competitions:
                    continue
                
                competition = competitions[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) != 2:
                    logger.warning(f"Expected 2 competitors, found {len(competitors)}")
                    continue
                
                # Identify home and away teams
                home_team = None
                away_team = None
                
                from config import normalize_team_abbr
                for competitor in competitors:
                    team_obj = competitor.get('team', {})
                    team_abbr = team_obj.get('abbreviation')
                    # Normalize ESPN abbreviations (e.g., WSH -> WAS, LA -> LAR)
                    team_abbr = normalize_team_abbr(team_abbr)
                    home_away = competitor.get('homeAway')
                    
                    if not team_abbr:
                        continue
                    
                    if home_away == 'home':
                        home_team = team_abbr
                    elif home_away == 'away':
                        away_team = team_abbr
                
                if home_team and away_team:
                    games.append((home_team, away_team))
                    logger.info(f"  Game: {away_team} @ {home_team}")
                else:
                    logger.warning(f"Could not determine home/away teams for event {event.get('id')}")
                    
            except Exception as e:
                logger.error(f"Error parsing event: {e}")
                continue
        
        if not games:
            logger.error("No valid games parsed from scoreboard")
            return None
        
        # Step 3: Create team-opponent mapping
        team_matchups = []
        for home_team, away_team in games:
            team_matchups.append({'team': home_team, 'opponent': away_team})
            team_matchups.append({'team': away_team, 'opponent': home_team})
        
        logger.info(f"Created matchups for {len(team_matchups)} team entries")
        
        # Step 4: Fetch rosters for all teams
        all_rosters = []
        unique_teams = set(tm['team'] for tm in team_matchups)
        
        logger.info(f"Fetching rosters for {len(unique_teams)} teams...")
        
        failed_teams = []
        for team_abbr in sorted(unique_teams):
            roster_df = self._get_espn_roster(team_abbr)
            
            if roster_df is not None and not roster_df.empty:
                all_rosters.append(roster_df)
            else:
                failed_teams.append(team_abbr)
                logger.warning(f"Failed to fetch roster for {team_abbr}")
        
        if failed_teams:
            logger.warning(f"Failed to fetch rosters for {len(failed_teams)} team(s): {failed_teams}")
        
        if not all_rosters:
            logger.error("No rosters fetched successfully - cannot build live week data")
            return None
        
        # Step 5: Combine all rosters
        combined_roster = pd.concat(all_rosters, ignore_index=True)
        logger.info(f"Combined {len(combined_roster)} players from {len(all_rosters)} team rosters")
        
        # Step 6: Merge with matchups to add opponent information
        matchup_df = pd.DataFrame(team_matchups)
        live_data = combined_roster.merge(matchup_df, on='team', how='left')
        
        # Step 7: Add metadata columns
        live_data['season'] = season
        live_data['week'] = week
        
        # Step 8: Standardize column order to match historical data
        column_order = [
            'player_id',
            'player_display_name',
            'position',
            'team',
            'opponent',
            'season',
            'week'
        ]
        
        # Ensure all columns exist
        for col in column_order:
            if col not in live_data.columns:
                logger.error(f"Missing expected column: {col}")
                return None
        
        live_data = live_data[column_order]
        
        # Step 9: Validation
        missing_opponents = live_data['opponent'].isna().sum()
        if missing_opponents > 0:
            logger.warning(f"{missing_opponents} players missing opponent information")
        
        # Log summary statistics
        logger.info("=" * 60)
        logger.info("LIVE WEEK DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Season: {season}, Week: {week}")
        logger.info(f"Total players: {len(live_data)}")
        logger.info(f"Teams: {live_data['team'].nunique()}")
        logger.info(f"Games: {len(games)}")
        logger.info(f"Position breakdown: {live_data['position'].value_counts().to_dict()}")
        logger.info("=" * 60)
        
        return live_data
    
    # =============================================================================
    # WEATHER API - FREE TIER
    # =============================================================================
    
    def get_weather_data(self, city, save=True):
        """Get weather data for a city"""
        if not OPENWEATHER_API_KEY:
            logger.warning("No OpenWeather API key found")
            return None
            
        try:
            url = f"{API_URLS['openweather']}?q={city}&appid={OPENWEATHER_API_KEY}&units=imperial"
            response = requests.get(url, timeout=10)
            self._rate_limit('openweather')
            
            if response.status_code == 200:
                data = response.json()
                
                if save:
                    filename = DATA_FILES['weather_data'].format(
                        date=datetime.now().strftime('%Y%m%d_%H%M')
                    )
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                        
                return data
            else:
                logger.warning(f"Weather API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Weather API error for {city}: {e}")
            return None
    
    def get_game_weather(self, team_abbr):
        """Get weather for a team's home city"""
        city = get_team_city(team_abbr)
        if not city:
            return None
            
        # Check if it's a dome team
        if is_dome_team(team_abbr):
            return {
                'dome': True,
                'city': city,
                'weather_impact': None
            }
            
        weather_data = self.get_weather_data(city)
        if weather_data:
            return {
                'dome': False,
                'city': city,
                'temp': weather_data.get('main', {}).get('temp'),
                'feels_like': weather_data.get('main', {}).get('feels_like'),
                'humidity': weather_data.get('main', {}).get('humidity'),
                'wind_speed': weather_data.get('wind', {}).get('speed'),
                'weather_main': weather_data.get('weather', [{}])[0].get('main'),
                'description': weather_data.get('weather', [{}])[0].get('description')
            }
        return None
    
    # =============================================================================
    # ODDS API - FREE TIER
    # =============================================================================
    
    def get_nfl_odds(self, markets=['h2h', 'spreads', 'totals'], save=True):
        """Get NFL odds from The Odds API"""
        if not ODDS_API_KEY:
            logger.warning("No Odds API key found")
            return None
            
        try:
            markets_str = ','.join(markets)
            url = f"{API_URLS['odds_api']}/odds/?apiKey={ODDS_API_KEY}&regions=us&markets={markets_str}"
            
            response = requests.get(url, timeout=15)
            self._rate_limit('odds_api')
            
            if response.status_code == 200:
                data = response.json()
                
                # Log remaining requests
                remaining = response.headers.get('x-requests-remaining', 'Unknown')
                logger.info(f"📊 Odds API requests remaining: {remaining}")
                
                if save:
                    filename = DATA_FILES['odds_data'].format(
                        date=datetime.now().strftime('%Y%m%d_%H%M')
                    )
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                        
                return data
            else:
                logger.warning(f"Odds API returned status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Odds API error: {e}")
            return None
    
    def get_player_props_odds(self, markets: List[str] = ['player_pass_yds', 'player_rush_yds', 'player_reception_yds', 'player_receptions', 'player_pass_tds', 'player_rush_tds', 'player_reception_tds']) -> Optional[List[Dict]]:
        """
        Get player prop odds using the correct two-step process based on API documentation.
        1. Fetch all upcoming games to get their event IDs from the free /events endpoint.
        2. For each event ID, fetch the detailed player prop markets.
        
        Args:
            markets: A list of specific player prop market keys to query.
            
        Returns:
            A formatted list of available player prop bet dictionaries.
        """
        if not ODDS_API_KEY:
            logger.warning("No Odds API key found")
            return None

        # --- Step 1: Get upcoming games and their event IDs ---
        try:
            events_url = f"{API_URLS['odds_api_events']}/?apiKey={ODDS_API_KEY}"
            response = requests.get(events_url, timeout=15)
            # This endpoint does not use quota, no need to self.rate_limit
            
            if response.status_code != 200:
                logger.error(f"❌ Odds API (events) returned status {response.status_code}: {response.text}")
                return None
            
            games = response.json()
            game_ids = [game['id'] for game in games]
            logger.info(f"Found {len(game_ids)} upcoming games from the /events endpoint.")
            
        except Exception as e:
            logger.error(f"❌ Odds API (events) error: {e}")
            return None

        # --- Step 2: For each game, fetch its player prop odds ---
        all_bets = {}
        markets_str = ','.join(markets)
        
        logger.info(f"Fetching player props for {len(game_ids)} games...")
        for game_id in game_ids:
            try:
                props_url = (
                    f"{API_URLS['odds_api']}/events/{game_id}/odds"
                    f"?apiKey={ODDS_API_KEY}&regions=us&markets={markets_str}&oddsFormat=american"
                )
                
                response = requests.get(props_url, timeout=20)
                self._rate_limit('odds_api')
                
                if response.status_code == 200:
                    game_with_odds = response.json()
                    
                    # Parse the complex nested JSON response
                    for bookmaker in game_with_odds.get('bookmakers', []):
                        for market in bookmaker.get('markets', []):
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description')
                                line = outcome.get('point')
                                price = outcome.get('price')
                                side = outcome.get('name') # "Over" or "Under"

                                if not all([player_name, line, price, side]):
                                    continue

                                # Group the Over/Under outcomes for the same player/line
                                bet_key = (player_name, market['key'], line, bookmaker['key'])
                                if bet_key not in all_bets:
                                    all_bets[bet_key] = {
                                        'player_name': player_name,
                                        'prop_type': market['key'],
                                        'line': line,
                                        'bookmaker': bookmaker['title']
                                    }
                                
                                if side == 'Over':
                                    all_bets[bet_key]['over_odds'] = price
                                elif side == 'Under':
                                    all_bets[bet_key]['under_odds'] = price
                else:
                    logger.warning(f"Odds API (props for game {game_id}) returned status {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Odds API (props for game {game_id}) error: {e}")
                continue
        
        # Filter for bets that have both Over and Under odds
        final_bets = [bet for bet in all_bets.values() if 'over_odds' in bet and 'under_odds' in bet]
        
        logger.info(f"Successfully formatted {len(final_bets)} complete player prop markets.")
        return final_bets
    
    # =============================================================================
    # DATA INTEGRATION AND ANALYSIS
    # =============================================================================
    
    def get_player_complete_data(self, player_name, weeks_back=10):
        """Get comprehensive data for a player"""
        logger.info(f"🔍 Getting complete data for {player_name}")
        
        # Get weekly stats
        weekly_data = self.get_weekly_stats(save=False)
        if weekly_data is None:
            return None
            
        # Find player across display name and abbreviated name columns
        name_parts = [p for p in str(player_name).split() if p]
        abbr_name = None
        if len(name_parts) >= 2:
            abbr_name = f"{name_parts[0][0]}.{name_parts[-1]}"  # e.g., T.Hill

        has_display = 'player_display_name' in weekly_data.columns
        has_abbrev = 'player_name' in weekly_data.columns

        mask = pd.Series(False, index=weekly_data.index)
        if has_display:
            mask = mask | weekly_data['player_display_name'].str.fullmatch(player_name, case=False, na=False)
            mask = mask | weekly_data['player_display_name'].str.contains(player_name, case=False, na=False)
        if has_abbrev and abbr_name:
            mask = mask | weekly_data['player_name'].str.fullmatch(abbr_name, case=False, na=False)
            mask = mask | weekly_data['player_name'].str.contains(abbr_name, case=False, na=False)

        player_data = weekly_data[mask].copy()

        if player_data.empty:
            logger.warning(f"No data found for {player_name}")
            return None
        
        # Get recent games
        current_season = player_data[player_data['season'] == self.current_season]
        recent_games = current_season.sort_values('week').tail(weeks_back)
        
        # Get team and weather info
        if not recent_games.empty:
            team_col = 'team' if 'team' in recent_games.columns else ('recent_team' if 'recent_team' in recent_games.columns else None)
            team = recent_games[team_col].iloc[-1] if team_col else None
            weather = self.get_game_weather(team)
        else:
            team = None
            weather = None
        
        resolved_name = (
            player_data['player_display_name'].iloc[0]
            if has_display and not player_data['player_display_name'].isna().all()
            else (player_data['player_name'].iloc[0] if has_abbrev else player_name)
        )

        return {
            'player_name': resolved_name,
            'team': team,
            'weekly_data': recent_games,
            'career_data': player_data,
            'weather_conditions': weather,
            'data_timestamp': datetime.now()
        }
    
    def get_matchup_data(self, team1, team2):
        """Get matchup-specific data"""
        logger.info(f"🏈 Getting matchup data: {team1} vs {team2}")
        
        matchup_data = {
            'team1': team1,
            'team2': team2,
            'team1_weather': self.get_game_weather(team1),
            'team2_weather': self.get_game_weather(team2),
            'odds': self.get_nfl_odds(save=False),
            'timestamp': datetime.now()
        }
        
        return matchup_data
    
    def test_all_connections(self):
        """Test all data source connections"""
        logger.info("🧪 Testing all data connections")
        
        tests = []
        
        # Test NFL data
        try:
            test_data = self.get_weekly_stats(years=[2024], save=False)
            if test_data is not None:
                tests.append(("✅ NFL Data", f"{len(test_data):,} records"))
            else:
                tests.append(("❌ NFL Data", "Failed to load"))
        except Exception as e:
            tests.append(("❌ NFL Data", str(e)))
        
        # Test ESPN API
        try:
            scoreboard = self.get_espn_scoreboard()
            if scoreboard:
                tests.append(("✅ ESPN API", "Connected"))
            else:
                tests.append(("❌ ESPN API", "No data"))
        except Exception as e:
            tests.append(("❌ ESPN API", str(e)))
        
        # Test Weather API
        try:
            weather = self.get_weather_data("Kansas City", save=False)
            if weather:
                temp = weather.get('main', {}).get('temp', 'Unknown')
                tests.append(("✅ Weather API", f"{temp}°F in KC"))
            else:
                tests.append(("❌ Weather API", "No data"))
        except Exception as e:
            tests.append(("❌ Weather API", str(e)))
        
        # Test Odds API
        try:
            odds = self.get_nfl_odds(save=False)
            if odds:
                tests.append(("✅ Odds API", f"{len(odds)} games"))
            else:
                tests.append(("❌ Odds API", "No data"))
        except Exception as e:
            tests.append(("❌ Odds API", str(e)))
        
        # Print results
        print("\n" + "="*50)
        print("🔌 DATA SOURCE CONNECTION TEST")
        print("="*50)
        for test_name, result in tests:
            print(f"{test_name}: {result}")
        print("="*50)
        
        return tests

def main():
    """Test the data sources"""
    print("🚀 NFL Data Sources Pipeline Test")
    
    # Initialize data sources
    data_sources = NFLDataSources()
    
    # Test all connections
    data_sources.test_all_connections()
    
    # Test player data retrieval
    print("\n🎯 Testing Player Data Retrieval")
    player_data = data_sources.get_player_complete_data("Tyreek Hill")
    
    if player_data:
        print(f"✅ Found data for {player_data['player_name']}")
        print(f"📊 Team: {player_data['team']}")
        print(f"🎮 Recent games: {len(player_data['weekly_data'])}")
        if player_data['weather_conditions']:
            print(f"🌤️ Weather: {player_data['weather_conditions']}")
    else:
        print("❌ No player data found")
    
    print("\n🎉 Data sources test complete!")

if __name__ == "__main__":
    main()