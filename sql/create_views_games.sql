PRAGMA enable_verification;

CREATE SCHEMA IF NOT EXISTS analytics;

-- Base games view with handy computed fields
CREATE OR REPLACE VIEW analytics.games AS
SELECT
  game_id,
  season,
  week,
  CAST(gameday AS TIMESTAMP)               AS game_ts,
  DATE(gameday)                            AS game_date,
  home_team,
  away_team,
  stadium_id,
  roof,
  spread_line,
  total_line,
  (LOWER(roof) IN ('dome','closed'))       AS is_dome,
  (LOWER(roof) = 'outdoors')               AS is_outdoors,
  CASE
    WHEN spread_line IS NULL THEN NULL
    WHEN spread_line < 0 THEN home_team
    WHEN spread_line > 0 THEN away_team
    ELSE NULL
  END                                       AS favorite_team,
  ABS(spread_line)                          AS favorite_spread_abs
FROM historical_games;

-- Stadium metadata view (columns exist per complete_stadium_data.py)
CREATE OR REPLACE VIEW analytics.games_with_stadium AS
SELECT
  g.*,
  s.stadium_name,
  s.city,
  s.state
FROM analytics.games g
LEFT JOIN stadiums s
  ON g.stadium_id = s.stadium_id;

-- Weather joined view (column names align with collect_historical_data.py schema)
CREATE OR REPLACE VIEW analytics.games_with_weather AS
SELECT
  g.*,
  w.temperature_f         AS temp_f,
  w.apparent_temp_f       AS feels_f,
  w.wind_speed_mph        AS wind_mph,
  w.wind_gust_mph         AS wind_gust_mph,
  w.wind_direction_deg    AS wind_deg,
  w.humidity_pct          AS humidity_pct,
  w.precipitation_inches  AS precip_in,
  w.cloud_cover_pct       AS cloud_cover_pct,
  w.weather_summary       AS weather_summary
FROM analytics.games g
LEFT JOIN historical_weather w
  ON w.game_id = g.game_id;


