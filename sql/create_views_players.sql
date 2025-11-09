PRAGMA enable_verification;

CREATE SCHEMA IF NOT EXISTS analytics;

-- Base passthrough (change 'joined_games' if you persisted a different name)
CREATE OR REPLACE VIEW analytics.player_games AS
SELECT * FROM joined_games;

-- Minimal, fast-to-scan subset for common modeling keys
CREATE OR REPLACE VIEW analytics.player_keys AS
SELECT
  season, week, game_date, game_id,
  posteam, player_id, player,
  has_qb, has_rb, has_rec
FROM analytics.player_games;

-- Role presence quick filters
CREATE OR REPLACE VIEW analytics.player_qbs AS
SELECT * FROM analytics.player_games WHERE has_qb;

CREATE OR REPLACE VIEW analytics.player_rbs AS
SELECT * FROM analytics.player_games WHERE has_rb;

CREATE OR REPLACE VIEW analytics.player_recs AS
SELECT * FROM analytics.player_games WHERE has_rec;

-- Example rolling 5-game means (safe if these columns exist)
CREATE OR REPLACE VIEW analytics.player_games_roll5 AS
WITH base AS (
  SELECT
    *,
    AVG(COALESCE(qb_pass_att, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS qb_pass_att_last5,
    AVG(COALESCE(rb_carries, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rb_carries_last5,
    AVG(COALESCE(rec_targets, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rec_targets_last5,
    -- Rolling yards (mean over last 5)
    AVG(COALESCE(qb_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS qb_yards_last5,
    AVG(COALESCE(rb_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rb_yards_last5,
    AVG(COALESCE(rec_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rec_yards_last5,
    -- Rolling yards (std dev over last 5)
    STDDEV_SAMP(COALESCE(qb_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS qb_yards_std5,
    STDDEV_SAMP(COALESCE(rb_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rb_yards_std5,
    STDDEV_SAMP(COALESCE(rec_yards, 0)) OVER (
      PARTITION BY player_id ORDER BY game_date
      ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rec_yards_std5
  FROM analytics.player_games
)
SELECT * FROM base;


