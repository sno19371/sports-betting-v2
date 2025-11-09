#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
	API_URLS,
	MIN_EDGE, MIN_KELLY_FRACTION, MAX_BET_SIZE,
	MODEL_DIR,
	normalize_team_abbr,
)
from db.duck import connect, query

ET = ZoneInfo("America/Detroit")
SUPPORTED_PROPS = {"passing_yards", "rushing_yards", "receiving_yards"}


def _now_et() -> datetime:
	return datetime.now(tz=ET)


def normalize_name(s: str) -> str:
	if s is None:
		return ""
	s = s.strip().lower()
	s = s.replace(".", " ").replace("-", " ").replace("'", "")
	# drop common suffixes
	for suf in (" jr", " sr", " iii", " ii", " iv"):
		if s.endswith(suf):
			s = s[: -len(suf)]
	s = " ".join(s.split())
	return s


def american_to_decimal(x: str | float | int) -> Optional[float]:
	try:
		if isinstance(x, (int, float)):
			if x >= 1.2:
				return float(x)
			odds = float(x)
		else:
			xs = str(x).strip()
			if xs == "":
				return None
			if xs.replace(".", "", 1).isdigit():
				val = float(xs)
				if val >= 1.2:
					return val
				odds = val
			else:
				odds = float(xs)
		if odds > 0:
			return 1.0 + (odds / 100.0)
		else:
			return 1.0 + (100.0 / abs(odds))
	except Exception:
		return None


def phi(z: float) -> float:
	return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_over_prob(line: float, mu: float, sigma: float) -> float:
	sigma = max(8.0, float(sigma))
	z = (line - mu) / sigma
	return 1.0 - phi(z)


def latest_file(patterns: List[str]) -> Optional[Path]:
	paths: List[Path] = []
	for pat in patterns:
		for p in glob.glob(pat):
			paths.append(Path(p))
	if not paths:
		return None
	paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return paths[0]


def fetch_upcoming_teams(hours_ahead: int) -> Tuple[List[str], List[Dict]]:
	url = API_URLS["espn_scoreboard"]
	r = requests.get(url, timeout=15)
	r.raise_for_status()
	data = r.json()

	now = _now_et()
	cutoff = now + timedelta(hours=hours_ahead)
	teams: set[str] = set()
	games: List[Dict] = []

	for ev in data.get("events", []):
		comps = ev.get("competitions", [])
		if not comps:
			continue
		comp = comps[0]
		dt_str = comp.get("date")
		try:
			gtime = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(ET)
		except Exception:
			continue
		# Only games that start in the future and within the window
		if gtime < now or gtime > cutoff:
			continue
		comp_teams = []
		for c in comp.get("competitors", []):
			abbr = c.get("team", {}).get("abbreviation")
			if abbr:
				abbr = normalize_team_abbr(abbr.upper().strip())
				teams.add(abbr)
				comp_teams.append(abbr)
		games.append({"time_et": gtime.isoformat(), "teams": comp_teams})

	return sorted(teams), games


def autodiscover_odds_file() -> Optional[Path]:
	return latest_file([
		"data/processed/player_props_*.csv",
		"data/processed/odds_*.csv",
		"data/processed/*props*.csv",
	])


def load_odds(path: Path, upcoming_teams: List[str]) -> pd.DataFrame:
	df = pd.read_csv(path)
	cols = {c.lower(): c for c in df.columns}

	def pick(*names):
		for n in names:
			if n.lower() in cols:
				return cols[n.lower()]
		return None

	c_player = pick("player", "player_name", "name")
	c_team = pick("posteam", "team", "abbr")
	c_prop = pick("prop_type", "prop", "market")
	c_line = pick("line", "prop_line", "threshold", "value")
	c_o = pick("over_odds", "over_price", "o_price", "over", "o")
	c_u = pick("under_odds", "under_price", "u_price", "under", "u")

	for need, var in [("player", c_player), ("team", c_team), ("prop_type", c_prop), ("line", c_line), ("over_odds", c_o), ("under_odds", c_u)]:
		if var is None:
			raise SystemExit(f"Odds file missing required column like '{need}' (got columns: {list(df.columns)})")

	out = df[[c_player, c_team, c_prop, c_line, c_o, c_u]].copy()
	out.columns = ["player", "posteam", "prop_type", "line", "over_odds", "under_odds"]
	out["posteam"] = out["posteam"].astype(str).str.strip().str.upper().map(normalize_team_abbr)
	out["player_norm"] = out["player"].astype(str).map(normalize_name)
	out = out[out["prop_type"].isin(SUPPORTED_PROPS)]
	out = out[out["posteam"].isin(set(upcoming_teams))]
	out["dec_over"] = out["over_odds"].map(american_to_decimal)
	out["dec_under"] = out["under_odds"].map(american_to_decimal)
	out = out.dropna(subset=["dec_over", "dec_under", "line"])
	out["line"] = out["line"].astype(float)
	return out


def get_latest_player_features(upcoming_teams: List[str]) -> pd.DataFrame:
	con = connect(read_only=True)
	if not upcoming_teams:
		return pd.DataFrame()
	# normalize team abbreviations for consistency with DuckDB
	upcoming_teams = [normalize_team_abbr(t) for t in upcoming_teams]
	placeholders = ",".join(["?"] * len(upcoming_teams))
	sql = f"""
    WITH latest AS (
      SELECT
        player_id, player, posteam, game_date,
        qb_yards_last5, rb_yards_last5, rec_yards_last5,
        qb_yards_std5,  rb_yards_std5,  rec_yards_std5,
        qb_pass_att_last5, rb_carries_last5, rec_targets_last5,
        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY game_date DESC) rn
      FROM analytics.player_games_roll5
      WHERE posteam IN ({placeholders})
    )
    SELECT * FROM latest WHERE rn=1
    """
	return query(con, sql, params=list(upcoming_teams)).assign(
		player_norm=lambda d: d["player"].astype(str).map(normalize_name)
	)


def choose_side(row: pd.Series) -> pd.Series:
	prop = row["prop_type"]
	line = float(row["line"])

	if prop == "passing_yards":
		mu = row.get("qb_yards_last5")
		if pd.isna(mu) or mu is None:
			mu = (row.get("qb_pass_att_last5") or 0.0) * 6.9
		sigma = row.get("qb_yards_std5") or 45.0
		sigma = max(float(sigma), 8.0)
	elif prop == "rushing_yards":
		mu = row.get("rb_yards_last5")
		if pd.isna(mu) or mu is None:
			mu = (row.get("rb_carries_last5") or 0.0) * 3.9
		sigma = row.get("rb_yards_std5") or 20.0
		sigma = max(float(sigma), 8.0)
	else:
		mu = row.get("rec_yards_last5")
		if pd.isna(mu) or mu is None:
			mu = (row.get("rec_targets_last5") or 0.0) * 8.7
		sigma = row.get("rec_yards_std5") or 22.0
		sigma = max(float(sigma), 8.0)

	p_over = normal_over_prob(line, mu, sigma)
	p_under = 1.0 - p_over

	d_over = row["dec_over"]
	d_under = row["dec_under"]
	p_be_over = 1.0 / d_over
	p_be_under = 1.0 / d_under

	edge_over = p_over - p_be_over
	edge_under = p_under - p_be_under

	def kelly(p: float, d: float) -> float:
		b = max(d - 1.0, 1e-9)
		f = (p * b - (1.0 - p)) / b
		return max(0.0, f)

	k_over = kelly(p_over, d_over)
	k_under = kelly(p_under, d_under)

	if edge_over >= edge_under:
		side, best_edge, k_rec = "over", edge_over, k_over
	else:
		side, best_edge, k_rec = "under", edge_under, k_under

	actionable = (best_edge >= MIN_EDGE) and (k_rec >= MIN_KELLY_FRACTION)
	k_rec = min(k_rec, MAX_BET_SIZE)

	return pd.Series({
		"pred_median": mu,
		"pred_sigma": sigma,
		"p_over": p_over,
		"p_under": p_under,
		"edge_over": edge_over,
		"edge_under": edge_under,
		"best_side": side,
		"best_edge": best_edge,
		"kelly_rec": k_rec,
		"is_actionable": bool(actionable),
	})


def maybe_load_model(_path: Path):
	try:
		from modeling import ModelManager  # noqa: F401
		return None
	except Exception:
		return None


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--hours-ahead", type=int, default=48)
	ap.add_argument("--odds", type=str, default=None)
	ap.add_argument("--model-file", type=str, default=str(Path(MODEL_DIR) / "model_manager.pkl"))
	ap.add_argument("--output", type=str, default=f"data/processed/bets_{_now_et().strftime('%Y%m%d')}.csv")
	args = ap.parse_args()

	teams, games = fetch_upcoming_teams(args.hours_ahead)
	if not teams:
		print(f"Found 0 upcoming NFL teams in next {args.hours_ahead}h. Try --hours-ahead 72.")
		return
	print("UPCOMING TEAMS:", ", ".join(teams))

	if args.odds:
		odds_path = Path(args.odds)
	else:
		odds_path = autodiscover_odds_file()
	if not odds_path or not Path(odds_path).exists():
		raise SystemExit("No odds CSV provided/found. Put one under data/processed/ (e.g., player_props_YYYYMMDD.csv).")

	odds = load_odds(Path(odds_path), teams)
	if odds.empty:
		raise SystemExit("Odds loaded, but none matched upcoming teams/props. Check team abbreviations and columns.")

	feats = get_latest_player_features(teams)
	if feats.empty:
		raise SystemExit("No features returned for upcoming teams. Verify analytics.player_games_roll5 exists & has data.")

	feats["key"] = feats["player_norm"] + "|" + feats["posteam"].astype(str)
	odds["key"] = odds["player_norm"] + "|" + odds["posteam"].astype(str)

	merged = odds.merge(feats, on="key", how="left", suffixes=("", "_feat"))
	missing = merged["player"].isna().sum()
	if missing:
		print(f"⚠️  {missing} odds rows did not match features (name/team mismatch). They will be dropped.")
		merged = merged.dropna(subset=["player"])

	results = merged.copy()
	enrich = results.apply(choose_side, axis=1)
	results = pd.concat([results, enrich], axis=1)
	results.insert(0, "as_of", _now_et().strftime("%Y-%m-%d %H:%M:%S %Z"))

	show_cols = ["player", "posteam", "prop_type", "line", "pred_median", "pred_sigma", "best_side", "best_edge", "kelly_rec", "is_actionable"]
	out_cols = [
		"as_of", "player", "posteam", "prop_type", "line",
		"over_odds", "under_odds", "dec_over", "dec_under",
		"pred_median", "pred_sigma", "p_over", "p_under",
		"edge_over", "edge_under", "best_side", "best_edge",
		"kelly_rec", "is_actionable",
	]

	results_sorted = results.sort_values("best_edge", ascending=False)
	print("\nTOP ACTIONABLE:")
	print(results_sorted[results_sorted["is_actionable"]][show_cols].head(20).to_string(index=False))
	print("\nTOP OVERALL:")
	print(results_sorted[show_cols].head(15).to_string(index=False))

	out_path = Path(args.output)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	results_sorted[out_cols].to_csv(out_path, index=False)
	print(f"\n✓ Wrote {out_path} ({len(results_sorted)} rows)")


if __name__ == "__main__":
	main()


