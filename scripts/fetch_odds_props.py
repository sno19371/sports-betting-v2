#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from config import API_URLS, PROCESSED_DATA_DIR

# Global map we will extend at runtime once we discover exact keys
MARKET_MAP: Dict[str, str] = {
    "player_pass_yds":        "passing_yards",
    "player_rush_yds":        "rushing_yards",
    "player_reception_yds":   "receiving_yards",   # sometimes seen
    "player_rec_yds":         "receiving_yards",   # legacy alias
    # Common official spellings we may see:
    "player_passing_yards":   "passing_yards",
    "player_rushing_yards":   "rushing_yards",
    "player_receiving_yards": "receiving_yards",
}

# Team full-name -> abbreviation (best-effort)
TEAM_ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
    "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS"
}

def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")

def _norm_name(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return " ".join(s.split())

def get_supported_markets(api_key: str) -> List[str]:
    base = API_URLS["odds_api"].rstrip("/")
    url = f"{base}/markets"
    params = {"regions": "us", "apiKey": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    # The endpoint can return list of strings or list of objects with 'key'
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [d.get("key") for d in data if d.get("key")]
    return list(data)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_in(hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

def _team_to_abbr(s: str) -> str:
    if not s:
        return ""
    s_up = str(s).strip().upper()
    if len(s_up) <= 4:
        return s_up
    return TEAM_ABBR.get(s_up, s_up)


def fetch_events(api_key: str, hours_ahead: int) -> List[Dict]:
    """
    Get upcoming NFL events within a window so we can request per-event odds.
    """
    base = API_URLS["odds_api"].rstrip("/")
    url = f"{base}/events"
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
        "commenceTimeFrom": _now_utc_iso(),
        "commenceTimeTo": _iso_in(hours_ahead),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or []
    print(f"• Found {len(data)} upcoming events in next {hours_ahead}h")
    return data

def fetch_event_markets(api_key: str, event_id: str, regions: str = "us,us2") -> List[str]:
    base = API_URLS["odds_api"].rstrip("/")
    url = f"{base}/events/{event_id}/markets"
    params = {"apiKey": api_key, "regions": regions, "dateFormat": "iso"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    keys: List[str] = []
    for bm in data.get("bookmakers", []):
        for mk in bm.get("markets", []):
            k = mk.get("key")
            if k:
                keys.append(k)
    return keys

def discover_prop_markets(api_key: str, events: List[Dict], want: Tuple[str, str, str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Look through /events/{id}/markets for a handful of upcoming games and
    pick the actual keys for passing/rushing/receiving yards.
    Returns (selected_market_keys, discovered_map_for_MARKET_MAP).
    """
    want_pass, want_rush, want_recv = want
    seen_counts: Dict[str, int] = {}
    sample_events = events[:10]  # keep it cheap

    for ev in sample_events:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        try:
            keys = fetch_event_markets(api_key, ev_id)
        except Exception:
            continue
        for k in keys:
            seen_counts[k] = seen_counts.get(k, 0) + 1

    def pick(fn):
        matches = [k for k in seen_counts if fn(k)]
        matches.sort(key=lambda x: (-seen_counts[x], len(x)))  # prefer most common / shorter key
        return matches[0] if matches else None

    pass_key = pick(lambda k: "pass" in k and ("yard" in k or "yd" in k))
    rush_key = pick(lambda k: "rush" in k and ("yard" in k or "yd" in k))
    recv_key = pick(lambda k: ("receiv" in k or "recept" in k or "rec_" in k) and ("yard" in k or "yd" in k))

    chosen = [k for k in [pass_key, rush_key, recv_key] if k]
    discovered_map: Dict[str, str] = {}
    if pass_key: discovered_map[pass_key] = "passing_yards"
    if rush_key: discovered_map[rush_key] = "rushing_yards"
    if recv_key: discovered_map[recv_key] = "receiving_yards"

    print("• Discovered market keys (top frequency):")
    for k, c in sorted(seen_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  - {k}  (seen {c}x)")

    print(f"• Selected for yards: pass={pass_key}, rush={rush_key}, recv={recv_key}")
    return chosen, discovered_map


def fetch_event_odds(api_key: str, event_id: str, markets: List[str], bookmakers: List[str]) -> Optional[Dict]:
    """
    Fetch player prop markets for a single event using the per-event endpoint.
    Returns the event object (same shape as /odds items) or None if empty.
    """
    base = API_URLS["odds_api"].rstrip("/")
    url = f"{base}/events/{event_id}/odds"

    attempts: List[Dict] = []
    base_params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "regions": "us",
    }
    if bookmakers:
        attempts.append({**base_params, "bookmakers": ",".join(bookmakers)})
    attempts.append(base_params)  # no bookmaker filter
    attempts.append({**base_params, "regions": "us,us2"})  # widen regions

    for params in attempts:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data or not isinstance(data, dict):
                continue
            bms = data.get("bookmakers", [])
            if not bms:
                continue
            has_any_market = any(mk.get("key") in markets for bm in bms for mk in bm.get("markets", []))
            if not has_any_market:
                continue
            return data
        except requests.HTTPError:
            continue
        except Exception:
            continue
    return None


def fetch_props(api_key: str, markets: List[str], bookmakers: List[str], hours_ahead: int = 72) -> List[Dict]:
    """
    Player props are fetched via per-event endpoint.
    If markets == ['auto'], we discover the correct keys first.
    """
    events = fetch_events(api_key, hours_ahead)
    if not events:
        return []

    # Auto-discover if requested
    discovered_map: Dict[str, str] = {}
    if len(markets) == 1 and markets[0].lower() == "auto":
        selected, discovered_map = discover_prop_markets(
            api_key, events, want=("passing_yards","rushing_yards","receiving_yards")
        )
        if not selected:
            print("• No suitable player-yard markets discovered.")
            return []
        markets = selected
        MARKET_MAP.update(discovered_map)
    else:
        # Still try to augment MARKET_MAP if requested keys differ on this slate
        selected, discovered_map = discover_prop_markets(
            api_key, events, want=("passing_yards","rushing_yards","receiving_yards")
        )
        MARKET_MAP.update(discovered_map)  # harmless if keys already present

    event_ids = [ev.get("id") for ev in events if isinstance(ev, dict) and ev.get("id")]
    aggregated: List[Dict] = []
    for ev_id in event_ids:
        ev = fetch_event_odds(api_key, ev_id, markets, bookmakers)
        if ev:
            aggregated.append(ev)

    print(f"• Aggregated events with requested markets: {len(aggregated)}")
    return aggregated

def to_rows(events: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for ev in events:
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                market_key = mk.get("key")
                prop_type = MARKET_MAP.get(market_key)
                if not prop_type:
                    continue
                for out in mk.get("outcomes", []):
                    player = out.get("participant") or out.get("name") or out.get("description") or ""
                    player = _norm_name(player)

                    posteam = (
                        out.get("team") or out.get("abbreviation") or
                        out.get("participant_team") or ""
                    )
                    posteam = _team_to_abbr(posteam)

                    line = out.get("point") or out.get("line") or out.get("value")
                    price = out.get("price") or out.get("odds")
                    side  = (out.get("name") or out.get("description") or "").lower()
                    if line is None or price is None:
                        continue

                    base_row = {
                        "player": player,
                        "posteam": posteam,
                        "prop_type": prop_type,
                        "line": float(line),
                        "over_odds": None,
                        "under_odds": None,
                    }
                    if "over" in side:
                        base_row["over_odds"] = price
                    elif "under" in side:
                        base_row["under_odds"] = price
                    else:
                        continue
                    rows.append(base_row)
    return rows

def collapse_sides(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["player","posteam","prop_type","line","over_odds","under_odds"])
    df = pd.DataFrame(rows)
    # Combine over/under for the same (player, team, prop, line)
    keys = ["player","posteam","prop_type","line"]
    agg = df.groupby(keys, as_index=False).agg(
        over_odds=("over_odds", "max"),
        under_odds=("under_odds", "max"),
    )
    # drop rows missing either side
    agg = agg.dropna(subset=["over_odds","under_odds"])
    # tidy types
    agg["player"] = agg["player"].map(_norm_name)
    agg["posteam"] = agg["posteam"].astype(str).str.upper().str.strip()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="auto",
                    help="Comma list of market keys or 'auto' to discover (recommended).")
    ap.add_argument("--bookmakers", default="draftkings,fanduel,betmgm")
    ap.add_argument("--hours-ahead", type=int, default=72)
    ap.add_argument("--out", default=f"{PROCESSED_DATA_DIR}/player_props_{_today_tag()}.csv")
    args = ap.parse_args()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("Set ODDS_API_KEY in your environment or .env")

    if args.markets.strip().lower() == "auto":
        markets = ["auto"]
    else:
        # Normalize CLI market names (map legacy aliases, de-dupe)
        ALIASES = {"player_rec_yds": "player_reception_yds"}
        markets = [ALIASES.get(m.strip(), m.strip()) for m in args.markets.split(",") if m.strip()]
        markets = sorted(set(markets))
    bookmakers = [b.strip() for b in args.bookmakers.split(",") if b.strip()]

    events = fetch_props(api_key, markets, bookmakers, hours_ahead=args.hours_ahead)
    rows = to_rows(events)
    print(f"• Raw rows parsed: {len(rows)}")
    df = collapse_sides(rows)
    print(f"• Rows after collapsing O/U: {len(df)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # simple summary
    by_prop = df.groupby("prop_type").size().to_dict() if not df.empty else {}
    print("Wrote", out_path, "rows:", len(df), "by_prop:", by_prop)

if __name__ == "__main__":
    main()


