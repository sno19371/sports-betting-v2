from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# Default configuration for bet evaluation
DEFAULT_CFG = {
    "edge_threshold": 0.02,
    "min_probability": 0.05,
    "kelly_fraction": 0.25,
    "max_bet_size": 0.05,
    "slate_kelly_cap": 0.25,
    "prefer_side": "auto",
    "books_priority": ["pinnacle", "circa", "betmgm", "draftkings", "fanduel"],
    "prop_aliases": {},
}


def _clean_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_prop_key(k: str, aliases: dict) -> str:
    k = str(k).strip().lower()
    return aliases.get(k, k)


def _american_to_decimal(ao: int) -> float:
    if ao == 0:
        raise ValueError("American odds cannot be 0")
    return 1 + (100 / abs(ao)) if ao < 0 else 1 + (ao / 100)


def _implied_prob_from_decimal(d: float) -> float:
    if d <= 1:
        raise ValueError("Decimal odds must be > 1")
    return 1.0 / d


def _no_vig_two_way(odds_over: int, odds_under: int) -> tuple[float, float, float]:
    d_o = _american_to_decimal(odds_over)
    d_u = _american_to_decimal(odds_under)
    p_o = _implied_prob_from_decimal(d_o)
    p_u = _implied_prob_from_decimal(d_u)
    total = p_o + p_u
    overround = total - 1.0
    # Power method (binary search for exponent t)
    lo, hi = 0.1, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = (p_o ** mid) + (p_u ** mid)
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    po_nv = (p_o ** t)
    pu_nv = (p_u ** t)
    denom = po_nv + pu_nv
    if denom <= 0:
        # Fallback to simple normalization
        return (p_o / total if total > 0 else 0.5), (p_u / total if total > 0 else 0.5), overround
    return po_nv / denom, pu_nv / denom, overround


def _prob_over_from_quantiles(
    line: float,
    q_levels: list[float],
    q_values: list[float],
    *,
    min_tail_prob: float = 0.01,
) -> Optional[float]:
    if not q_levels or not q_values or len(q_levels) != len(q_values):
        return None

    # Sort by quantile level
    pairs = sorted(zip(q_levels, q_values), key=lambda x: x[0])
    qs = np.array([q for q, _ in pairs], dtype=float)
    xs = np.array([x for _, x in pairs], dtype=float)

    # Enforce monotonicity in predicted quantiles (non-decreasing)
    xs = np.maximum.accumulate(xs)

    # Strict tails: treat equality as interior/plateau, not tail
    if line < xs[0]:
        return float(max(1.0 - qs[0], min_tail_prob))
    if line > xs[-1]:
        return float(min(1.0 - qs[-1], 1.0 - min_tail_prob))

    # Plateau detection: indices spanning all entries equal to 'line'
    left = int(np.searchsorted(xs, line, side="left"))
    right = int(np.searchsorted(xs, line, side="right")) - 1

    if left <= right:
        # Line sits on a flat segment of the predicted quantiles.
        # Convention: be indifferent at that mass point → P(Over) ≈ 0.5
        p_over = 0.5
        return float(np.clip(p_over, min_tail_prob, 1.0 - min_tail_prob))

    # Otherwise interpolate between the nearest bracket (left-1, left)
    idx = left - 1  # since xs[idx] < line < xs[idx+1]
    x0, x1 = xs[idx], xs[idx + 1]
    q0, q1 = qs[idx], qs[idx + 1]

    if x1 == x0:
        # Extra safety: degenerate local bracket — average the quantiles
        q_star = 0.5 * (q0 + q1)
    else:
        w = (line - x0) / (x1 - x0)
        q_star = q0 + w * (q1 - q0)

    p_over = 1.0 - float(q_star)
    return float(np.clip(p_over, min_tail_prob, 1.0 - min_tail_prob))


def _hash_bet_id(player: str, prop: str, side: str, line: float, book: str, asof: str) -> str:
    s = f"{player}|{prop}|{side}|{line}|{book}|{asof}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class BetFinder:
    def __init__(self, model_predictions: pd.DataFrame, market_odds: list[dict], *, config: dict | None = None) -> None:
        self.model_predictions = model_predictions.copy()
        self.market_odds = list(market_odds or [])
        self.cfg = {**DEFAULT_CFG, **(config or {})}
        # Determine player name column
        self.player_col = (
            "player_display_name" if "player_display_name" in self.model_predictions.columns else "player_name"
        )
        # Index predictions by (clean_name, prop)
        self._pred_index: dict[tuple[str, str], pd.Series] = {}
        self._prop_quantiles: dict[str, list[tuple[float, str]]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        aliases = self.cfg["prop_aliases"]
        # discover quantile columns per prop
        for col in self.model_predictions.columns:
            if "_q" in col:
                base, _, qtxt = col.rpartition("_q")
                try:
                    q = float(qtxt)
                except ValueError:
                    continue
                prop = _normalize_prop_key(base, aliases)
                self._prop_quantiles.setdefault(prop, []).append((q, col))
        # sort quantile columns by q
        for prop, lst in self._prop_quantiles.items():
            self._prop_quantiles[prop] = sorted(lst, key=lambda t: t[0])
        # row index by player
        for _, row in self.model_predictions.iterrows():
            pname = _clean_name(row.get(self.player_col, ""))
            for prop in self._prop_quantiles.keys():
                self._pred_index[(pname, prop)] = row

    def _evaluate_market(self, m: dict) -> Optional[dict]:
        aliases = self.cfg["prop_aliases"]
        min_prob = self.cfg["min_probability"]
        edge_thr = self.cfg["edge_threshold"]
        kelly_frac = self.cfg["kelly_fraction"]
        max_bet = self.cfg["max_bet_size"]
        prefer = self.cfg["prefer_side"]

        player = _clean_name(m.get("player_name", ""))
        prop = _normalize_prop_key(m.get("prop_type", ""), aliases)
        line = float(m.get("line"))
        book = m.get("bookmaker", "")

        row = self._pred_index.get((player, prop))
        qspec = self._prop_quantiles.get(prop)
        if row is None or not qspec:
            return None
        # Collect only non-NaN quantiles; require at least 2 points for interpolation
        valid_pairs: list[tuple[float, float]] = []
        for q, col in qspec:
            val = row.get(col)
            if pd.notna(val):
                try:
                    valid_pairs.append((q, float(val)))
                except Exception:
                    continue
        if len(valid_pairs) < 2:
            return None
        # Unzip
        q_levels = [q for q, _ in valid_pairs]
        q_values = [v for _, v in valid_pairs]

        p_over_model = _prob_over_from_quantiles(line, q_levels, q_values, min_tail_prob=min_prob)
        if p_over_model is None:
            return None

        ao_over = int(m.get("over_odds"))
        ao_under = int(m.get("under_odds"))
        d_over = _american_to_decimal(ao_over)
        d_under = _american_to_decimal(ao_under)
        p_over_nv, p_under_nv, overround = _no_vig_two_way(ao_over, ao_under)

        edge_over = p_over_model - p_over_nv
        p_under_model = 1.0 - p_over_model
        edge_under = p_under_model - p_under_nv

        if prefer == "over":
            side, model_p, dec_odds, ao, market_nv = "Over", p_over_model, d_over, ao_over, p_over_nv
        elif prefer == "under":
            side, model_p, dec_odds, ao, market_nv = "Under", p_under_model, d_under, ao_under, p_under_nv
        else:
            if edge_over >= edge_under:
                side, model_p, dec_odds, ao, market_nv = "Over", p_over_model, d_over, ao_over, p_over_nv
            else:
                side, model_p, dec_odds, ao, market_nv = "Under", p_under_model, d_under, ao_under, p_under_nv

        edge = model_p - market_nv
        if (edge < edge_thr) or (model_p <= min_prob) or (model_p >= 1 - min_prob):
            return None

        b = dec_odds - 1.0
        full_kelly = (model_p * b - (1 - model_p)) / b
        frac_kelly = max(0.0, full_kelly * kelly_frac)
        kelly_capped = min(frac_kelly, max_bet)

        ev = model_p * (dec_odds - 1) - (1 - model_p)

        asof = datetime.now(timezone.utc).isoformat()
        bet = {
            "bet_id": _hash_bet_id(player, prop, side, line, book, asof),
            "player_name": m.get("player_name"),
            "prop_type": prop,
            "side": side,
            "line": line,
            "american_odds": ao,
            "decimal_odds": round(dec_odds, 3),
            "bookmaker": book,
            "model_probability": round(model_p, 4),
            "market_implied_prob": round(_implied_prob_from_decimal(dec_odds), 4),
            "no_vig_implied_prob": round(market_nv, 4),
            "edge": round(edge, 4),
            "kelly_fraction": round(kelly_capped, 4),
            "recommended_stake_pct": f"{kelly_capped:.2%}",
            "expected_value": round(ev, 4),
            "quantiles_used": q_levels,
            "overround": round(overround, 4),
            "asof": asof,
            "notes": "no-vig power; monotone quantile CDF",
        }
        return bet

    def find_bets(self) -> list[dict]:
        markets = []
        for m in self.market_odds:
            if "over_odds" in m and "under_odds" in m and m.get("line") is not None:
                markets.append(
                    {
                        **m,
                        "player_name": m["player_name"],
                        "prop_type": _normalize_prop_key(m["prop_type"], self.cfg["prop_aliases"]),
                        "bookmaker": m.get("bookmaker", ""),
                    }
                )

        bets = []
        for m in markets:
            bet = self._evaluate_market(m)
            if bet:
                bets.append(bet)

        total_kelly = sum(b.get("kelly_fraction", 0.0) for b in bets)
        cap = self.cfg["slate_kelly_cap"]
        if total_kelly > cap and total_kelly > 0:
            scale = cap / total_kelly
            for b in bets:
                b["kelly_fraction"] = round(b["kelly_fraction"] * scale, 4)
                b["recommended_stake_pct"] = f"{b['kelly_fraction']:.2%}"

        bets.sort(key=lambda x: (x["edge"], x["expected_value"]), reverse=True)
        return bets

    def save_log(self, path: str | None = None) -> str:
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"logs/bets_{ts}.jsonl"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for b in self.find_bets():
                f.write(json.dumps(b) + "\n")
        return path

