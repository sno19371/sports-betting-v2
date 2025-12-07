#!/usr/bin/env python3
"""
generate_predictions.py

Generates rush attempts predictions using the trained model and combines
with betting lines from sportsbooks.

Inputs:
  - player_model_rows_curr.json: Player data with historical TCN features
  - artifacts_rb_rush_volume_v10_curr/: Trained model artifacts
  - odds_history/: Current betting lines

Output:
  - predictions_curr.json: Predictions with betting line comparisons

Usage:
  python generate_predictions.py
"""

import json
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import lightgbm as lgb

# =========================
# CONFIGURATION
# =========================
SCRIPT_DIR = Path(__file__).parent
ARTIFACTS_DIR = SCRIPT_DIR.parent / "artifacts_rb_rush_volume_v10_curr"

# Input files
PLAYER_ROWS_PATH = SCRIPT_DIR / "player_model_rows_curr.json"
ODDS_HISTORY_DIR = SCRIPT_DIR / "odds_history"

# Output
OUTPUT_PATH = SCRIPT_DIR / "predictions_curr.json"

# Model parameters (must match training)
EMBED_DIM = 64
L = 6  # Window size

# TCN Features (from feature_meta.json)
TCN_COLS = [
    "rb_carries", "rb_ed_carry_share_all", "rb_rz_carries",
    "rb_rush_yards", "rb_stuffed_rate",
    "starter_qb_scramble_rate_db",
    "implied_game_script", "team_spread", "is_home"
]

# Context features for LightGBM
CTX_COLS = [
    "posteam", "opponent",
    "implied_game_script", "team_spread", "is_home", "wind_effective",
    "opponent_rush_epa_allowed", "opponent_rush_yards_per_game_allowed",
    "depth_rank", "is_projected_starter", "is_active",
    "last_week_carries", "last_week_snap_share"
]

# Team name mapping
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}


# =========================
# TCN MODEL (must match training)
# =========================
class RushTCN(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 64, 3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, embed_dim, 1),
            nn.AdaptiveMaxPool1d(1)
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        pred = self.head(z).squeeze(-1)
        return pred, z


def normalize_name(name):
    """Normalize player name for matching."""
    if not name:
        return ""
    name = str(name).strip().lower()
    for suffix in [" jr.", " sr.", " iii", " ii", " iv", " v"]:
        name = name.replace(suffix, "")
    return name.replace("'", "").replace("-", " ").replace(".", " ").strip()


def load_models():
    """Load all model artifacts."""
    print("📦 Loading model artifacts...")
    
    # Load scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"   ✓ Loaded scaler")
    
    # Load TCN
    tcn_path = ARTIFACTS_DIR / "tcn_rush_volume.pt"
    tcn = RushTCN(len(TCN_COLS), EMBED_DIM)
    tcn.load_state_dict(torch.load(tcn_path, map_location="cpu"))
    tcn.eval()
    print(f"   ✓ Loaded TCN")
    
    # Load LightGBM
    lgb_path = ARTIFACTS_DIR / "lgb_rush_volume.txt"
    gbm = lgb.Booster(model_file=str(lgb_path))
    print(f"   ✓ Loaded LightGBM")
    
    return scaler, tcn, gbm


def load_player_data():
    """Load player model rows."""
    print(f"📂 Loading player data from: {PLAYER_ROWS_PATH}")
    
    with open(PLAYER_ROWS_PATH, "r") as f:
        data = json.load(f)
    
    players = data.get("players", [])
    print(f"   ✓ Loaded {len(players)} players")
    
    return data, players


def load_betting_lines():
    """Load betting lines from odds_history."""
    print(f"💰 Loading betting lines from: {ODDS_HISTORY_DIR}")
    
    lines = {}  # {normalized_name: {bookmaker: {line, over_price, under_price}}}
    
    history_files = glob.glob(str(ODDS_HISTORY_DIR / "history_*.json"))
    
    for filepath in history_files:
        with open(filepath, "r") as f:
            history = json.load(f)
        
        if not history:
            continue
        
        latest = history[-1] if isinstance(history, list) else history
        
        bookmakers = latest.get("bookmakers", [])
        for book in bookmakers:
            book_name = book.get("title", book.get("key", "Unknown"))
            markets = book.get("markets", [])
            
            for market in markets:
                if market.get("key") == "player_rush_attempts":
                    outcomes = market.get("outcomes", [])
                    
                    # Group outcomes by player
                    player_outcomes = {}
                    for outcome in outcomes:
                        player_name = outcome.get("description")
                        if not player_name:
                            continue
                        
                        name_key = normalize_name(player_name)
                        if name_key not in player_outcomes:
                            player_outcomes[name_key] = {"player_name": player_name}
                        
                        direction = outcome.get("name", "").lower()
                        if direction == "over":
                            player_outcomes[name_key]["line"] = outcome.get("point")
                            player_outcomes[name_key]["over_price"] = outcome.get("price")
                        elif direction == "under":
                            player_outcomes[name_key]["under_price"] = outcome.get("price")
                    
                    # Store in lines dict
                    for name_key, player_line in player_outcomes.items():
                        if name_key not in lines:
                            lines[name_key] = {}
                        lines[name_key][book_name] = player_line
    
    print(f"   ✓ Loaded lines for {len(lines)} players from {len(history_files)} games")
    return lines


def build_tcn_input(player, scaler):
    """Build TCN input tensor from player's historical data."""
    tcn_history = player.get("tcn_history", [])
    
    # Build feature matrix (L x C)
    feature_matrix = []
    for week_data in tcn_history:
        week_features = [float(week_data.get(col, 0.0)) for col in TCN_COLS]
        feature_matrix.append(week_features)
    
    # Pad if needed
    while len(feature_matrix) < L:
        feature_matrix.insert(0, [0.0] * len(TCN_COLS))
    
    # Convert to numpy (L, C)
    X = np.array(feature_matrix, dtype=np.float32)
    
    # Scale
    X_flat = X.reshape(-1, len(TCN_COLS))  # (L, C)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(1, L, len(TCN_COLS))  # (1, L, C)
    
    # Transpose to (1, C, L) for Conv1d
    X_scaled = X_scaled.transpose(0, 2, 1)
    
    return torch.from_numpy(X_scaled).float()


def build_lgb_features(player, embedding):
    """Build LightGBM feature DataFrame."""
    # Embedding columns
    embed_cols = [f"E{i}" for i in range(EMBED_DIM)]
    embed_df = pd.DataFrame([embedding], columns=embed_cols)
    
    # Context features
    ctx_data = {
        "posteam": player.get("posteam", "UNK"),
        "opponent": player.get("opponent", "UNK"),
        "implied_game_script": float(player.get("implied_game_script", 0)),
        "team_spread": float(player.get("team_spread", 0)),
        "is_home": int(player.get("is_home", 0)),
        "wind_effective": float(player.get("wind_effective", 0)),
        "opponent_rush_epa_allowed": float(player.get("opponent_rush_epa_allowed", 0)),
        "opponent_rush_yards_per_game_allowed": float(player.get("opponent_rush_yards_per_game_allowed", 100)),
        "depth_rank": int(player.get("depth_rank", 4)),
        "is_projected_starter": int(player.get("is_projected_starter", 0)),
        "is_active": int(player.get("is_active", 1)),
        "last_week_carries": float(player.get("last_week_carries", 0)),
        "last_week_snap_share": float(player.get("last_week_snap_share", 0)),
    }
    ctx_df = pd.DataFrame([ctx_data])
    
    # Convert categorical
    ctx_df["posteam"] = ctx_df["posteam"].astype("category")
    ctx_df["opponent"] = ctx_df["opponent"].astype("category")
    
    # Combine
    X_lgb = pd.concat([embed_df, ctx_df], axis=1)
    
    return X_lgb


def get_best_lines(player_name, betting_lines):
    """Get best available lines for a player."""
    name_key = normalize_name(player_name)
    
    if name_key not in betting_lines:
        return None
    
    player_lines = betting_lines[name_key]
    
    # Find best over (highest price = best value for over)
    # Find best under (highest price = best value for under)
    best_over = {"book": None, "line": None, "price": -999}
    best_under = {"book": None, "line": None, "price": -999}
    all_lines = []
    
    for book, line_data in player_lines.items():
        line = line_data.get("line")
        over_price = line_data.get("over_price")
        under_price = line_data.get("under_price")
        
        all_lines.append({
            "book": book,
            "line": line,
            "over_price": over_price,
            "under_price": under_price
        })
        
        if over_price and over_price > best_over["price"]:
            best_over = {"book": book, "line": line, "price": over_price}
        
        if under_price and under_price > best_under["price"]:
            best_under = {"book": book, "line": line, "price": under_price}
    
    return {
        "all_lines": all_lines,
        "best_over": best_over if best_over["book"] else None,
        "best_under": best_under if best_under["book"] else None
    }


def main():
    print("=" * 60)
    print("GENERATE PREDICTIONS")
    print("=" * 60)
    
    # 1. Load models
    scaler, tcn, gbm = load_models()
    
    # 2. Load player data
    data, players = load_player_data()
    
    if not players:
        print("❌ No players to predict!")
        return
    
    # 3. Load betting lines
    betting_lines = load_betting_lines()
    
    # 4. Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = []
    
    for player in players:
        player_name = player.get("player_name", "Unknown")
        
        # Build TCN input and get embedding
        X_tcn = build_tcn_input(player, scaler)
        
        with torch.no_grad():
            _, embedding = tcn(X_tcn)
            embedding = embedding.numpy()[0]  # (EMBED_DIM,)
        
        # Build LightGBM features
        X_lgb = build_lgb_features(player, embedding)
        
        # Predict
        pred_carries = float(gbm.predict(X_lgb)[0])
        
        # Get betting lines
        lines_data = get_best_lines(player_name, betting_lines)
        
        # Determine edge
        rush_line = player.get("rush_attempts_line")
        edge = None
        recommendation = None
        
        if rush_line:
            diff = pred_carries - rush_line
            edge = round(diff, 2)
            
            if diff > 1.5:
                recommendation = "OVER"
            elif diff < -1.5:
                recommendation = "UNDER"
            else:
                recommendation = "NO EDGE"
        
        # Get recommended book based on direction
        recommended_book = None
        recommended_line = None
        recommended_price = None
        
        if lines_data and recommendation:
            if recommendation == "OVER" and lines_data.get("best_over"):
                best = lines_data["best_over"]
                recommended_book = best["book"]
                recommended_line = best["line"]
                recommended_price = best["price"]
            elif recommendation == "UNDER" and lines_data.get("best_under"):
                best = lines_data["best_under"]
                recommended_book = best["book"]
                recommended_line = best["line"]
                recommended_price = best["price"]
        
        # Build prediction record
        pred_record = {
            "player_name": player_name,
            "team": player.get("posteam"),
            "opponent": player.get("opponent"),
            "is_home": player.get("is_home"),
            "matchup": f"{player.get('posteam', '?')} {'vs' if player.get('is_home') else '@'} {player.get('opponent', '?')}",
            
            # Prediction
            "predicted_carries": round(pred_carries, 1),
            "rush_attempts_line": rush_line,
            "edge": edge,
            "recommendation": recommendation,
            
            # Recommended sportsbook
            "recommended_book": recommended_book,
            "recommended_line": recommended_line,
            "recommended_price": recommended_price,
            
            # Context
            "team_spread": player.get("team_spread"),
            "depth_rank": player.get("depth_rank"),
            "is_projected_starter": player.get("is_projected_starter"),
            "is_active": player.get("is_active"),
            "last_week_carries": player.get("last_week_carries"),
            "history_weeks": player.get("history_weeks"),
            
            # All betting lines
            "all_betting_lines": lines_data.get("all_lines") if lines_data else None
        }
        
        predictions.append(pred_record)
        
        # Log
        edge_str = f"{edge:+.1f}" if edge else "N/A"
        rec_emoji = "🟢" if recommendation == "OVER" else "🔴" if recommendation == "UNDER" else "⚪"
        book_str = f" @ {recommended_book}" if recommended_book else ""
        print(f"   {rec_emoji} {player_name}: Pred={pred_carries:.1f}, Line={rush_line}, Edge={edge_str}{book_str}")
    
    # Sort by absolute edge
    predictions.sort(key=lambda x: abs(x.get("edge", 0) or 0), reverse=True)
    
    # 5. Save output
    print(f"\n💾 Saving to: {OUTPUT_PATH}")
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "prediction_season": data.get("prediction_season"),
        "prediction_week": data.get("prediction_week"),
        "model_mae": 3.01,  # From training log
        "total_players": len(predictions),
        "predictions": predictions
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    overs = [p for p in predictions if p["recommendation"] == "OVER"]
    unders = [p for p in predictions if p["recommendation"] == "UNDER"]
    no_edge = [p for p in predictions if p["recommendation"] == "NO EDGE"]
    
    print(f"Total predictions: {len(predictions)}")
    print(f"🟢 OVER recommendations: {len(overs)}")
    print(f"🔴 UNDER recommendations: {len(unders)}")
    print(f"⚪ No edge: {len(no_edge)}")
    
    # Top picks
    if overs or unders:
        print("\n--- TOP PICKS (by edge) ---")
        top_picks = [p for p in predictions if p["recommendation"] in ["OVER", "UNDER"]][:10]
        for p in top_picks:
            emoji = "🟢" if p["recommendation"] == "OVER" else "🔴"
            book_info = ""
            if p.get("recommended_book"):
                price_str = f"({p['recommended_price']:+d})" if p.get("recommended_price") else ""
                book_info = f" → {p['recommended_book']}: {p['recommended_line']} {price_str}"
            
            print(f"{emoji} {p['player_name']:<20} | Pred: {p['predicted_carries']:5.1f} | Line: {p['rush_attempts_line']:5.1f} | Edge: {p['edge']:+5.1f} | {p['recommendation']}{book_info}")
    
    print(f"\n✅ Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

