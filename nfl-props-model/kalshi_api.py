# kalshi_api.py
"""
Kalshi API integration for NFL prop betting.
Handles both REST API and WebSocket connections for real-time market data.
"""

import asyncio
import websockets
import requests
import json
import time
import base64
import hashlib
import threading
import queue
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from config import (
    KALSHI_API_KEY, KALSHI_PRIVATE_KEY, API_URLS,
    API_RATE_LIMITS, KALSHI_CONFIG, DATA_FILES, CURRENT_SEASON
)
from utils import (
    convert_american_to_decimal, implied_probability,
    calculate_edge, remove_vig
)

logger = logging.getLogger(__name__)


class KalshiAPIClient:
    """REST API client for Kalshi prediction markets."""
    
    def __init__(self):
        self.api_key = KALSHI_API_KEY
        self.base_url = API_URLS['kalshi_api']
        
        # Load private key for authentication
        if KALSHI_PRIVATE_KEY:
            self.private_key = serialization.load_pem_private_key(
                KALSHI_PRIVATE_KEY.encode(), 
                password=None
            )
        else:
            self.private_key = None
            logger.warning("Kalshi private key not loaded")
        
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _generate_signature(self, method: str, path: str, timestamp: str, body: str = "") -> str:
        """Generate RSA-PSS signature for API authentication."""
        if not self.private_key:
            raise ValueError("Private key not available")
        
        # Construct the message to sign
        message = f"{timestamp}{method}{path}{body}"
        
        # Sign with RSA-PSS SHA-256
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication headers for API requests."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(method, path, timestamp, body)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        min_interval = API_RATE_LIMITS.get('kalshi', 0.5)
        elapsed = time.time() - self.last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, path: str, params: Dict = None, body: Dict = None) -> Optional[Dict]:
        """Make authenticated API request to Kalshi."""
        self._rate_limit()
        
        url = f"{self.base_url}{path}"
        body_str = json.dumps(body) if body else ""
        headers = self._get_headers(method, path, body_str)
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method == "POST":
                response = self.session.post(url, headers=headers, data=body_str, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Log the response for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Kalshi API error: {response.status_code}"
                try:
                    error_body = response.json()
                    error_msg += f" - {error_body}"
                except:
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                print(f"API Error: {error_msg}")  # Print for immediate debugging
                return None
                
        except Exception as e:
            logger.error(f"Kalshi API request failed: {e}")
            print(f"Request failed: {e}")  # Print for immediate debugging
            print(f"URL attempted: {url}")
            return None
    
    # =============================================================================
    # MARKET DATA METHODS
    # =============================================================================
    
    def get_nfl_markets(self, week: int = None) -> Optional[List[Dict]]:
        """Get all available NFL markets by filtering ticker prefixes.

        Kalshi markets commonly use ticker prefixes like:
        - KXMVENFLSINGLEGAME-...
        - KXMVENFLMULTIGAME-...
        - KXNFLMENTION-...
        """
        params = {"limit": 500}
        response = self._make_request("GET", "/trade-api/v2/markets", params=params)
        
        if response and "markets" in response:
            all_markets = response["markets"]
            prefixes = ("KXMVENFLSINGLEGAME", "KXMVENFLMULTIGAME", "KXNFLMENTION", "KXMVENFL")
            def is_nfl(m: Dict) -> bool:
                t = (m.get("ticker") or "")
                return any(t.startswith(p) for p in prefixes)
            markets = [m for m in all_markets if is_nfl(m)]
            logger.info(f"Found {len(markets)} NFL markets (from {len(all_markets)} total)")
            return markets
        
        return None
    
    def get_player_prop_markets(self, player_name: str = None, prop_type: str = None) -> Optional[List[Dict]]:
        """Get player prop markets, optionally filtered by player or prop type.

        Uses current NFL rosters to identify player names in market titles
        and filters to touchdown-related markets by default.
        """
        all_markets = self.get_nfl_markets()
        
        if not all_markets:
            return None
        
        # Build player name set from rosters for robust matching
        try:
            import nfl_data_py as nfl  # Lazy import
            rosters = nfl.import_rosters(years=[CURRENT_SEASON])
            player_names = set(n.lower() for n in rosters['player_name'].dropna().unique())
        except Exception:
            # Fallback to a small seed set if roster fetch fails
            player_names = set(n.lower() for n in [
                "Tyreek Hill", "Travis Kelce", "Josh Allen", "Stefon Diggs",
                "Justin Jefferson", "Ja'Marr Chase", "Patrick Mahomes",
            ])

        # Filter for player props
        prop_markets = []
        
        for market in all_markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            title_lower = title.lower()
            
            # Check if it's a player prop market
            is_td_keyword = any(kw in title_lower for kw in ["touchdown", " any", " first td", " td", "2+ td"])
            has_player_name = any(p in title_lower for p in player_names)

            is_player_prop = is_td_keyword and has_player_name
            
            if not is_player_prop:
                continue
            
            # Filter by player name if provided
            if player_name and player_name.lower() not in title_lower:
                continue
            
            # Filter by prop type if provided
            if prop_type:
                prop_keywords = {
                    'receiving': ['receiving', 'reception', 'catch'],
                    'rushing': ['rushing', 'carries', 'rush'],
                    'passing': ['passing', 'completion', 'throw', 'yards'],
                    'touchdown': ['touchdown', 'td', 'anytime', 'first']
                }
                
                if prop_type in prop_keywords:
                    if not any(kw in title_lower for kw in prop_keywords[prop_type]):
                        continue
            
            prop_markets.append(market)
        
        logger.info(f"Found {len(prop_markets)} player prop markets")
        return prop_markets
    
    def get_market_orderbook(self, market_ticker: str) -> Optional[Dict]:
        """Get orderbook (bid/ask) for a specific market."""
        path = f"/trade-api/v2/markets/{market_ticker}/orderbook"
        response = self._make_request("GET", path)
        
        if response and "orderbook" in response:
            return response["orderbook"]
        
        return None
    
    def get_touchdown_markets(self, game_filter: str = None) -> Optional[List[Dict]]:
        """Get touchdown prop markets for NFL games.
        
        Args:
            game_filter: Optional filter for specific game (e.g., "NYJ", "MIA")
        
        Returns:
            List of touchdown markets with enriched data
        """
        all_markets = self.get_nfl_markets()
        
        if not all_markets:
            return None
        
        touchdown_markets = []
        
        for market in all_markets:
            title = market.get("title", "")
            ticker = market.get("ticker", "")
            
            # Apply game filter if provided
            if game_filter and game_filter.upper() not in ticker and game_filter not in title:
                continue
            
            # Kalshi touchdown markets often just have player names as titles
            # or include "Anytime TD", "First TD", "2+ TDs"
            # Based on the screenshot, player names alone indicate touchdown markets
            nfl_touchdown_players = [
                "De'Von Achane", "Tyreek Hill", "Breece Hall", "Garrett Wilson",
                "Jaylen Waddle", "Justin Fields", "Braelon Allen", "Ollie Gordon",
                "Raheem Mostert", "Tyler Conklin", "Allen Lazard", "Xavier Gipson",
                "Travis Kelce", "Patrick Mahomes", "Josh Allen", "Stefon Diggs",
                "Justin Jefferson", "Ja'Marr Chase", "Cooper Kupp", "Davante Adams"
            ]
            
            is_touchdown_market = False
            matched_player = None
            
            # Check if title matches a player name exactly or contains it
            for player in nfl_touchdown_players:
                if player in title or title == player:
                    is_touchdown_market = True
                    matched_player = player
                    break
            
            # Also check for explicit touchdown keywords
            if not is_touchdown_market and any(kw in title.lower() for kw in ["touchdown", " td", "anytime", "first td", "2+ td"]):
                is_touchdown_market = True
            
            if is_touchdown_market:
                # Get current prices
                prices = self.get_market_prices(ticker)
                
                market_data = {
                    "ticker": ticker,
                    "title": title,
                    "player": matched_player or title,
                    "market_type": "touchdown",
                    "prices": prices,
                    "expiry": market.get("expiry_time"),
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest")
                }
                
                touchdown_markets.append(market_data)
        
        logger.info(f"Found {len(touchdown_markets)} touchdown markets")
        return touchdown_markets
    
    # =============================================================================
    # MARKET ANALYSIS METHODS
    # =============================================================================
    
    def get_market_prices(self, market_ticker: str) -> Optional[Dict]:
        """Get current best bid/ask prices for a market."""
        orderbook = self.get_market_orderbook(market_ticker)
        
        if not orderbook:
            return None
        
        yes_orders = orderbook.get("yes", [])
        no_orders = orderbook.get("no", [])
        
        # Extract best prices (Kalshi uses cents, convert to dollars)
        yes_bid = max([o["price"] for o in yes_orders if o.get("price")], default=0) / 100
        yes_ask = min([o["price"] for o in yes_orders if o.get("price")], default=0) / 100
        no_bid = max([o["price"] for o in no_orders if o.get("price")], default=0) / 100
        no_ask = min([o["price"] for o in no_orders if o.get("price")], default=0) / 100
        
        result = {
            "ticker": market_ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "spread": yes_ask - yes_bid if yes_ask and yes_bid else None,
            "mid_price": (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else None,
            "timestamp": datetime.now()
        }
        
        return result
    
    def analyze_market_opportunity(self, market: Dict, model_prob: float) -> Optional[Dict]:
        """Analyze if a market presents a betting opportunity based on model probability."""
        ticker = market.get("ticker")
        prices = self.get_market_prices(ticker)
        
        if not prices or not prices.get("yes_bid"):
            return None
        
        # Calculate edges for both sides
        yes_edge = model_prob - prices["yes_ask"] if prices.get("yes_ask") else None
        no_edge = (1 - model_prob) - prices["no_ask"] if prices.get("no_ask") else None
        
        # Account for Kalshi fees
        kalshi_fee = KALSHI_CONFIG.get('kalshi_fee', 0.007)
        min_edge = KALSHI_CONFIG.get('min_kalshi_edge', 0.04)
        
        # Determine best side to bet
        best_side = None
        best_edge = None
        bet_price = None
        
        if yes_edge and yes_edge > min_edge:
            # Adjust edge for fees (paid on profits only)
            adjusted_edge = yes_edge - (kalshi_fee * model_prob)
            if adjusted_edge > min_edge:
                best_side = "yes"
                best_edge = adjusted_edge
                bet_price = prices["yes_ask"]
        
        if no_edge and no_edge > min_edge:
            # Adjust edge for fees
            adjusted_edge = no_edge - (kalshi_fee * (1 - model_prob))
            if adjusted_edge > min_edge and (not best_edge or adjusted_edge > best_edge):
                best_side = "no"
                best_edge = adjusted_edge
                bet_price = prices["no_ask"]
        
        if not best_side:
            return None
        
        # Convert Kalshi probability to decimal odds for Kelly calculation
        decimal_odds = 1 / bet_price if bet_price > 0 else None
        
        return {
            "ticker": ticker,
            "market_title": market.get("title"),
            "side": best_side,
            "model_probability": model_prob,
            "market_probability": bet_price,
            "edge": best_edge,
            "decimal_odds": decimal_odds,
            "spread": prices.get("spread"),
            "timestamp": datetime.now()
        }
    
    def get_portfolio_positions(self) -> Optional[List[Dict]]:
        """Get current portfolio positions."""
        response = self._make_request("GET", "/trade-api/v2/portfolio/positions")
        
        if response and "positions" in response:
            return response["positions"]
        
        return None
    
    def get_balance(self) -> Optional[float]:
        """Get account balance in dollars."""
        response = self._make_request("GET", "/trade-api/v2/portfolio/balance")
        
        if response and "balance" in response:
            # Convert cents to dollars
            return response["balance"] / 100
        
        return None


class KalshiWebSocketClient:
    """WebSocket client for real-time Kalshi market data."""
    
    def __init__(self, update_callback=None):
        self.api_key = KALSHI_API_KEY
        
        if KALSHI_PRIVATE_KEY:
            self.private_key = serialization.load_pem_private_key(
                KALSHI_PRIVATE_KEY.encode(), 
                password=None
            )
        else:
            self.private_key = None
            
        self.update_callback = update_callback
        self.update_queue = queue.Queue()
        self.websocket = None
        self.running = False
        self.last_prices = {}
        self.subscribed_markets = set()
        
    def _generate_signature(self, timestamp: str) -> str:
        """Generate signature for WebSocket authentication."""
        if not self.private_key:
            raise ValueError("Private key not available")
            
        message = f"{timestamp}GET/trade-api/ws/v2"
        
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers for WebSocket connection."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature
        }
    
    async def _process_market_update(self, data: Dict):
        """Process incoming market update."""
        try:
            if data.get("type") == "ticker" and "msg" in data:
                msg_data = data["msg"]
                ticker = msg_data.get("market_ticker")
                
                if ticker:
                    # Extract prices (convert from cents)
                    yes_bid = msg_data.get("yes_bid", 0) / 100
                    yes_ask = msg_data.get("yes_ask", 0) / 100
                    
                    # Check for price changes
                    current_prices = (yes_bid, yes_ask)
                    last_prices = self.last_prices.get(ticker, (None, None))
                    
                    if current_prices != last_prices:
                        self.last_prices[ticker] = current_prices
                        
                        # Create update event
                        update = {
                            "type": "price_update",
                            "ticker": ticker,
                            "yes_bid": yes_bid,
                            "yes_ask": yes_ask,
                            "spread": yes_ask - yes_bid,
                            "mid_price": (yes_bid + yes_ask) / 2,
                            "timestamp": datetime.now()
                        }
                        
                        # Send update via callback or queue
                        if self.update_callback:
                            self.update_callback(update)
                        else:
                            self.update_queue.put(update)
                        
                        # Log significant price movements
                        if last_prices[0] is not None:
                            if abs(yes_bid - last_prices[0]) > 0.02:  # 2 cent movement
                                logger.info(f"Price movement on {ticker}: {last_prices[0]:.2f} → {yes_bid:.2f}")
                                
        except Exception as e:
            logger.error(f"Error processing market update: {e}")
    
    async def _connect_and_listen(self, market_tickers: List[str]):
        """Connect to WebSocket and listen for updates."""
        uri = API_URLS['kalshi_ws']
        headers = self._get_auth_headers()
        
        try:
            logger.info(f"Connecting to Kalshi WebSocket: {uri}")
            header_list = [(k, v) for k, v in headers.items()]
            
            async with websockets.connect(uri, additional_headers=header_list) as websocket:
                self.websocket = websocket
                logger.info("WebSocket connected successfully")
                
                # Subscribe to markets
                subscribe_message = {
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker", "orderbook"],
                        "market_tickers": market_tickers
                    }
                }
                
                await websocket.send(json.dumps(subscribe_message))
                self.subscribed_markets = set(market_tickers)
                logger.info(f"Subscribed to {len(market_tickers)} markets")
                
                # Listen for messages
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self._process_market_update(data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse message: {message}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.update_queue.put({"error": str(e)})
    
    def start(self, market_tickers: List[str]):
        """Start WebSocket client in background thread."""
        self.running = True
        
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._connect_and_listen(market_tickers))
            except Exception as e:
                logger.error(f"WebSocket thread error: {e}")
                self.update_queue.put({"error": str(e)})
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        logger.info("Kalshi WebSocket client started")
        return thread
    
    def stop(self):
        """Stop WebSocket client."""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        logger.info("Kalshi WebSocket client stopped")
    
    def get_updates(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get updates from queue (non-blocking)."""
        try:
            return self.update_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class KalshiDataManager:
    """Manages Kalshi market data collection and storage."""
    
    def __init__(self):
        self.api_client = KalshiAPIClient()
        self.ws_client = None
        self.market_cache = {}
        self.price_history = {}
        
    def start_real_time_tracking(self, market_tickers: List[str]):
        """Start tracking markets in real-time."""
        def handle_update(update: Dict):
            ticker = update.get("ticker")
            if ticker:
                # Update cache
                self.market_cache[ticker] = update
                
                # Store price history
                if ticker not in self.price_history:
                    self.price_history[ticker] = []
                self.price_history[ticker].append(update)
                
                # Keep only last 1000 updates per market
                if len(self.price_history[ticker]) > 1000:
                    self.price_history[ticker] = self.price_history[ticker][-1000:]
        
        self.ws_client = KalshiWebSocketClient(update_callback=handle_update)
        self.ws_client.start(market_tickers)
        logger.info(f"Started real-time tracking for {len(market_tickers)} markets")
    
    def stop_real_time_tracking(self):
        """Stop real-time market tracking."""
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None
    
    def get_tracked_markets_df(self) -> pd.DataFrame:
        """Get DataFrame of currently tracked markets."""
        if not self.market_cache:
            return pd.DataFrame()
        
        data = []
        for ticker, market_data in self.market_cache.items():
            data.append({
                'ticker': ticker,
                'yes_bid': market_data.get('yes_bid'),
                'yes_ask': market_data.get('yes_ask'),
                'spread': market_data.get('spread'),
                'mid_price': market_data.get('mid_price'),
                'last_update': market_data.get('timestamp')
            })
        
        return pd.DataFrame(data)
    
    def save_market_data(self):
        """Save current market data to file."""
        if not self.market_cache:
            logger.warning("No market data to save")
            return
        
        filename = DATA_FILES['kalshi_markets'].format(
            date=datetime.now().strftime('%Y%m%d_%H%M')
        )
        
        with open(filename, 'w') as f:
            json.dump({
                'markets': list(self.market_cache.values()),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        logger.info(f"Saved market data to {filename}")


def debug_kalshi_api():
    """Debug Kalshi API to understand the exact issue."""
    print("🔍 Debugging Kalshi API Connection")
    print("=" * 50)
    
    # Check configuration
    print("Configuration Check:")
    print(f"  API Key present: {bool(KALSHI_API_KEY)}")
    print(f"  Private Key present: {bool(KALSHI_PRIVATE_KEY)}")
    print(f"  API URL: {API_URLS.get('kalshi_api')}")
    print()
    
    client = KalshiAPIClient()
    
    # Test 1: Get markets with different status values
    print("Test 1: Checking different market statuses...")
    for status in ["active", "open", "closed"]:
        params = {"limit": 5, "status": status}
        response = client._make_request("GET", "/trade-api/v2/markets", params=params)
        if response and "markets" in response:
            print(f"  {status}: {len(response['markets'])} markets found")
    
    # Test 2: Get ALL markets without status filter
    print("\nTest 2: Getting all available markets...")
    params = {"limit": 500}  # Get more markets
    response = client._make_request("GET", "/trade-api/v2/markets", params=params)
    
    if response and 'markets' in response:
        all_markets = response['markets']
        print(f"Total markets fetched: {len(all_markets)}")
        
        # Analyze market patterns
        print("\nMarket Analysis:")
        
        # Look for player names from the screenshot
        player_names = [
            "De'Von Achane", "Achane", "Tyreek Hill", "Hill", "Breece Hall", "Hall",
            "Garrett Wilson", "Wilson", "Jaylen Waddle", "Waddle", "Justin Fields", "Fields",
            "Braelon Allen", "Allen", "Ollie Gordon", "Gordon"
        ]
        
        player_markets = []
        for market in all_markets:
            title = market.get("title", "")
            for player in player_names:
                if player.lower() in title.lower():
                    player_markets.append(market)
                    break
        
        if player_markets:
            print(f"\n✅ Found {len(player_markets)} player markets:")
            for market in player_markets[:10]:  # Show first 10
                print(f"  Title: {market.get('title')}")
                print(f"  Ticker: {market.get('ticker')}")
                print(f"  Event: {market.get('event_ticker')}")
                print(f"  Status: {market.get('status')}")
                print()
        
        # Look for any touchdown/TD markets
        td_markets = []
        for market in all_markets:
            title = market.get("title", "").lower()
            if "touchdown" in title or " td" in title or "score" in title:
                td_markets.append(market)
        
        if td_markets:
            print(f"\n✅ Found {len(td_markets)} touchdown-related markets:")
            for market in td_markets[:5]:
                print(f"  - {market.get('title')[:80]}...")
        
        # Show sample of all tickers to understand patterns
        print("\nSample market tickers (first 20):")
        for market in all_markets[:20]:
            ticker = market.get("ticker", "")
            event = market.get("event_ticker", "")
            print(f"  {ticker[:40]}... (event: {event[:20] if event else 'None'})")
        
        # Look for specific ticker patterns
        print("\nTicker pattern analysis:")
        ticker_prefixes = {}
        for market in all_markets:
            ticker = market.get("ticker", "")
            if ticker:
                # Get prefix (first part before hyphen)
                prefix = ticker.split("-")[0] if "-" in ticker else ticker[:10]
                ticker_prefixes[prefix] = ticker_prefixes.get(prefix, 0) + 1
        
        print("Most common ticker prefixes:")
        for prefix, count in sorted(ticker_prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {prefix}: {count} markets")
    
    else:
        print("❌ Failed to fetch markets")
    
    print("\n" + "=" * 50)
    print("Debug complete!")


def test_kalshi_connection():
    """Test Kalshi API connection and functionality."""
    if not KALSHI_CONFIG.get('enabled'):
        print("⚠️ Kalshi integration is not enabled. Check API keys.")
        return
    
    print("🧪 Testing Kalshi API Connection")
    print("=" * 50)
    
    client = KalshiAPIClient()
    
    # Test 1: Get balance
    balance = client.get_balance()
    if balance is not None:
        print(f"✅ Account Balance: ${balance:.2f}")
    else:
        print("❌ Failed to get account balance")
    
    # Test 2: Get all NFL markets
    markets = client.get_nfl_markets()
    if markets:
        print(f"✅ Found {len(markets)} NFL markets")
        
        # Show first few markets
        for market in markets[:3]:
            print(f"  - {market.get('ticker')}: {market.get('title')}")
    else:
        print("❌ Failed to get NFL markets")
    
    # Test 3: Look for touchdown markets specifically
    print("\n🏈 Searching for Touchdown Markets...")
    
    if markets:
        touchdown_markets = []
        for market in markets:
            title = market.get("title", "")
            # Look for player names that appear to be in touchdown markets
            # Based on the screenshot, these are typically just player names
            # or "Anytime TD", "First TD", etc.
            if any(name in title for name in ["Achane", "Hill", "Hall", "Wilson", "Waddle", "Fields", "Allen", "Gordon"]):
                touchdown_markets.append(market)
            elif "touchdown" in title.lower() or " td" in title.lower():
                touchdown_markets.append(market)
        
        if touchdown_markets:
            print(f"✅ Found {len(touchdown_markets)} touchdown markets")
            for market in touchdown_markets[:5]:  # Show first 5
                ticker = market.get('ticker')
                title = market.get('title')
                print(f"\n  Market: {title}")
                print(f"  Ticker: {ticker}")
                
                # Get prices for this market
                prices = client.get_market_prices(ticker)
                if prices:
                    print(f"  Yes: Bid ${prices['yes_bid']:.2f} / Ask ${prices['yes_ask']:.2f}")
                    print(f"  No:  Bid ${prices['no_bid']:.2f} / Ask ${prices['no_ask']:.2f}")
                    print(f"  Spread: ${prices['spread']:.2f}")
        else:
            print("❌ No touchdown markets found")
    
    # Test 4: Get player prop markets using the new method
    prop_markets = client.get_player_prop_markets(prop_type='touchdown')
    if prop_markets:
        print(f"\n✅ Found {len(prop_markets)} player prop markets (touchdown filter)")
    else:
        print("\n⚠️ No player prop markets found with current filter")
    
    print("\n" + "=" * 50)
    print("✨ Kalshi API test complete")


if __name__ == "__main__":
    # Run debug first to understand the issue
    debug_kalshi_api()
    print("\n\n")
    # Then run the regular test
    test_kalshi_connection()