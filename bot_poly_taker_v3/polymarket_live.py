import asyncio
import websockets
import json
import requests
import time
from datetime import datetime, timezone

# --- TES URLS (INCHANGÉES) ---
GAMMA_URL = "https://gamma-api.polymarket.com/markets/slug/"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"

THROTTLE_DELAY = 1  # Plus rapide pour le Maker

# Variable globale
current_spot_price = None

# --- TES FONCTIONS (INCHANGÉES) ---
def get_current_slug_info():
    now_utc = datetime.now(timezone.utc)
    minute_rounded = (now_utc.minute // 15) * 15
    start_of_window = now_utc.replace(minute=minute_rounded, second=0, microsecond=0)
    
    start_ts = int(start_of_window.timestamp())
    end_ts = start_ts + 900 
    
    return f"btc-updown-15m-{start_ts}", start_ts, end_ts

def fetch_market_info(slug):
    try:
        res = requests.get(GAMMA_URL + slug, timeout=5)
        if res.status_code == 200:
            data = res.json()
            try:
                outcomes = json.loads(data.get("outcomes", "[]"))
                tokens = json.loads(data.get("clobTokenIds", "[]"))
            except json.JSONDecodeError:
                outcomes, tokens = [], []
            return outcomes, tokens
    except requests.exceptions.RequestException:
        pass
    return [], []

def fetch_strike_K(start_ts):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "startTime": start_ts * 1000, "limit": 1}
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                return float(data[0][1])
    except Exception:
        pass
    return None

async def watch_spot_price():
    global current_spot_price
    while True:
        try:
            async with websockets.connect(RTDS_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                subscribe_msg = {
                    "action": "subscribe",
                    "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": "{\"symbol\":\"btc/usd\"}"}]
                }
                await ws.send(json.dumps(subscribe_msg))

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        if msg == "PONG" or not msg.strip(): continue
                        data = json.loads(msg)
                        if data.get("topic") == "crypto_prices_chainlink" and data.get("type") == "update":
                            price = data.get("payload", {}).get("value")
                            if price: current_spot_price = float(price)
                    except asyncio.TimeoutError: break
        except Exception:
            await asyncio.sleep(2)

# --- NOUVELLE FONCTION watch_clob_market (ADAPTÉE MAKER) ---

async def watch_clob_market(queue):
    """Surveille le carnet et envoie le vecteur riche [bid_up, ask_up, bid_dn, ask_dn, ...]"""
    last_print_time = 0.0
    
    while True:
        # 1. Récupération des IDs via ton slug
        current_slug, current_ts, end_ts = get_current_slug_info()
        outcomes, asset_ids = fetch_market_info(current_slug)
        
        while not asset_ids or len(asset_ids) < 2:
            await asyncio.sleep(1)
            current_slug, current_ts, end_ts = get_current_slug_info()
            outcomes, asset_ids = fetch_market_info(current_slug)

        id_up = asset_ids[0]
        id_down = asset_ids[1]
        
        book = {
            "Up": {"bid": 0.0, "ask": 0.0},
            "Down": {"bid": 0.0, "ask": 0.0}
        }
        K = None

        try:
            async with websockets.connect(CLOB_WS_URL) as ws:
                subscribe_msg = {
                    "type": "market",
                    "assets_ids": [id_up, id_down],
                    "custom_feature_enabled": True 
                }
                await ws.send(json.dumps(subscribe_msg))

                async def keep_alive_clob():
                    while True:
                        await asyncio.sleep(10)
                        try: await ws.send("PING")
                        except: break
                ping_task = asyncio.create_task(keep_alive_clob())

                print(f"🎧 Flux Maker actif pour : {current_slug}")

                while True:
                    now = time.time()
                    
                    _, new_ts, _ = get_current_slug_info()
                    if new_ts != current_ts:
                        ping_task.cancel()
                        break 

                    if K is None and now >= current_ts:
                        K = fetch_strike_K(current_ts)

                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if message == "PONG" or not message.strip(): continue
                            
                        data_list = json.loads(message)
                        if not isinstance(data_list, list): data_list = [data_list]
                        
                        state_changed = False
                        
                        for data in data_list:
                            event_type = data.get("event_type")
                            asset = data.get("asset_id")
                            direction = "Up" if asset == id_up else "Down" if asset == id_down else None
                            if not direction: continue

                            if event_type == "book":
                                if data.get("bids"): book[direction]["bid"] = float(data["bids"][0]["price"])
                                if data.get("asks"): book[direction]["ask"] = float(data["asks"][0]["price"])
                                state_changed = True
                            
                            elif event_type == "best_bid_ask":
                                book[direction]["bid"] = float(data.get("best_bid", 0.0))
                                book[direction]["ask"] = float(data.get("best_ask", 0.0))
                                state_changed = True
                                            
                        # ======================================================
                        # SÉCURITÉ ANTI-NONETYPE : ON VÉRIFIE SPOT ET STRIKE
                        # ======================================================
                        if state_changed and current_spot_price is not None and K is not None:
                            if (now - last_print_time) >= THROTTLE_DELAY:
                                rich_data = {
                                    "bid_up": book["Up"]["bid"],
                                    "ask_up": book["Up"]["ask"],
                                    "bid_dn": book["Down"]["bid"],
                                    "ask_dn": book["Down"]["ask"],
                                    "strike": K,
                                    "tau": round(max(0, end_ts - now), 2),
                                    "spot": current_spot_price
                                }
                                await queue.put(("polymarket", rich_data))
                                last_print_time = now
                        # ======================================================

                    except asyncio.TimeoutError:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Erreur WS Clob: {e}")
            await asyncio.sleep(2)