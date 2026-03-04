import asyncio
import websockets
import json
import requests
import time
from datetime import datetime, timezone

GAMMA_URL = "https://gamma-api.polymarket.com/markets/slug/"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"

MAX_SPREAD = 0.03
THROTTLE_DELAY = 1  # Temps d'attente minimum entre deux envois de vecteurs (en secondes)

# Variable globale partagée entre les deux tâches asynchrones pour stocker le spot price
current_spot_price = None

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
    """Récupère le prix d'ouverture via l'API Binance (proxy Chainlink)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": start_ts * 1000,
        "limit": 1
    }
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
    """Tâche d'arrière-plan avec reconnexion automatique robuste."""
    global current_spot_price
    
    while True:
        try:
            # AJOUT DU PING : On vérifie la connexion toutes les 20s. 
            # Si pas de réponse en 20s, ça coupe et ça relance la boucle.
            async with websockets.connect(RTDS_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                print("🔌 (Re)Connexion au flux Chainlink Spot Price...")
                
                subscribe_msg = {
                    "action": "subscribe",
                    "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": "{\"symbol\":\"btc/usd\"}"}]
                }
                await ws.send(json.dumps(subscribe_msg))

                while True:
                    # On attend un message avec un timeout de 60 secondes
                    # Si Chainlink ne dit rien pendant 1 minute, on considère que c'est suspect
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        
                        if msg == "PONG" or not msg.strip():
                            continue
                            
                        data = json.loads(msg)
                        if data.get("topic") == "crypto_prices_chainlink" and data.get("type") == "update":
                            price = data.get("payload", {}).get("value")
                            if price:
                                current_spot_price = float(price)
                                # (Optionnel) Décommente pour vérifier que ça vit :
                                # print(f"tick spot: {current_spot_price}")
                                
                    except asyncio.TimeoutError:
                        print("⚠️ Aucun tick Spot depuis 60s, reconnexion forcée...")
                        break # On sort du while interne pour relancer la connexion
                        
        except Exception as e:
            print(f"❌ Erreur Spot Price (Reconnexion dans 2s): {e}")
            current_spot_price = None # SÉCURITÉ : On efface le prix pour ne pas trader sur du vieux
            await asyncio.sleep(2)

async def watch_clob_market(queue):
    """Tâche principale qui surveille le carnet d'ordres et recrache le vecteur final."""
    last_print_time = 0.0
    
    while True:
        current_slug, current_ts, end_ts = get_current_slug_info()
        outcomes, asset_ids = fetch_market_info(current_slug)
        
        while not asset_ids:
            await asyncio.sleep(1)
            current_slug, current_ts, end_ts = get_current_slug_info()
            outcomes, asset_ids = fetch_market_info(current_slug)

        token_to_outcome = dict(zip(asset_ids, outcomes))
        prices = {"Up": 0.0, "Down": 0.0}
        K = None

        try:
            async with websockets.connect(CLOB_WS_URL) as ws:
                subscribe_msg = {
                    "assets_ids": asset_ids,
                    "type": "market",
                    "custom_feature_enabled": True
                }
                await ws.send(json.dumps(subscribe_msg))

                async def keep_alive_clob():
                    while True:
                        await asyncio.sleep(10)
                        try:
                            await ws.send("PING")
                        except:
                            break
                ping_task = asyncio.create_task(keep_alive_clob())

                while True:
                    _, new_ts, _ = get_current_slug_info()
                    if new_ts != current_ts:
                        ping_task.cancel()
                        break 

                    if K is None and time.time() >= current_ts:
                        K = fetch_strike_K(current_ts)

                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        
                        if message == "PONG" or not message.strip():
                            continue
                            
                        parsed_message = json.loads(message)
                        events = parsed_message if isinstance(parsed_message, list) else [parsed_message]
                        
                        state_changed = False
                        
                        for data in events:
                            if not isinstance(data, dict):
                                continue 
                                
                            event_type = data.get("event_type")
                            
                            if event_type in ["price_change", "book"]:
                                changes = data.get("price_changes", [data]) if event_type == "price_change" else [data]
                                
                                for change in changes:
                                    asset = change.get("asset_id")
                                    outcome = token_to_outcome.get(asset)
                                    
                                    if outcome in prices:
                                        bids = change.get("bids", [])
                                        asks = change.get("asks", [])
                                        
                                        bid = float(bids[0]["price"] if bids else change.get("best_bid", "0"))
                                        ask = float(asks[0]["price"] if asks else change.get("best_ask", "0"))
                                        
                                        if bid > 0 and ask > 0:
                                            spread = ask - bid
                                            if spread <= MAX_SPREAD:
                                                prix = (bid + ask) / 2
                                            else:
                                                prix = 0.0
                                        else:
                                            prix = 0.0
                                            
                                        if prix > 0 and prices[outcome] != prix:
                                            prices[outcome] = round(prix, 4)
                                            state_changed = True
                                            
                        # --- LE THROTTLE EST ICI ---
                        # Si l'état a changé, on vérifie si assez de temps s'est écoulé depuis le dernier print
                        if state_changed:
                            now = time.time()
                            if (now - last_print_time) >= THROTTLE_DELAY:
                                time_to_maturity = max(0.0, end_ts - now)
                                
                                # Vecteur: [Prix_Up, Prix_Down, Strike_K, Time_to_Maturity, Spot_Price]
                                vector = [prices.get("Up", 0), prices.get("Down", 0), K, round(time_to_maturity, 2), current_spot_price]
                                # Envoi du vecteur dans le tuyau principal
                                await queue.put(("polymarket", vector))
                                
                                last_print_time = now

                    except asyncio.TimeoutError:
                        continue
                        
        except Exception:
            await asyncio.sleep(1)