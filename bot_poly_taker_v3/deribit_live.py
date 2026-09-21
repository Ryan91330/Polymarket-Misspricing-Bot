import asyncio
import websockets
import json
import time

DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"

MAX_DAYS = 1         # Options expirant dans les 30 prochains jours
UPDATE_INTERVAL = 10   # Fréquence de récupération du dataset complet (en secondes)

async def fetch_options_dataset(queue):
    async with websockets.connect(DERIBIT_WS_URL) as ws:
        
        # 1. INITIALISATION : Cartographie des options
        # On récupère les expirations exactes une seule fois au début
        print("Récupération de la liste des instruments...")
        msg_instruments = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/get_instruments",
            "params": {"currency": "BTC", "kind": "option", "expired": False}
        }
        await ws.send(json.dumps(msg_instruments))
        resp = json.loads(await ws.recv())
        
        now_ms = time.time() * 1000
        cutoff_ms = now_ms + (MAX_DAYS * 24 * 3600 * 1000)
        
        option_map = {}
        for inst in resp.get("result", []):
            exp_ts = inst["expiration_timestamp"]
            if exp_ts <= cutoff_ms:
                option_map[inst["instrument_name"]] = {
                    "type": "C" if inst["option_type"] == "call" else "P",
                    "strike": inst["strike"],
                    "exp_ts": exp_ts / 1000.0 # On garde en secondes pour le calcul de Tau
                }
        
        print(f"Cartographie terminée : {len(option_map)} options BTC retenues (< 30 jours).")
        print("Démarrage de l'extraction du dataset en direct...\n")

        msg_id = 2
        
        # 2. BOUCLE PRINCIPALE : Snapshot du dataset
        while True:
            # Demande l'état de TOUTES les options BTC à l'instant T
            msg_summary = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "public/get_book_summary_by_currency",
                "params": {"currency": "BTC", "kind": "option"}
            }
            await ws.send(json.dumps(msg_summary))
            
            # On attend spécifiquement la réponse à notre requête (via l'ID)
            while True:
                response = json.loads(await ws.recv())
                if response.get("id") == msg_id:
                    break
            
            msg_id += 1
            summaries = response.get("result", [])
            
            dataset = []
            now = time.time()
            
            for opt in summaries:
                name = opt["instrument_name"]
                
                # On ne garde que celles qui sont dans notre filtre < 30 jours
                if name in option_map:
                    info = option_map[name]
                    
                    tau_sec = max(0.0, info["exp_ts"] - now)
                    
                    # Extraction et conversion des données (Deribit cote en BTC, on veut de l'USD)
                    mark_price_btc = opt.get("mark_price", 0.0)
                    underlying = opt.get("underlying_price", 0.0)
                    price_usd = mark_price_btc * underlying
                    
                    iv = opt.get("mark_iv", 0.0)
                    
                    # On ignore les options sans liquidité ou sans IV calculable
                    if price_usd > 0 and iv > 0:
                        row = [
                            info["type"],           # Call / Put
                            info["strike"],         # Strike (K)
                            round(tau_sec, 2),      # Tau (Secondes)
                            round(price_usd, 2),    # Prix (USD)
                            round(iv, 2)            # IV (%)
                        ]
                        dataset.append(row)
            
            # Affichage du dataset sous forme de matrice (liste de listes)
            # Envoi du dataset dans le tuyau principal
            await queue.put(("deribit", dataset))
            
            # Pause avant le prochain snapshot
            await asyncio.sleep(UPDATE_INTERVAL)
