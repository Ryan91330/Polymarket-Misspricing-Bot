import numpy as np
import requests
import asyncio
import websockets
import json
import time
from collections import deque
from datetime import datetime

class FeatureEngine:
    def __init__(self):
        # --- MODIFIÉ : Mémoire réduite à 100 minutes (largement suffisant pour 1h) ---
        self.closes = deque(maxlen=100)
        self.highs = deque(maxlen=100)
        self.lows = deque(maxlen=100)
        
        # Le cache des "Fast Features" (30m)
        self.RV_30m = 0.0
        self.Mom_30m = 0.0
        self.RV_Ratio = 1.0
        self.Parkinson_30m = 0.0
        self.current_r = 0.05  
        
        # --- MODIFIÉ : Le cache des "Macro Features" (15m, 1h, Temps) ---
        self.Mom_15m = 0.0
        self.Mom_1h = 0.0
        self.Dist_SMA_15m = 0.0
        self.Dist_SMA_1h = 0.0
        self.RSI_1h = 50.0       # Valeur neutre par défaut
        self.RV_1h = 0.0
        self.Vol_Ratio_30m_1h = 1.0
        self.Hour_of_day = 0
        self.Day_of_week = 0
        
        self.annualization_factor = np.sqrt(365.25 * 24 * 60)

    def warmup(self):
        """S'échauffe avec la DERNIÈRE HEURE (60 minutes) de Spot."""
        print("🔥 Échauffement du Feature Engine (Téléchargement de 1h de data)...")
        
        self._update_r_sync()
        
        # Warmup des Prix Binance sur 100 bougies (1 seule requête suffit)
        try:
            url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
            res = requests.get(url, timeout=5).json()
            
            for kline in res:
                self.closes.append(float(kline[4])) # Close
                self.highs.append(float(kline[2]))  # High
                self.lows.append(float(kline[3]))   # Low
            
            self._calculate_slow_features()
            print(f"✅ Feature Engine prêt ! ({len(self.closes)} bougies en mémoire)")
            
        except Exception as e:
            print(f"❌ Erreur lors du warmup Binance: {e}")

    def _update_r_sync(self):
        try:
            url_inst = "https://www.deribit.com/api/v2/public/get_instruments"
            params = {"currency": "BTC", "kind": "future", "expired": "false"}
            res = requests.get(url_inst, params=params, timeout=5).json()
            instruments = res.get("result", [])
            valid_futures = []
            now_ms = time.time() * 1000
            for inst in instruments:
                if "PERPETUAL" in inst["instrument_name"]: continue
                expiry_ts = inst["expiration_timestamp"]
                tau_f = (expiry_ts - now_ms) / (1000 * 3600 * 24 * 365.25)
                if (7/365.25) <= tau_f <= (120/365.25):
                    valid_futures.append({"name": inst["instrument_name"], "tau": tau_f})
            if valid_futures:
                target = min(valid_futures, key=lambda x: x["tau"])
                tick_url = f"https://www.deribit.com/api/v2/public/ticker?instrument_name={target['name']}"
                tick_res = requests.get(tick_url, timeout=5).json()
                ticker = tick_res.get("result", {})
                S = ticker.get("index_price")
                F = ticker.get("last_price")
                if S and F:
                    self.current_r = float(np.log(F/S) / target["tau"])
        except Exception as e:
            pass

    async def watch_live_r(self):
        while True:
            await asyncio.sleep(3600)
            await asyncio.to_thread(self._update_r_sync)

    def _calculate_slow_features(self):
        """Calcule TOUTES les volatilités, momentum, et indicateurs macro."""
        if len(self.closes) < 61: # On a besoin de 1h (60m) pour tourner à plein régime
            return 
            
        closes_arr = np.array(self.closes)
        highs_arr = np.array(self.highs)
        lows_arr = np.array(self.lows)
        
        # --- 1. FEATURES COURT TERME (30m) ---
        log_rets = np.log(closes_arr[1:] / closes_arr[:-1])
        
        self.RV_30m = np.std(log_rets[-30:], ddof=1) * self.annualization_factor
        rv_5m = np.std(log_rets[-5:], ddof=1) * self.annualization_factor
        self.RV_Ratio = (rv_5m / self.RV_30m) if self.RV_30m > 0 else 1.0
            
        self.Mom_30m = (closes_arr[-1] / closes_arr[-30]) - 1.0
        
        max_high = np.max(highs_arr[-30:])
        min_low = np.min(lows_arr[-30:])
        self.Parkinson_30m = np.log(max_high / min_low) if min_low > 0 else 0.0

        # --- 2. MODIFIÉ : FEATURES MACRO (15m et 1h) ---
        
        # Momentum
        self.Mom_15m = (closes_arr[-1] / closes_arr[-15]) - 1.0
        self.Mom_1h = (closes_arr[-1] / closes_arr[-60]) - 1.0
        
        # Distances SMA
        sma_15m = np.mean(closes_arr[-15:])
        self.Dist_SMA_15m = (closes_arr[-1] - sma_15m) / sma_15m
        
        sma_1h = np.mean(closes_arr[-60:])
        self.Dist_SMA_1h = (closes_arr[-1] - sma_1h) / sma_1h
        
        # RV 1h et Ratio
        self.RV_1h = np.std(log_rets[-60:], ddof=1) * self.annualization_factor
        self.Vol_Ratio_30m_1h = (self.RV_30m / self.RV_1h) if self.RV_1h > 0 else 1.0

        # RSI 1h (Basé sur des changements d'1 minute sur les 60 dernières minutes)
        if len(closes_arr) >= 61:
            relevant_closes = closes_arr[-61:] 
            delta = relevant_closes[1:] - relevant_closes[:-1]
            
            gain = np.mean(np.where(delta > 0, delta, 0))
            loss = np.mean(np.where(delta < 0, -delta, 0))
            
            if loss == 0:
                self.RSI_1h = 100.0
            else:
                rs = gain / loss
                self.RSI_1h = 100.0 - (100.0 / (1.0 + rs))

        # --- 3. CONTEXTE TEMPOREL ---
        now = datetime.utcnow() # Utiliser UTC pour être aligné avec les exchanges
        self.Hour_of_day = now.hour
        self.Day_of_week = now.weekday()

    async def watch_binance_1m(self):
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    print("🔌 Connexion au flux Binance 1m établie.")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        kline = data['k']
                        
                        if kline['x']: # Clôture de bougie (1x par minute)
                            self.closes.append(float(kline['c']))
                            self.highs.append(float(kline['h']))
                            self.lows.append(float(kline['l']))
                            
                            # On met à jour toutes les features macro et micro
                            self._calculate_slow_features()
                            
            except Exception as e:
                print(f"⚠️ Erreur Binance (Reconnexion dans 2s): {e}")
                await asyncio.sleep(2)

    def get_live_vector(self, spot_price, strike_k, tau_years):
        """
        Génère le vecteur avec l'ordre EXACT des colonnes.
        """
        # Calcul des features instantanées liées à l'option
        M = spot_price / strike_k
        abs_dist_ATM = abs(M - 1.0)
        tau_sqrt = np.sqrt(tau_years)

        # Assemblage STRICT (Modifié pour 15m et 1h)
        vector = [
            M,
            tau_years,
            abs_dist_ATM,
            
            self.RV_30m,
            self.Mom_30m,
            self.RV_Ratio,
            self.Parkinson_30m,
            
            self.Mom_15m,
            self.Mom_1h,
            self.Dist_SMA_15m,
            self.Dist_SMA_1h,
            self.RSI_1h,
            self.RV_1h,
            self.Vol_Ratio_30m_1h,
            
            self.Hour_of_day,
            self.Day_of_week
        ]
        
        return np.array(vector).reshape(1, -1)