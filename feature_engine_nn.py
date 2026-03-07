import numpy as np
import requests
import asyncio
import websockets
import json
import time
import joblib
import torch
import pandas as pd
from collections import deque
from datetime import datetime

class FeatureEngine:
    def __init__(self, scaler_path='/home/ryan/GitProject/IV_Model/NN_IV/scaler/nn_scaler_v2.pkl', device=None):
        # Configuration PyTorch & Scaler
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler chargé depuis {scaler_path}")
        except Exception as e:
            print(f"⚠️ ATTENTION : Impossible de charger le scaler ({e}). Le NN va prédire n'importe quoi sans ça !")
            self.scaler = None

        # Buffer étendu pour les prix Spot (OHLC 1m) - 24 heures + marge
        self.closes = deque(maxlen=1450)
        self.highs = deque(maxlen=1450)
        self.lows = deque(maxlen=1450)
        
        # Le cache des "Fast Features" (30m)
        self.RV_30m = 0.0
        self.Mom_30m = 0.0
        self.RV_Ratio = 1.0
        self.Parkinson_30m = 0.0
        self.current_r = 0.05  # <--- R est bien là
        
        # Le cache des "Macro Features" (4h, 24h)
        self.Mom_4h = 0.0
        self.Mom_24h = 0.0
        self.Dist_SMA_4h = 0.0
        self.Dist_SMA_24h = 0.0
        self.RSI_14h = 50.0
        self.RV_24h = 0.0
        self.Vol_Ratio_30m_24h = 1.0
        
        # Contexte Temporel
        self.Hour_of_day = 0
        self.Day_of_week = 0
        
        self.annualization_factor = np.sqrt(365.25 * 24 * 60)

    def warmup(self):
        """S'échauffe avec les 1440 dernières minutes (24h) pour le NN."""
        print("🔥 Échauffement du Feature Engine (Téléchargement 24h)...")
        
        self._update_r_sync()
        
        try:
            # Requête 1 : Les 1000 minutes les plus anciennes
            url1 = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1000"
            res1 = requests.get(url1, timeout=5).json()
            
            # Requête 2 : Les 450 suivantes pour compléter les 24h
            last_timestamp = res1[-1][0]
            url2 = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=450&startTime={last_timestamp + 60000}"
            res2 = requests.get(url2, timeout=5).json()
            
            full_data = res1 + res2
            
            for kline in full_data:
                self.closes.append(float(kline[4])) # Close
                self.highs.append(float(kline[2]))  # High
                self.lows.append(float(kline[3]))   # Low
            
            self._calculate_slow_features()
            print(f"✅ Feature Engine prêt ! ({len(self.closes)} bougies en mémoire)")
        except Exception as e:
            print(f"❌ Erreur lors du warmup Binance: {e}")

    def _update_r_sync(self):
        """Récupère le taux r en direct depuis Deribit"""
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
                    # print(f"🏦 Taux sans risque 'r' mis à jour : {self.current_r:.4f}")
        except Exception as e:
            pass

    async def watch_live_r(self):
        while True:
            await asyncio.sleep(3600)
            await asyncio.to_thread(self._update_r_sync)

    def _calculate_slow_features(self):
        """Calcule TOUTES les features macro et micro (1x par minute)"""
        if len(self.closes) < 1440:
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

        # --- 2. FEATURES MACRO (4h et 24h) ---
        self.Mom_4h = (closes_arr[-1] / closes_arr[-240]) - 1.0
        self.Mom_24h = (closes_arr[-1] / closes_arr[-1440]) - 1.0
        
        sma_4h = np.mean(closes_arr[-240:])
        self.Dist_SMA_4h = (closes_arr[-1] - sma_4h) / sma_4h
        
        sma_24h = np.mean(closes_arr[-1440:])
        self.Dist_SMA_24h = (closes_arr[-1] - sma_24h) / sma_24h
        
        self.RV_24h = np.std(log_rets[-1440:], ddof=1) * self.annualization_factor
        self.Vol_Ratio_30m_24h = (self.RV_30m / self.RV_24h) if self.RV_24h > 0 else 1.0

        # RSI 14h (840 minutes)
        if len(closes_arr) >= 840 + 60:
            relevant_closes = closes_arr[-(840+60):] 
            delta_1h = relevant_closes[60:] - relevant_closes[:-60]
            
            gain = np.mean(np.where(delta_1h > 0, delta_1h, 0))
            loss = np.mean(np.where(delta_1h < 0, -delta_1h, 0))
            
            if loss == 0:
                self.RSI_14h = 100.0
            else:
                rs = gain / loss
                self.RSI_14h = 100.0 - (100.0 / (1.0 + rs))

        # --- 3. CONTEXTE TEMPOREL ---
        now = datetime.utcnow()
        self.Hour_of_day = now.hour
        self.Day_of_week = now.weekday()

    async def watch_binance_1m(self):
        """Écoute Binance pour mettre à jour les bougies."""
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    print("🔌 Connexion au flux Binance 1m établie.")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        kline = data['k']
                        
                        if kline['x']: # Clôture de bougie
                            self.closes.append(float(kline['c']))
                            self.highs.append(float(kline['h']))
                            self.lows.append(float(kline['l']))
                            self._calculate_slow_features()
                            
            except Exception as e:
                print(f"⚠️ Erreur Binance (Reconnexion dans 2s): {e}")
                await asyncio.sleep(2)

    def get_live_vector(self, spot_price, strike_k, tau_years):
        """
        Génère le tenseur PyTorch normalisé pour le réseau de neurones.
        17 Features attendues.
        """
        # 1. Calculs Live
        M = spot_price / strike_k
        abs_dist_ATM = abs(M - 1.0)
        
        # 2. Assemblage Brut (L'ordre est STRICTEMENT celui de l'entraînement)
        raw_vector = [
            M,
            tau_years,
            self.current_r,       # <--- R a été replacé ici (Index 2)
            abs_dist_ATM,
            self.RV_30m,
            self.Mom_30m,
            self.RV_Ratio,
            self.Parkinson_30m,
            self.Mom_4h,
            self.Mom_24h,
            self.Dist_SMA_4h,
            self.Dist_SMA_24h,
            self.RSI_14h,
            self.RV_24h,
            self.Vol_Ratio_30m_24h,
            self.Hour_of_day,
            self.Day_of_week
        ]
        
        # Format numpy (1 ligne, 17 colonnes)
        raw_array = np.array(raw_vector).reshape(1, -1)
        
        # 3. Standardisation (VITAL pour le NN)
        if self.scaler:
            # On recrée un DataFrame éphémère avec les noms de l'entraînement incluant 'r'
            feature_names = [
                'M', 'tau', 'r', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m',
                'Mom_4h', 'Mom_24h', 'Dist_SMA_4h', 'Dist_SMA_24h', 'RSI_14h', 
                'RV_24h', 'Vol_Ratio_30m_24h', 'Hour_of_day', 'Day_of_week'
            ]
            
            df_live = pd.DataFrame(raw_array, columns=feature_names)
            scaled_array = self.scaler.transform(df_live)
        else:
            scaled_array = raw_array 
            
        # 4. Conversion PyTorch
        tensor = torch.tensor(scaled_array, dtype=torch.float32).to(self.device)
        
        return tensor