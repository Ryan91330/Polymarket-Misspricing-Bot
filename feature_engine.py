import numpy as np
import requests
import asyncio
import websockets
import json
import time
from collections import deque

class FeatureEngine:
    def __init__(self):
        # Buffer pour les prix Spot (OHLC 1m)
        self.closes = deque(maxlen=31)
        self.highs = deque(maxlen=30)
        self.lows = deque(maxlen=30)
        
        # Le cache des "Slow Features"
        self.RV_30m = 0.0
        self.Mom_30m = 0.0
        self.RV_Ratio = 1.0
        self.Parkinson_30m = 0.0
        self.current_r = 0.05  # Valeur par défaut de sécurité
        
        self.annualization_factor = np.sqrt(365.25 * 24 * 60)

    def warmup(self):
        """S'échauffe avec les 30 dernières minutes Spot et le taux 'r' actuel."""
        print("🔥 Échauffement du Feature Engine...")
        
        # 1. Warmup du Taux 'r'
        self._update_r_sync()
        
        # 2. Warmup des Prix Binance
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=31"
        try:
            res = requests.get(url, timeout=5)
            data = res.json()
            for kline in data:
                self.closes.append(float(kline[4])) # Close
                self.highs.append(float(kline[2]))  # High
                self.lows.append(float(kline[3]))   # Low
            
            self._calculate_slow_features()
            print("✅ Feature Engine prêt !")
        except Exception as e:
            print(f"❌ Erreur lors du warmup Binance: {e}")

    def _update_r_sync(self):
        """Récupère le taux r en direct depuis Deribit (version Live de ton script)"""
        try:
            # On cherche les instruments live
            url_inst = "https://www.deribit.com/api/v2/public/get_instruments"
            params = {"currency": "BTC", "kind": "future", "expired": "false"}
            res = requests.get(url_inst, params=params, timeout=5).json()
            instruments = res.get("result", [])
            
            valid_futures = []
            now_ms = time.time() * 1000
            
            for inst in instruments:
                if "PERPETUAL" in inst["instrument_name"]:
                    continue
                    
                expiry_ts = inst["expiration_timestamp"]
                tau_f = (expiry_ts - now_ms) / (1000 * 3600 * 24 * 365.25)
                
                # Front-month future (entre 7 et 120 jours)
                if (7/365.25) <= tau_f <= (120/365.25):
                    valid_futures.append({"name": inst["instrument_name"], "tau": tau_f})
            
            if valid_futures:
                target = min(valid_futures, key=lambda x: x["tau"])
                
                # Récupération du prix actuel de ce Future
                tick_url = f"https://www.deribit.com/api/v2/public/ticker?instrument_name={target['name']}"
                tick_res = requests.get(tick_url, timeout=5).json()
                ticker = tick_res.get("result", {})
                
                S = ticker.get("index_price")
                F = ticker.get("last_price")
                
                if S and F:
                    self.current_r = float(np.log(F/S) / target["tau"])
                    print(f"🏦 Taux sans risque 'r' mis à jour : {self.current_r:.4f}")
        except Exception as e:
            print(f"⚠️ Impossible de fetch 'r', maintien de la valeur précédente : {self.current_r}")

    async def watch_live_r(self):
        """Tâche de fond : met à jour 'r' toutes les heures sans bloquer le bot"""
        while True:
            await asyncio.sleep(3600) # 1 heure
            # On utilise to_thread pour ne pas bloquer la boucle asyncio avec requests
            await asyncio.to_thread(self._update_r_sync)

    def _calculate_slow_features(self):
        """Calcule les volatilités et momentum historiques (1x par minute)"""
        if len(self.closes) < 31:
            return 
            
        closes_arr = np.array(self.closes)
        highs_arr = np.array(self.highs)
        lows_arr = np.array(self.lows)
        
        log_rets = np.log(closes_arr[1:] / closes_arr[:-1])
        
        self.RV_30m = np.std(log_rets, ddof=1) * self.annualization_factor
        
        rv_5m = np.std(log_rets[-5:], ddof=1) * self.annualization_factor
        self.RV_Ratio = (rv_5m / self.RV_30m) if self.RV_30m > 0 else 1.0
            
        self.Mom_30m = (closes_arr[-1] / closes_arr[0]) - 1.0
        
        max_high = np.max(highs_arr)
        min_low = np.min(lows_arr)
        self.Parkinson_30m = np.log(max_high / min_low) if min_low > 0 else 0.0

    async def watch_binance_1m(self):
        """Écoute Binance avec gestion des déconnexions silencieuses."""
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        while True:
            try:
                # Ping interval est vital pour Binance
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    print("🔌 (Re)Connexion au flux Binance 1m...")
                    while True:
                        msg = await ws.recv() # Ici, le ping_interval gère le timeout interne
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
        Le générateur du vecteur final pour le modèle ML.
        Ordre EXACT : ['M', 'tau', 'r', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m', 'smile_curve', 'M_squared', 'tau_sqrt']
        """
        # 1. Calcul des features instantanées (Fast)
        M = spot_price / strike_k
        abs_dist_ATM = abs(M - 1.0)
        smile_curve = (M - 1.0)**2
        M_squared = M**2
        tau_sqrt = np.sqrt(tau_years)
        
        # 2. Assemblage strict
        vector = [
            M,
            tau_years,
            self.current_r,       # Le taux en live
            abs_dist_ATM,
            self.RV_30m,          # Cache
            self.Mom_30m,         # Cache
            self.RV_Ratio,        # Cache
            self.Parkinson_30m,   # Cache
            smile_curve,
            M_squared,
            tau_sqrt
        ]
        
        # Retourne une matrice numpy propre pour XGBoost/PyTorch
        return np.array(vector).reshape(1, -1)