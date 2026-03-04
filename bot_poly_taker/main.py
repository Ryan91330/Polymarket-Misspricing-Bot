import asyncio
import time
import xgboost as xgb
# Importation de tes scripts
from deribit_live import fetch_options_dataset
from polymarket_live import watch_spot_price, watch_clob_market
from feature_engine import FeatureEngine
from pricing import BS_Bin
from trade import calculate_trade_signal,PaperTrader
from risk_manager import get_dynamic_thresholds

async def expiration_watchdog(portfolio):
    """
    Force la fermeture 10s avant la fin en utilisant le dernier prix connu.
    """
    print("👮 Watchdog d'expiration activé.")
    
    while True:
        await asyncio.sleep(1)
        now = time.time()
        
        # Fin du cycle de 15 minutes (ex: 14:00, 14:15...)
        current_window_end = (int(now) // 900 + 1) * 900
        seconds_remaining = current_window_end - now
        
        # ZONE DE DANGER : Moins de 10 secondes
        if seconds_remaining < 10.0:
            
            if portfolio.positions:
                print(f"⏰ URGENT : Fin du cycle dans {seconds_remaining:.1f}s ! Force Close...")
                
                active_directions = list(portfolio.positions.keys())
                
                for direction in active_directions:
                    # 1. On récupère la position pour voir son dernier prix
                    pos_data = portfolio.positions.get(direction)
                    
                    if pos_data:
                        # 2. On utilise le dernier prix vu par le bot (mis à jour à chaque tick)
                        # Si jamais le prix est None (bug), on met 0 par sécurité pour ne pas inventer d'argent
                        exit_price = pos_data.get("last_price", 0.0)
                        
                        portfolio.close_position(direction, exit_price, "FORCE CLOSE (Watchdog)")
                
                portfolio.print_stats() # On affiche les stats après le nettoyage
                print("🧹 Portefeuille nettoyé.")
                
            await asyncio.sleep(10)


async def orchestrator():
    print("🤖 Initialisation XGBoost...")
    live_xgb = xgb.XGBRegressor()
    live_xgb.load_model("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/models/best_xgb_model_r_4h.json")
    print("🚀 Démarrage du Bot de Trading...")
    # --- INITIALISATION DU MOTEUR DE FEATURES ---
    engine = FeatureEngine()
    engine.warmup() # Télécharge le passé immédiat en REST
    # 1. Le tuyau de communication central
    data_queue = asyncio.Queue()
    
    # 2. Lancement des 3 tâches en arrière-plan
    # (Elles vont tourner à l'infini et nourrir la queue)
    asyncio.create_task(engine.watch_live_r())
    asyncio.create_task(engine.watch_binance_1m())
    asyncio.create_task(fetch_options_dataset(data_queue))
    asyncio.create_task(watch_spot_price()) # Met à jour la variable globale dans polymarket_live
    asyncio.create_task(watch_clob_market(data_queue))
    
    # 3. Mémoire d'état du marché
    latest_deribit_dataset = None
    portfolio = PaperTrader(initial_balance=500.0)
    print("🎧 En écoute des flux live...")
    asyncio.create_task(expiration_watchdog(portfolio))
    
    # ⏱️ PARAMÈTRE DE PERSISTANCE
    CONFIRMATION_DELAY = 2  # Le signal doit rester valide 5 secondes pour tirer
    
    # État du chronomètre
    pending_signal = {
        "direction": None,  # "UP" ou "DOWN"
        "start_time": None
    }
    
    # 4. Boucle de consommation des événements
    while True:
        # On attend qu'une des tâches envoie une donnée
        source, data = await data_queue.get()
        
        if source == "deribit":
            # On met à jour la "photo" du marché des options
            latest_deribit_dataset = data
            print(f"[DERIBIT] Surface de volatilité mise à jour ({len(data)} options en mémoire).")
            
        elif source == "polymarket":
            poly_vector = data
            prix_up, prix_down, strike_k, time_to_maturity, spot_price = poly_vector
            
            # SÉCURITÉ : Vérifier si le Spot Price est frais
            # Si current_spot_price est None (à cause de la modif 1) -> On ne trade pas
            if spot_price is None:
                print("⏳ Attente du flux Spot Price...")
                continue
            # On met à jour le "Dernier Prix Connu" pour le Watchdog
            if "UP" in portfolio.positions:
                portfolio.update_price("UP", prix_up)
            if "DOWN" in portfolio.positions:
                portfolio.update_price("DOWN", prix_down)
                
            if latest_deribit_dataset is not None and strike_k is not None and spot_price is not None:
                try:
                    tau_years = time_to_maturity / (365.25 * 24 * 3600)
                    
                    # 1. Prédiction XGBoost et Pricing Binaire
                    X_live = engine.get_live_vector(spot_price, strike_k, tau_years)
                    sigma = float(live_xgb.predict(X_live)[0])
                    r_live = float(X_live[0][2])
                    
                    fair_up, fair_dn = BS_Bin(spot_price, r_live, strike_k, sigma, tau_years)
                    
                    # Affichage des prix (Optionnel : tu peux commenter pour que la console soit moins spammée)
                    print(f"Tick PM | S: {spot_price} | Tau: {time_to_maturity:.0f}s || UP: PM={prix_up} FV={fair_up:.3f}")
                    
                    # ==========================================
                    # 2. GESTION DES SORTIES (MODE HYBRIDE : LOTTO + SMART)
                    # ==========================================
                    active_pos_keys = list(portfolio.positions.keys())
                    now = time.time()
                    
                    for direction in active_pos_keys:
                        pos = portfolio.positions[direction]
                        
                        entry = pos["entry_price"]
                        highest = pos["highest_price"]
                        start_t = pos["start_time"]
                        current_p = prix_up if direction == "UP" else prix_down
                        
                    
                        # PROTECTION 2 : Période de Grâce (7s)
                        if (now - start_t) < 5.0:
                            # Exception : Crash immédiat sur un trade normal
                            # if entry >= 0.10 and current_p < entry * 0.60:
                            #     portfolio.close_position(direction, current_p, "Panic Sell (Early Crash)")
                            continue
                        
                        # --- CAS SPÉCIAL : MODE "LOTTO" (< 0.10$) ---
                        # Ici, on applique ta règle stricte : x2 ou rien.
                        # Pas de Trailing Stop, pas de Time Stop (sauf expiration watchdog).
                        if entry < 0.08:
                            # 1. Take Profit : On vise +100% (x2)
                            if current_p >= entry * 2.0:
                                portfolio.close_position(direction, current_p, "🏆 LOTTO WIN (+100%)")
                                portfolio.print_stats()
                            
                            # 2. Stop Loss : On accepte de tout perdre (0.005$)
                            elif current_p <= 0.005:
                                portfolio.close_position(direction, current_p, "💀 LOTTO DEAD")
                                portfolio.print_stats()
                                
                            # IMPORTANT : On saute tout le reste pour ce trade (continue)
                            continue

                        # ======================================================
                        # LOGIQUE STANDARD (POUR LES TRADES > 0.10$)
                        # ======================================================

                        # --- C. TAKE PROFIT ULTIME ---
                        if current_p >= 0.94:
                            portfolio.close_position(direction, current_p, "Take Profit FINAL (0.94$)")
                            portfolio.print_stats()
                            continue

                        # --- A. TRAILING STOP "3 ÉTAGES" ---
                        drawdown_from_top = (highest - current_p) / highest

                        # ÉTAGE 1 : La Lune (Gain > +40%)
                        if highest >= entry * 1.40:
                            if drawdown_from_top >= 0.10:
                                portfolio.close_position(direction, current_p, f"Trailing Stop TIGHT (Sommet {highest:.2f})")
                                portfolio.print_stats()
                                continue

                        # ÉTAGE 2 : La Belle Perf (Gain > +20%)
                        elif highest >= entry * 1.20:
                            if drawdown_from_top >= 0.15:
                                if current_p > entry * 1.05: # On sort en profit net
                                    portfolio.close_position(direction, current_p, f"Trailing Stop MED (Sommet {highest:.2f})")
                                    portfolio.print_stats()
                                    continue

                        # ÉTAGE 3 : Le Décollage (Gain > +10%)
                        elif highest >= entry * 1.10:
                            break_even = entry * 1.04
                            if current_p <= break_even:
                                portfolio.close_position(direction, current_p, "Break-Even (Retour case départ)")
                                portfolio.print_stats()
                                continue

                        # --- B. STOP LOSS STANDARD ---
                        # On accepte -15% de perte max sur les trades normaux
                        if current_p <= entry * 0.85:
                            portfolio.close_position(direction, current_p, "Stop Loss (-15%)")
                            portfolio.print_stats()
                            continue
                            
                        # --- C. TIME STOP ---
                        if time_to_maturity < 45.0:
                            if 0.25 < current_p < 0.75:
                                portfolio.close_position(direction, current_p, "Time Stop (Incertain)")
                                portfolio.print_stats()
                                continue

                    # ==========================================
                    # 3. GESTION DES ENTRÉES AVEC CONFIRMATION TEMPORELLE
                    # ==========================================
                    if not portfolio.positions:
                        
                        # 1. On demande au modèle ce qu'il en pense MAINTENANT
                        action, taille_mise, raison = calculate_trade_signal(
                            prix_up, prix_down, fair_up, fair_dn, tau_years, portfolio.balance
                        )
                        
                        # A. SI AUCUN SIGNAL (HOLD)
                        if action == "HOLD":
                            if pending_signal["direction"] is not None:
                                print(f"❌ Signal {pending_signal['direction']} perdu ! (Bruit de marché)")
                                # On remet le chrono à zéro
                                pending_signal = {"direction": None, "start_time": None}
                        
                        # B. SI SIGNAL DÉTECTÉ (BUY_UP ou BUY_DOWN)
                        else:
                            current_direction = action.split("_")[1] # "UP" ou "DOWN"
                            now = time.time()
                            
                            # Cas 1 : C'est un nouveau signal
                            if pending_signal["direction"] != current_direction:
                                print(f"⏳ Signal {current_direction} détecté... En attente de confirmation ({CONFIRMATION_DELAY}s)")
                                pending_signal["direction"] = current_direction
                                pending_signal["start_time"] = now
                                
                            # Cas 2 : C'est le même signal qui persiste
                            else:
                                elapsed = now - pending_signal["start_time"]
                                remaining = CONFIRMATION_DELAY - elapsed
                                
                                if elapsed >= CONFIRMATION_DELAY:
                                    # ✅ VALIDATION FINALE ! Le signal est stable depuis 5s
                                    print(f"🚀 SIGNAL CONFIRMÉ ({elapsed:.1f}s) ! EXÉCUTION IMMÉDIATE.")
                                    
                                    if current_direction == "UP":
                                        portfolio.open_position("UP", prix_up, taille_mise)
                                    else:
                                        portfolio.open_position("DOWN", prix_down, taille_mise)
                                        
                                    # On reset le chrono après le tir
                                    pending_signal = {"direction": None, "start_time": None}
                                    
                                else:
                                    # On attend encore un peu...
                                    # Optionnel : Afficher un décompte pour debug
                                    # print(f"⏳ Confirmation dans {remaining:.1f}s...")
                                    pass
                            
                except Exception as e:
                    print(f"⚠️ Erreur tick: {e}")
            else:
                # DEBUG : On affiche ce qui bloque le démarrage
                status_deribit = "✅" if latest_deribit_dataset is not None else "❌"
                status_strike = f"✅ ({strike_k})" if strike_k is not None else "❌"
                status_spot = f"✅ ({spot_price})" if spot_price is not None else "❌"
                
                print(f"⏳ En attente des flux... Deribit:{status_deribit} | Strike:{status_strike} | Spot:{status_spot}")

if __name__ == "__main__":
    try:
        asyncio.run(orchestrator())
    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel du Bot.")