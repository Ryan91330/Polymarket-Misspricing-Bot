import asyncio
import time
import xgboost as xgb
# Importation de tes scripts
from deribit_live import fetch_options_dataset
from polymarket_live import watch_spot_price, watch_clob_market
from feature_engine import FeatureEngine
from pricing import BS_Bin,BS_Bin_Skew
from trade import calculate_trade_signal,PaperTrader
from trade_logger import TradeRecorder

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
    live_xgb.load_model("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/models/best_xgb_model_r_4h_v3_s.json")
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
    recorder = TradeRecorder()
    portfolio = PaperTrader(initial_balance=500.0,recorder=recorder)
    print("🎧 En écoute des flux live...")
    asyncio.create_task(expiration_watchdog(portfolio))
    
    # ⏱️ PARAMÈTRE DE PERSISTANCE
    CONFIRMATION_DELAY = 2  # Le signal doit rester valide 5 secondes pour tirer
    
    # État du chronomètre
    pending_signal = {
        "direction": None,  # "UP" ou "DOWN"
        "start_time": None
    }
    
    # --- NOUVEAU : GESTION DU COOLDOWN ---
    # Stocke le timestamp jusqu'auquel le trading est interdit pour une direction
    trading_cooldowns = {
        "UP": 0.0,
        "DOWN": 0.0
    }
    COOLDOWN_DURATION = 150
    
    # 4. Boucle de consommation des événements
    while True:
        # On attend qu'une des tâches envoie une donnée
        source, data = await data_queue.get()
        
        if source == "deribit":
            # On met à jour la "photo" du marché des options
            latest_deribit_dataset = data
            #print(f"[DERIBIT] Surface de volatilité mise à jour ({len(data)} options en mémoire).")
            
        elif source == "polymarket":
            # On récupère le dictionnaire riche qu'on vient de créer
            data_poly = data 
            
            # 1. On extrait les données précises
            bid_up = data_poly.get("bid_up", 0.0)
            ask_up = data_poly.get("ask_up", 0.0)
            bid_dn = 1 - ask_up
            ask_dn = 1 - bid_up
            strike_k = data_poly.get("strike")
            time_to_maturity = data_poly.get("tau")
            spot_price = data_poly.get("spot")

            # --- 2. FILTRE DE SPREAD (CRUCIAL) ---
            # Si l'écart entre le vendeur et l'acheteur est trop grand, le marché est "toxique".
            # On ne trade pas.
            spread_up = ask_up - bid_up
            spread_dn = ask_dn - bid_dn
        
           
            MAX_SPREAD = 0.06 # 6 centimes max d'écart accepté
            
            market_is_healthy = True
            if spread_up > MAX_SPREAD and spread_dn > MAX_SPREAD:
                #print(f"⚠️ Spread trop large ({spread_up:.2f}), on ignore.")
                market_is_healthy = False

            # --- 3. MISE À JOUR DU PORTFOLIO (VALORISATION) ---
            # Si on possède des parts, elles valent ce que les acheteurs offrent (BID)
            # C'est la "Mark-to-Market" valuation.
            if "UP" in portfolio.positions:
                portfolio.update_price("UP", bid_up) 
            if "DOWN" in portfolio.positions:
                portfolio.update_price("DOWN", bid_dn)

            # --- 4. CALCUL DU SIGNAL DE TRADING ---
            if latest_deribit_dataset is not None and strike_k is not None and market_is_healthy:
                try:
                    tau_years = time_to_maturity / (365.25 * 24 * 3600)
                    
                    # 1. Prédiction XGBoost Standard (au Strike K)
                    X_live = engine.get_live_vector(spot_price, strike_k, tau_years)
                    sigma = float(live_xgb.predict(X_live)[0])
                    r_live = float(X_live[0][2])
                    
                    # --- CORRECTION DU SKEW : DIFFÉRENCE CENTRALE + LISSAGE ---
                    # Au lieu de K+100, on prend K-500 et K+500 pour "gommer" les marches d'escalier de XGBoost
                    # delta_k = 500.0 
                    # X_live_plus = engine.get_live_vector(spot_price, strike_k + delta_k, tau_years)
                    # X_live_minus = engine.get_live_vector(spot_price, strike_k - delta_k, tau_years)
                    
                    # sigma_plus = float(live_xgb.predict(X_live_plus)[0])
                    # sigma_minus = float(live_xgb.predict(X_live_minus)[0])
                    
                    # # Calcul de la pente sur une large zone (Central Difference)
                    # raw_skew_slope = (sigma_plus - sigma_minus) / (2 * delta_k)
                    
                    # # --- SÉCURITÉ CRITIQUE : LE CLAMPING ---
                    # # Un vrai "Skew Slope" sur le BTC dépasse rarement 0.00005 par dollar.
                    # # On bloque la valeur pour empêcher le Vega d'exploser le prix binaire.
                    # MAX_SKEW_SLOPE = 0.00005
                    # skew_slope = max(-MAX_SKEW_SLOPE, min(MAX_SKEW_SLOPE, raw_skew_slope))
                    # print(raw_skew_slope)
                    # # 2. Pricing Binaire avec Ajustement du Skew sécurisé
                    # fair_up, fair_dn = BS_Bin_Skew(spot_price, r_live, strike_k, sigma, tau_years,raw_skew_slope)
                    fair_up, fair_dn = BS_Bin(spot_price, r_live, strike_k, sigma, tau_years)
                    # --- MODE MAKER : ON UTILISE LE BID ---
                    # Pour l'entrée : C'est notre prix cible (Limit Order).
                    # Pour la sortie : C'est le prix de vente immédiat (Market Sell).
                    prix_up_sortie = bid_up if bid_up > 0 else 0.001
                    prix_down_sortie = bid_dn if bid_dn > 0 else 0.001
                    prix_up_entre = ask_up if ask_up > 0 else 0.001
                    prix_down_entre = ask_dn if ask_dn > 0 else 0.001
                    print(f"📉 SPOT: {spot_price:.1f} | 🎯 K: {strike_k:.1f} | ⏳ {tau_years*365*24*60:.1f}m || "
                              f"🟢 UP: {prix_up_entre:.3f} (Fair {fair_up:.3f}) | "
                              f"🔴 DN: {prix_down_entre:.3f} (Fair {fair_dn:.3f})")
                    # ... (Suite : Exécution du trade, inchangée) ...
                    
                    # ==========================================
                # 2. GESTION DES SORTIES (BASÉE SUR LA FAIR VALUE)
                # ==========================================
                
                    active_pos_keys = list(portfolio.positions.keys())
                    now = time.time()
                    
                    for direction in active_pos_keys:
                        pos = portfolio.positions[direction]
                        
                        entry = pos["entry_price"]
                        start_t = pos["start_time"]
                        highest = pos["highest_price"]
                        
                        # 1. Mise à jour des données Live
                        current_p = prix_up_sortie if direction == "UP" else prix_down_sortie
                        target_fv = fair_up if direction == "UP" else fair_dn
                        
                        # --- MEMOIRE DE LA FAIR VALUE (Pour le FV Trailing) ---
                        # On initialise ou met à jour la FV la plus haute vue durant ce trade
                        if "highest_fv" not in pos:
                            pos["highest_fv"] = target_fv
                        if target_fv > pos["highest_fv"]:
                            pos["highest_fv"] = target_fv
                        
                        entry_fv = pos.get("metadata", {}).get("fair_value", entry)
                        highest_fv = pos["highest_fv"] # Variable locale pour lisibilité

                        # PROTECTION TEMPORELLE (5s)
                        if (now - start_t) < 5.0:
                            continue

                        # ==============================================================================
                        # BLOC 1 : SCÉNARIOS GAGNANTS (TAKE PROFIT)
                        # Priorité absolue : Si on a de l'argent et que les conditions sont là, on prend.
                        # ==============================================================================

                        # Calculs pour la zone
                        proximity_zone_low,proximity_zone_high = entry_fv * 0.95, entry_fv*1.05
                        dd_from_top = (highest - current_p) / highest
                        reason = None
                        if current_p >= 0.95 :
                            reason ="MAX PROFIT"
                            portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)
                            
                        # A. SCÉNARIO IDÉAL : On dépasse la Fair Value + Profit minimum
                        if current_p >= proximity_zone_high and current_p >= entry * 1.02:
                            if dd_from_top > 0.01: 
                                reason = f"TAKE PROFIT (FV Dépassée + Reversal)"
                                portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)


                        # B. SCÉNARIO "LOCK-IN" : On est dans la Zone (95% de la FV) mais ça faiblit
                        if current_p <= proximity_zone_high :
                            if current_p >= proximity_zone_low and current_p >= entry * 1.02:
                                if dd_from_top > 0.01: 
                                    reason = f"LOCK PROFIT (Zone FV atteinte mais rejet)"
                                    portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)


                        # ==============================================================================
                        # BLOC 2 : SCÉNARIOS D'INVALIDATION (STOP LOSS INTELLIGENT VIA MODEL)
                        # Le modèle nous dit de sortir AVANT que le prix ne touche le Hard Stop.
                        # ==============================================================================

                        # C. THESIS INVALIDATION (La FV passe sous le prix d'entrée)
                        # Le modèle pensait que ça valait 0.60 (Entry 0.50). Maintenant il dit 0.48.
                        # On sort immédiatement, car la raison fondamentale du trade a disparu.
                        if target_fv < entry:
                            reason = "FV INVALIDATION (FV < Entry)"
                            portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)
                            trading_cooldowns[direction] = time.time() + COOLDOWN_DURATION
                            print(f"❄️ COOLDOWN ACTIVÉ sur {direction} pendant {COOLDOWN_DURATION}s (Modèle invalide)")


                        # D. FV TRAILING STOP (Le modèle devient pessimiste)
                        # La FV était montée à 0.70, elle retombe à 0.60 (-14%).
                        # Le modèle détecte un changement de régime (volatilité, spot...) -> On sort.
                        fv_drawdown = (highest_fv - target_fv) / highest_fv
                        if fv_drawdown > 0.15: # Si la FV perd 10% depuis son sommet
                            reason = "FV REVERSAL (Modèle pessimiste)"
                            portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)
                            trading_cooldowns[direction] = time.time() + COOLDOWN_DURATION
                            print(f"❄️ COOLDOWN ACTIVÉ sur {direction} pendant {COOLDOWN_DURATION}s (Modèle invalide)")


                        # E. EDGE COMPRESSION (Prix = FV, mais sans profit suffisant)
                        # Le prix a rejoint la FV, mais la FV a baissé entre temps.
                        # Ex: Acheté 0.50, FV était 0.60. Maintenant Prix 0.52, FV 0.52.
                        # Il n'y a plus de "marge" (Edge) à gagner. On sort flat/léger gain pour libérer le capital.
                        edge = target_fv - current_p
                        if edge < 0.01 and current_p < entry * 1.02:
                            reason = "EDGE GONE (Plus de potentiel mathématique)"
                            portfolio.close_position(direction, current_p,reason, exit_spot=spot_price)


                        # ==============================================================================
                        # BLOC 3 : SCÉNARIOS DE PROTECTION (STOP LOSS CLASSIQUE)
                        # Dernier recours si le modèle est lent mais que le prix crash.
                        # ==============================================================================

                        # F. Break-Even (Si on a fait +10% puis revenu au prix d'entrée + marge)
                        if highest >= entry * 1.10 and current_p <= entry * 1.05:
                            reason = "BREAK-EVEN (Protection capital)"
                            portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)


                        # G. Stop Loss Classique (-15% max)
                        if current_p <= entry * 0.85:
                            reason = "STOP LOSS HARD"
                            portfolio.close_position(direction, current_p, reason, exit_spot=spot_price)
                            trading_cooldowns[direction] = time.time() + COOLDOWN_DURATION
                            print(f"❄️ COOLDOWN ACTIVÉ sur {direction} pendant {COOLDOWN_DURATION}s (Modèle invalide)")
                            

                        if reason is not None :
                            portfolio.print_stats()
                            continue
                        
                except Exception as e:
                    print(f"⚠️ Erreur Sortie: {e}")

                    # ==========================================
                    # 3. GESTION DES ENTRÉES AVEC CONFIRMATION (MODE MAKER)
                    # ==========================================
                try:
                    if not portfolio.positions:
                        
                        # 1. On demande au modèle ce qu'il en pense (Basé sur le BID)
                        # Note: prix_up et prix_down sont ici égaux aux BIDs (défini au bloc 4)
                        action, taille_mise, raison = calculate_trade_signal(
                            prix_up_entre, prix_down_entre, fair_up, fair_dn, tau_years, portfolio.balance
                        )
                        
                        # A. SI AUCUN SIGNAL (HOLD)
                        if action == "HOLD":
                            if pending_signal["direction"] is not None:
                                #print(f"❌ Signal {pending_signal['direction']} perdu ! (Bruit de marché)")
                                # On remet le chrono à zéro
                                pending_signal = {"direction": None, "start_time": None}
                        
                        # B. SI SIGNAL DÉTECTÉ (MAKER_BUY_UP ou MAKER_BUY_DOWN)
                        else:
                            # CORRECTION CRITIQUE ICI :
                            # Le signal est "MAKER_BUY_UP". 
                            # split("_") donne ['MAKER', 'BUY', 'UP']
                            # On veut le dernier élément : [-1]
                            current_direction = action.split("_")[-1] # "UP" ou "DOWN"
                            
                            now = time.time()
                            if now < trading_cooldowns.get(current_direction, 0):
                                #remaining = int(trading_cooldowns[current_direction] - now)
                                # On spamme pas le log, on affiche juste une fois de temps en temps si tu veux
                                # print(f"❄️ Signal {current_direction} ignoré (Cooldown actif: {remaining}s restants)")
                                
                                # ON FORCE LE HOLD
                                action = "HOLD" 
                                pending_signal = {"direction": None, "start_time": None}
                                
                            # Cas 1 : C'est un nouveau signal
                            if pending_signal["direction"] != current_direction:
                                print(f"⏳ Signal {current_direction} détecté... Attente stabilité ({CONFIRMATION_DELAY}s)")
                                pending_signal["direction"] = current_direction
                                pending_signal["start_time"] = now
                                
                            # Cas 2 : C'est le même signal qui persiste
                            else:
                                elapsed = now - pending_signal["start_time"]
                                
                                if elapsed >= CONFIRMATION_DELAY:
                                    print(f"🚀 SIGNAL MAKER CONFIRMÉ ({elapsed:.1f}s) !")
                                    
                                    # Nettoyage variables pour CSV
                                    clean_sigma = float(sigma) if isinstance(sigma, (float, int)) else float(sigma[0])
                                    clean_x_vector = X_live[0].tolist() if hasattr(X_live[0], 'tolist') else list(X_live[0])

                                    # Fair Value Initiale
                                    initial_fv = float(fair_up) if current_direction == "UP" else float(fair_dn)
                                    
                                    meta_data = {
                                        "entry_spot": float(spot_price) if spot_price else 0.0,
                                        "sigma_pred": clean_sigma,
                                        "x_live_vector": clean_x_vector, 
                                        "bid_entry": float(bid_up) if current_direction == "UP" else float(bid_dn),
                                        "ask_entry": float(ask_up) if current_direction == "UP" else float(ask_dn),
                                        "fair_value": initial_fv 
                                    }

                                    if current_direction == "UP":
                                        portfolio.open_position("UP", prix_up_entre, taille_mise, metadata=meta_data)
                                    else:
                                        portfolio.open_position("DOWN", prix_down_entre, taille_mise, metadata=meta_data)
                                    
                                        
                                    # On reset le chrono après le tir
                                    pending_signal = {"direction": None, "start_time": None}
                                    
                                else:
                                    # On attend encore un peu...
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