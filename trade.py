import time
from datetime import datetime

class PaperTrader:
    def __init__(self, initial_balance=1000.0, recorder=None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.history = []
        self.recorder = recorder  # <--- AJOUT: Le lien vers le CSV
        
        # --- COMPTEURS ---
        self.total_fees = 0.0          
        self.force_closed_count = 0    

    def update_price(self, direction, price):
        """Met à jour le dernier prix ET le plus haut prix historique du trade."""
        if direction in self.positions:
            self.positions[direction]["last_price"] = price
            
            # On retient le sommet (High Water Mark)
            current_high = self.positions[direction].get("highest_price", 0.0)
            if price > current_high:
                self.positions[direction]["highest_price"] = price

    def open_position(self, direction, price, amount_usdc, metadata=None):
        # ... (vérifications solde simplifiées ici pour la clarté) ...
        if amount_usdc > self.balance:
            print("❌ Solde insuffisant.")
            return False, "Insufficient Balance"

        # --- CALCUL DES FRAIS (MAKER = 0) ---
        fee_cost = 0.0 
        self.balance -= amount_usdc 

        shares = amount_usdc / price 
        
        self.positions[direction] = {
            "entry_price": price,
            "last_price": price,
            "highest_price": price,
            "shares": shares,
            "invested": amount_usdc,
            "highest_fair": metadata.get("sigma", 0),
            "start_time": time.time(),
            "type": "MAKER",
            "metadata": metadata if metadata else {} # <--- AJOUT: Stockage des données ML
        }
        
        print(f"🟢 [MAKER BUY] {direction} @ {price:.3f} | Frais: 0.00$")
        return True, "Success"

    def close_position(self, direction, current_price, reason, exit_spot=None):
        if direction not in self.positions:
            return False
            
        pos = self.positions.pop(direction)
        
        # Récupération pour le logger
        meta = pos.get("metadata", {})
        
        revenue_brut = pos["shares"] * current_price
        fee_cost = 0.0
        
        # --- CAS 1 : EXPIRATION (GRATUIT) ---
        if "EXPIRATION" in reason or "Settlement" in reason:
            fee_cost = 0.0
            
        # --- CAS 2 : VENTE ACTIVE (PAYANT) ---
        else:
            # Formule Polymarket (approx)
            fee_rate = 0.0156 * 4 * current_price * (1.0 - current_price)
            fee_cost = revenue_brut * fee_rate
            self.total_fees += fee_cost

        # --- CALCUL NET ---
        revenue_net = revenue_brut - fee_cost 
        pnl = revenue_net - pos["invested"]
        roi = (pnl / pos["invested"]) * 100
        
        # Mise à jour du solde
        self.balance += revenue_net
        
        # Compteurs
        if "Watchdog" in reason:
            self.force_closed_count += 1

        self.history.append({
            "direction": direction,
            "pnl": pnl,
            "reason": reason,
            "fee": fee_cost 
        })
        
        icon = "🤑" if pnl > 0 else "🩸"
        print(f"🔴 [CLOSE] {direction} @ {current_price:.3f} | {icon} PnL: {pnl:+.2f}$ ({roi:+.1f}%) | Frais: {fee_cost:.3f}$ | {reason}")

        # --- AJOUT: ENREGISTREMENT CSV ---
        if self.recorder:
            trade_record = {
                "timestamp_entry": datetime.fromtimestamp(pos["start_time"]),
                "timestamp_exit": datetime.now(),
                "direction": direction,
                "entry_price": pos["entry_price"],
                "exit_price": current_price,
                
                # ICI C'ETAIT FAUX AVANT : On aligne avec les clés de main.py
                "entry_spot": meta.get("entry_spot", 0),  # "entry_spot" et non "spot"
                "exit_spot": exit_spot if exit_spot else 0,
                
                "sigma_pred": meta.get("sigma_pred", 0),  # "sigma_pred" et non "sigma"
                "bid_entry": meta.get("bid_entry", 0),    # "bid_entry" et non "bid"
                "ask_entry": meta.get("ask_entry", 0),    # "ask_entry" et non "ask"
                "x_live_vector": str(meta.get("x_live_vector", [])), # "x_live_vector" et non "x_live"
                
                "stake": pos["invested"],
                "pnl_usd": round(pnl, 4),
                "roi_pct": round(roi, 2),
                "exit_reason": reason
            }
            self.recorder.record(trade_record)
        
        return True

    def print_stats(self):
        total_trades = len(self.history)
        if total_trades == 0: return
            
        winning_trades = [t for t in self.history if t["pnl"] > 0]
        losing_trades = [t for t in self.history if t["pnl"] <= 0]
        
        winning_count = len(winning_trades)
        win_rate = (winning_count / total_trades) * 100
        
        net_profit = self.balance - self.initial_balance
        # On s'assure que les fees sont bien déduites (le code original le faisait déjà via self.balance)
        # Mais pour l'affichage pur PnL vs Fees :
        
        # --- AJOUT: CALCULS MOYENNES ---
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

        print("\n" + "="*40)
        print("📊 STATISTIQUES AVANCÉES")
        print("="*40)
        print(f"Trades Totaux      : {total_trades}")
        print(f"Dont Force Close   : {self.force_closed_count} ⚠️")
        print(f"Win Rate           : {win_rate:.1f}%")
        print(f"Profit Moyen (Win) : {avg_win:+.2f} $")  # <--- AJOUT
        print(f"Perte Moyenne (Loss): {avg_loss:+.2f} $") # <--- AJOUT
        print("-" * 40)
        print(f"Frais Payés        : {self.total_fees:.2f} $")
        print(f"PnL Net (Compte)   : {net_profit:+.2f} $")
        print("="*40 + "\n")

def calculate_trade_signal(bid_up, bid_down, fair_up, fair_dn, tau_years, bankroll):
    """
    Version MAKER (0% Frais) :
    On se place du côté du Bid (Acheteur). On ne paye AUCUN frais.
    """
    # 1. Filtres temporels (Inchangés)
    tau_minutes = tau_years * 365.25 * 24 * 60
    if tau_minutes < 2.0 or tau_minutes > 14.0:
        return "HOLD", 0.0, "Timing mauvais"
        
    # 2. Paramètres
    # Comme on ne paye pas de frais et qu'on achète moins cher (au Bid),
    # on peut être beaucoup plus agressif sur l'Edge.
    MIN_EDGE = 0.06 # On accepte un edge plus petit car on a 0 frais !
    MIN_PRICE = 0
    MAX_PRICE = 0.80
    KELLY_FRACTION = 0.5
    BASE_ALLOCATION = 20.0
    
    # --- 3. FRAIS = 0 (C'est ICI que ça change) ---
    # En Maker, Polymarket ne prend RIEN.
    fee_up = 0.0
    fee_down = 0.0
    
    # Le prix effectif est juste le prix de l'ordre limite qu'on pose
    # Stratégie : On se met au niveau du Meilleur Bid actuel (on rejoint la file)
    target_price_up = bid_up
    target_price_down = bid_down
    if  not (MIN_PRICE <= target_price_up <= MAX_PRICE) and not (MIN_PRICE <= target_price_down <= MAX_PRICE) and not (fair_up>=0.9 or fair_dn>=0.9):
        return "HOLD", 0.0, "Prix hors zone de confort (Trop extrême)"
    
    # --- 4. CALCUL DE L'EDGE ---
    # Edge = Ma Fair Value - Mon Prix Limite (Bid)
    edge_up = (fair_up - target_price_up) if target_price_up > 0 else 0
    edge_down = (fair_dn - target_price_down) if target_price_down > 0 else 0
    
    # --- 5. DÉCISION ---
    if edge_up > MIN_EDGE and edge_up > edge_down:
        if target_price_up < 1.0:
            if fair_up < 0.30: return "HOLD", 0.0, "Conviction faible"

            kelly_pct = (fair_up - target_price_up) / (1.0 - target_price_up)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            # On renvoie "MAKER_BUY_UP" pour bien différencier
            return "MAKER_BUY_UP", bet_size, f"Maker UP | 0 Frais | Fair: {fair_up:.2f} vs Bid: {target_price_up:.3f}"
        
    elif edge_down > MIN_EDGE:
        if target_price_down < 1.0:
            if fair_dn < 0.30: return "HOLD", 0.0, "Conviction faible"

            kelly_pct = (fair_dn - target_price_down) / (1.0 - target_price_down)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            return "MAKER_BUY_DOWN", bet_size, f"Maker DOWN | 0 Frais | Fair: {fair_dn:.2f} vs Bid: {target_price_down:.3f}"
        
    return "HOLD", 0.0, "Pas d'opportunité Maker"