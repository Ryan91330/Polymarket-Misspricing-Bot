import time
from datetime import datetime

class PaperTrader:
    def __init__(self, initial_balance=1000.0, recorder=None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.pending_orders = []  # <--- AJOUT: File d'attente des ordres Maker
        self.history = []
        self.recorder = recorder
        
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

    def place_post_only_order(self, direction, limit_price, amount_usdc, metadata=None):
        """
        Simule le placement d'un ordre Maker.
        🔒 VERROUILLAGE DES FONDS : On débite l'argent TOUT DE SUITE.
        """
        if amount_usdc > self.balance:
            print(f"❌ Solde insuffisant ({self.balance:.2f}$) pour placer l'ordre ({amount_usdc:.2f}$).")
            return False

        # On vérifie qu'on n'a pas déjà une position
        direction_clean = direction.replace("MAKER_BUY_", "")
        if direction_clean in self.positions:
            return False 

        # Nettoyage des ordres précédents (Annule et Remplace)
        # ⚠️ IMPORTANT : Si on annule un ordre précédent, on doit REMBOURSER son montant !
        for old_order in self.pending_orders:
            if old_order['direction'] == direction:
                self.balance += old_order['amount'] # 💰 Remboursement
                print(f"♻️ Ordre précédent remboursé (+{old_order['amount']:.2f}$)")
        
        # On garde seulement les ordres des autres directions
        self.pending_orders = [o for o in self.pending_orders if o['direction'] != direction]

        # 🔒 DÉBIT IMMÉDIAT
        self.balance -= amount_usdc

        order = {
            "id": f"{direction}_{int(time.time()*1000)}",
            "direction": direction,
            "limit_price": limit_price,
            "amount": amount_usdc,
            "timestamp": time.time(),
            "metadata": metadata,
            "status": "OPEN"
        }
        
        self.pending_orders.append(order)
        print(f"⏳ [MAKER PENDING] Ordre {direction_clean} placé @ {limit_price:.3f}$ | Solde bloqué: {amount_usdc:.2f}$")
        return True

    def process_pending_orders(self, current_ask_up, current_bid_up, current_ask_down, current_bid_down):
        """
        MOTEUR DE MATCHING PESSIMISTE + GESTION TTL & REMBOURSEMENT
        """
        ORDER_TIMEOUT = 5.0 # 5 secondes de vie
        now = time.time()

        for order in self.pending_orders[:]:
            is_filled = False
            fill_price = 0.0
            trigger_reason = ""
            
            # --- 1. VÉRIFICATION D'EXPIRATION (TTL) ---
            if (now - order["timestamp"]) > ORDER_TIMEOUT:
                # 💰 REMBOURSEMENT AUTOMATIQUE
                self.balance += order["amount"]
                print(f"🗑️ [TIMEOUT] Ordre {order['direction']} annulé & remboursé (+{order['amount']:.2f}$)")
                self.pending_orders.remove(order)
                continue
            
            # --- 2. LOGIQUE PESSIMISTE (Avec Preuve) ---
            if order["direction"] == "MAKER_BUY_UP":
                # Condition: Ask descend sur notre Bid
                if current_ask_up < order["limit_price"]:
                    is_filled = True
                    fill_price = order["limit_price"]
                    trigger_reason = f"Ask ({current_ask_up}) a croisé notre Bid ({order['limit_price']})"
            
            elif order["direction"] == "MAKER_BUY_DOWN":
                if current_ask_down < order["limit_price"]:
                    is_filled = True
                    fill_price = order["limit_price"]
                    trigger_reason = f"Ask ({current_ask_down}) a croisé notre Bid ({order['limit_price']})"

            # --- 3. EXÉCUTION ---
            if is_filled:
                print(f"⚡ [EXECUTION MAKER] {order['direction']} @ {fill_price:.3f}$")
                print(f"   ↳ Preuve: {trigger_reason}")
                
                # Note: On ne débite plus le solde ici car c'est déjà fait au placement !
                self._convert_order_to_position(order, fill_price)
                self.pending_orders.remove(order)

    def _convert_order_to_position(self, order, price):
        """Transforme l'ordre en position. L'argent est DÉJÀ débité."""
        amount_usdc = order["amount"]
        
        # Frais Maker = 0
        shares = amount_usdc / price 
        
        direction_clean = order["direction"].replace("MAKER_BUY_", "")
        
        self.positions[direction_clean] = {
            "entry_price": price,
            "last_price": price,
            "highest_price": price,
            "shares": shares,
            "invested": amount_usdc,
            "highest_fair": order["metadata"].get("sigma_pred", 0) if order["metadata"] else 0,
            "start_time": time.time(),
            "type": "MAKER",
            "metadata": order["metadata"] if order["metadata"] else {}
        }
        # Pas de print ici, déjà fait dans process_pending_orders pour la clarté
    def cancel_specific_order(self, order, reason=""):
        """Annule un ordre spécifique et rembourse le solde."""
        if order in self.pending_orders:
            self.balance += order["amount"] # 💰 Remboursement
            self.pending_orders.remove(order)
            print(f"❌ [CANCEL] Ordre {order['direction']} annulé | Raison: {reason}")
            return True
        return False
    
    def close_position(self, direction, current_price, reason, exit_spot=None):
        """
        Ferme une position. 
        Note: On reste en mode TAKER (Market Sell) pour les sorties de sécurité.
        On paye donc les frais de sortie.
        """
        if direction not in self.positions:
            return False
            
        pos = self.positions.pop(direction)
        
        # On annule aussi tout ordre en attente inverse s'il y en avait (nettoyage)
        self.pending_orders = [o for o in self.pending_orders if o['direction'] != f"MAKER_BUY_{direction}"]

        # Récupération pour le logger
        meta = pos.get("metadata", {})
        
        revenue_brut = pos["shares"] * current_price
        fee_cost = 0.0
        
        # --- CAS 1 : EXPIRATION (GRATUIT) ---
        if "EXPIRATION" in reason or "Settlement" in reason:
            fee_cost = 0.0
            
        # --- CAS 2 : VENTE ACTIVE (PAYANT - TAKER FEE) ---
        # On assume qu'on sort en urgence (Stop Loss / Take Profit immédiat) -> TAKER
        else:
            # Formule officielle Polymarket Crypto (Mars 2026)
            c = pos["shares"]
            p = current_price
            
            raw_fee = c * p * 0.25 * ((p * (1.0 - p)) ** 2)
            fee_cost = round(raw_fee, 4)
            
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

        # --- ENREGISTREMENT CSV ---
        if self.recorder:
            trade_record = {
                "timestamp_entry": datetime.fromtimestamp(pos["start_time"]),
                "timestamp_exit": datetime.now(),
                "direction": direction,
                "entry_price": pos["entry_price"],
                "exit_price": current_price,
                "entry_spot": meta.get("entry_spot", 0),
                "exit_spot": exit_spot if exit_spot else 0,
                "sigma_pred": meta.get("sigma_pred", 0),
                "bid_entry": meta.get("bid_entry", 0),
                "ask_entry": meta.get("ask_entry", 0),
                "x_live_vector": str(meta.get("x_live_vector", [])),
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
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

        print("\n" + "="*40)
        print("📊 STATISTIQUES AVANCÉES (MODE MAKER/TAKER HYBRIDE)")
        print("="*40)
        print(f"Trades Totaux      : {total_trades}")
        print(f"Dont Force Close   : {self.force_closed_count} ⚠️")
        print(f"Win Rate           : {win_rate:.1f}%")
        print(f"Profit Moyen (Win) : {avg_win:+.2f} $")
        print(f"Perte Moyenne (Loss): {avg_loss:+.2f} $")
        print("-" * 40)
        print(f"Frais Payés (Sortie): {self.total_fees:.2f} $")
        print(f"PnL Net (Compte)   : {net_profit:+.2f} $")
        print("="*40 + "\n")

def calculate_trade_signal(bid_up, bid_down, fair_up, fair_dn, tau_years, bankroll):
    """
    Version MAKER (Post-Only) :
    On se place du côté Acheteur (Bid) ou juste devant.
    - Frais = 0.00$ (On ignore le rebate par prudence)
    - Spread = 0.00$ (On ne le traverse pas)
    """
    # 1. Filtres temporels
    tau_minutes = tau_years * 365.25 * 24 * 60
    if tau_minutes < 2.0 or tau_minutes > 14.0:
        return "HOLD", 0.0, "Timing mauvais"
        
    # 2. Paramètres (Optimisés pour Maker)
    # On n'a plus besoin de couvrir 3% de frais + spread.
    # Un edge de 2.5% est déjà excellent en Maker.
    MIN_EDGE_up = 0.09 
    MIN_EDGE_dn = 0.065
    MIN_PRICE = 0.02 # On évite juste les déchets à 1 centime
    MAX_PRICE = 0.8 # On peut aller haut car le Settlement est gratuit
    
    KELLY_FRACTION = 0.3 # Un peu plus prudent sur la taille car on est en "Limit"
    BASE_ALLOCATION = 20.0
    
    # --- 3. LOGIQUE MAKER : On vise le BID ---
    target_price_up = bid_up
    target_price_down = bid_down
    
    # Filtre de prix basique
    if not (MIN_PRICE <= target_price_up <= MAX_PRICE) and not (MIN_PRICE <= target_price_down <= MAX_PRICE):
        return "HOLD", 0.0, "Prix hors zone"
    
    # Sécurité Fair Value extrême
    if (fair_up >= 0.9 or fair_dn >= 0.9):
        return "HOLD", 0.0, "Fair value max atteinte"

    # --- 4. CALCUL DE L'EDGE BRUT (Frais = 0) ---
    # En Taker, on faisait : (fair - price - tax).
    # En Maker, c'est : (fair - price). C'est tout.
    
    edge_up = (fair_up - target_price_up) if target_price_up > 0 else 0
    edge_down = (fair_dn - target_price_down) if target_price_down > 0 else 0
    
    # --- 5. DÉCISION ---
    if edge_up > MIN_EDGE_up and edge_up > edge_down:
        if target_price_up < 1.0:
            # Filtre de conviction minimum
            if fair_up < 0.30: return "HOLD", 0.0, "Conviction faible"

            # Kelly Criterion sur le prix Limit
            kelly_pct = (fair_up - target_price_up) / (1.0 - target_price_up)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            # Note : On renvoie MAKER_BUY_UP
            return "MAKER_BUY_UP", bet_size, f"Maker UP | Bid: {target_price_up:.3f} | Edge: {edge_up:.3f}"
        
    elif edge_down > MIN_EDGE_dn:
        if target_price_down < 1.0:
            if fair_dn < 0.30: return "HOLD", 0.0, "Conviction faible"

            kelly_pct = (fair_dn - target_price_down) / (1.0 - target_price_down)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            return "MAKER_BUY_DOWN", bet_size, f"Maker DOWN | Bid: {target_price_down:.3f} | Edge: {edge_down:.3f}"
        
    return "HOLD", 0.0, f"Edge Maker insuffisant (Max: {max(edge_up, edge_down):.3f})"