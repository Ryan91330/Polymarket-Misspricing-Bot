import time
class PaperTrader:
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.history = []
        
        # --- NOUVEAUX COMPTEURS ---
        self.total_fees = 0.0          # Total des frais payés
        self.force_closed_count = 0    # Nombre de fermetures d'urgence

    def update_price(self, direction, price):
        """Met à jour le dernier prix ET le plus haut prix historique du trade."""
        if direction in self.positions:
            self.positions[direction]["last_price"] = price
            
            # --- NOUVEAU : On retient le sommet (High Water Mark) ---
            current_high = self.positions[direction].get("highest_price", 0.0)
            if price > current_high:
                self.positions[direction]["highest_price"] = price

    def open_position(self, direction, price, amount_usdc):
        if direction in self.positions:
            return False, "Position déjà ouverte."
        if self.balance < amount_usdc:
            amount_usdc = self.balance
        if amount_usdc <= 0:
            return False, "Fonds insuffisants."

        # Calcul des frais à l'entrée (Taker Fee)
        # Formule : 1.56% * 4 * P * (1-P)
        fee_rate = 0.0156 * 4 * price * (1.0 - price)
        fee_cost = amount_usdc * fee_rate
        
        self.total_fees += fee_cost
        
        # On déduit le montant investi du solde (les frais sont inclus dans le prix d'exécution en réalité, 
        # mais ici on les compte à part pour la stat)
        self.balance -= amount_usdc 

        shares = amount_usdc / price
        
        self.positions[direction] = {
            "entry_price": price,
            "last_price": price,    # <--- On initialise le dernier prix connu
            "shares": shares,
            "highest_price": price,
            "invested": amount_usdc,
            "start_time": time.time()
        }
        
        print(f"🟢 [PAPER BUY] {direction} | {amount_usdc:.2f}$ @ {price:.3f} | Frais: {fee_cost:.3f}$")
        return True, "Success"

    def close_position(self, direction, current_price, reason):
        if direction not in self.positions:
            return False
            
        pos = self.positions.pop(direction)
        
        # Calcul des frais à la sortie
        fee_rate = 0.0156 * 4 * current_price * (1.0 - current_price)
        revenue_brut = pos["shares"] * current_price
        fee_cost = revenue_brut * fee_rate
        
        self.total_fees += fee_cost
        
        revenue_net = revenue_brut # En simulation simple, on déduit les frais du PnL global
        pnl = revenue_net - pos["invested"]
        pnl_pct = (pnl / pos["invested"]) * 100
        
        self.balance += revenue_net
        
        # Si c'est un Force Close, on incrémente le compteur
        if "Watchdog" in reason or "Expiration" in reason:
            self.force_closed_count += 1

        self.history.append({
            "direction": direction,
            "pnl": pnl,
            "reason": reason
        })
        
        icon = "🤑" if pnl > 0 else "🩸"
        print(f"🔴 [PAPER SELL] {direction} @ {current_price:.3f} | {icon} PnL: {pnl:+.2f}$ | Frais Sortie: {fee_cost:.3f}$ | {reason}")
        return True

    def print_stats(self):
        total_trades = len(self.history)
        if total_trades == 0: return
            
        winning = len([t for t in self.history if t["pnl"] > 0])
        win_rate = (winning / total_trades) * 100
        net_profit = self.balance - self.initial_balance
        
        # On soustrait les frais du profit net affiché pour être réaliste
        net_profit_after_fees = net_profit - self.total_fees

        print("\n" + "="*40)
        print("📊 STATISTIQUES AVANCÉES")
        print("="*40)
        print(f"Trades Totaux    : {total_trades}")
        print(f"Dont Force Close : {self.force_closed_count} ⚠️")
        print(f"Win Rate         : {win_rate:.1f}%")
        print(f"Frais Payés      : {self.total_fees:.2f} $")
        print(f"PnL Net (Apres Fees) : {net_profit_after_fees:+.2f} $")
        print("="*40 + "\n")

def calculate_trade_signal(prix_up, prix_down, fair_up, fair_dn, tau_years, bankroll):
    """
    Version SNIPER : Plus sélective, moins de frais, meilleure espérance de gain.
    """
    # --- 1. FILTRE TEMPOREL (Inchangé) ---
    tau_minutes = tau_years * 365.25 * 24 * 60
    if tau_minutes < 2.0 :
        return "HOLD", 0.0, f"Trop proche expiration ({tau_minutes:.1f} m)"
    elif tau_minutes > 14:
        return "HOLD", 0.0, f"Trop proche début bet ({tau_minutes:.1f} m)"
        
    # --- 2. PARAMÈTRES DE RISQUE (DURCISSEMENT) ---
    # On passe de 5% à 9% d'edge minimum requis
    MIN_EDGE = 0.065       
    
    # On interdit les paris extrêmes (trop de variance)
    MIN_PRICE = 0.02  # On n'achète pas de tickets de loterie à 10 centimes
    MAX_PRICE = 0.80  # On ne ramasse pas les miettes à 90 centimes
    
    KELLY_FRACTION = 0.5    # On réduit le Kelly (0.5 -> 0.3) pour être plus conservateur
    BASE_ALLOCATION = 20.0  # On augmente la base car on trade moins souvent (Quality over Quantity)
    
    # --- 3. CALCUL DES FRAIS (Inchangé) ---
    MAX_FEE_RATE = 0.0156 
    fee_up = (MAX_FEE_RATE * 4 * prix_up * (1.0 - prix_up)) if prix_up > 0 else 0.0
    fee_down = (MAX_FEE_RATE * 4 * prix_down * (1.0 - prix_down)) if prix_down > 0 else 0.0
    
    eff_prix_up = prix_up + fee_up
    eff_prix_down = prix_down + fee_down
    
    # --- 4. FILTRE DE PRIX ABSOLU (NOUVEAU) ---
    # Si le prix est hors de la zone "Healthy", on ne touche pas
    if not (MIN_PRICE <= prix_up <= MAX_PRICE) and not (MIN_PRICE <= prix_down <= MAX_PRICE):
        return "HOLD", 0.0, "Prix hors zone de confort (Trop extrême)"

    # --- 5. CALCUL DE L'EDGE NET ---
    edge_up = (fair_up - eff_prix_up) if prix_up > 0 else 0
    edge_down = (fair_dn - eff_prix_down) if prix_down > 0 else 0
    
    # --- 6. DÉCISION ---
    # Opportunité UP
    if edge_up > MIN_EDGE and edge_up > edge_down:
        if eff_prix_up < 1.0 and prix_up >= MIN_PRICE and prix_up <= MAX_PRICE:
            
            # FILTRE DE CONVICTION : On veut que notre modèle soit sûr de lui (> 55%)
            if fair_up < 0.55: 
                return "HOLD", 0.0, f"Edge OK mais conviction trop faible ({fair_up:.2f})"

            kelly_pct = (fair_up - eff_prix_up) / (1.0 - eff_prix_up)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            return "BUY_UP", bet_size, f"Sniper UP | Edge: +{edge_up*100:.1f}% | Fair: {fair_up:.2f}"
        
    # Opportunité DOWN
    elif edge_down > MIN_EDGE:
        if eff_prix_down < 1.0 and prix_down >= MIN_PRICE and prix_down <= MAX_PRICE:
            
            # FILTRE DE CONVICTION
            if fair_dn < 0.55:
                return "HOLD", 0.0, f"Edge OK mais conviction trop faible ({fair_dn:.2f})"

            kelly_pct = (fair_dn - eff_prix_down) / (1.0 - eff_prix_down)
            applied_pct = max(0.0, kelly_pct * KELLY_FRACTION)
            bet_size = BASE_ALLOCATION * applied_pct
            
            return "BUY_DOWN", bet_size, f"Sniper DOWN | Edge: +{edge_down*100:.1f}% | Fair: {fair_dn:.2f}"
        
    return "HOLD", 0.0, "Pas d'opportunité nette"