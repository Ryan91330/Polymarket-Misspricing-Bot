import numpy as np
from scipy.stats import norm

def BS_Bin(S_t, r, K, sigma, tau):
    """
    Calcule le prix théorique d'une option binaire (Polymarket Up/Down)
    
    S_t   : Prix du Spot (ex: 65000)
    r     : Taux sans risque annualisé (ex: 0.0 pour la crypto à 15 min)
    K     : Strike du pari (ex: 65100)
    sigma : Volatilité Implicite prédite par ton GP ou XGBoost (ex: 0.45 pour 45%)
    tau   : Temps restant annualisé
    """
    # 1. Sécurité anti-crash : Si le pari est terminé ou que la vol est nulle
    if tau <= 0 or sigma <= 0:
        price_up = 1.0 if S_t > K else 0.0
        price_down = 1.0 - price_up
        return price_up, price_down
        
    # 2. Le cœur du moteur Black-Scholes (Calcul de d2)
    # Dans une option classique, on utilise d1 pour le prix et d2 pour la probabilité.
    # Pour une binaire, SEUL d2 compte !
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # 3. Probabilité Risque-Neutre N(d2)
    prob_up = norm.cdf(d2)
    
    # 4. Application du taux d'actualisation (Discount Factor)
    # Pour des paris à 15 minutes, exp(-r * tau) est quasiment égal à 1.
    price_up = np.exp(-r * tau) * prob_up
    price_down = np.exp(-r * tau) * (1.0 - prob_up)
    
    return round(price_up, 4), round(price_down, 4)


def BS_Bin_Skew(S, r, K, sigma, tau, skew_slope):
    """
    Calcule le prix d'une option binaire en prenant en compte le Volatility Skew.
    tau = temps restant avant expiration (en années).
    """
    # Sécurité pour éviter la division par zéro à l'expiration
    if tau <= 0:
        return (1.0, 0.0) if S >= K else (0.0, 1.0)
        
    # 1. Calcul classique Black-Scholes (No Skew)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # Prix binaire théorique (probabilité risque-neutre actualisée)
    discount = np.exp(-r * tau)
    fair_up_noskew = discount * norm.cdf(d2)
    
    # 2. Calcul du Vega de l'option Vanilla (classique)
    # Formule : S * sqrt(tau) * N'(d1)
    vega_vanilla = S * np.sqrt(tau) * norm.pdf(d1)
    
    # 3. Ajustement du Skew
    # C_bin = C_noskew - Vega_vanilla * d(sigma)/dK
    fair_up = fair_up_noskew - (vega_vanilla * skew_slope)
    
    # 4. Déduction du Put Binaire
    fair_dn = discount - fair_up
    
    # 5. Sécurité mathématique (Polymarket = probabilité entre 0 et 1)
    fair_up = max(0.001, min(0.999, fair_up))
    fair_dn = max(0.001, min(0.999, fair_dn))
    
    return fair_up, fair_dn