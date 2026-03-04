import pandas as pd
import numpy as np
import sys
import os

# 1. On récupère le chemin absolu du dossier
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
if dossier_actuel not in sys.path:
    sys.path.insert(0, dossier_actuel)
from data_downloader import load_ohlc, log_return

MAX_TAU = 1.0 * 4 / 24 / 365.25 # 4h

# ==========================================
# 2. CHARGEMENT ET FILTRAGE DES OPTIONS
# ==========================================
df = pd.read_csv("/home/ryan/GitProject/IPINN/data/Options Trade/BTC/deribit_data_BTC_option_2023-01-01_2026-02-23.csv")
df_spot = pd.read_csv("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/data/BTC_1m_20230101-20260223.csv")

df = df[(df['tau'] <= MAX_TAU)]
df = df[(df['iv'] > 0.0) & (df['iv'] <= 3.0)] # on ne prend pas les iv outliers
df = df[(df['M'] >= 0.90) & (df['M'] <= 1.10)]
print(f"✅ Options gardées après filtrage : {len(df)}")

# --- NOUVELLES FEATURES OPTIONS (Temps & Espace) ---
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

# Distance absolue à la monnaie (Le "U" du Smile)
df['abs_dist_ATM'] = abs(df['M'] - 1.0)
# --- FEATURES NON-LINÉAIRES (Spécial Réseaux de Neurones) ---

# 1. Le Smile pur (La parabole parfaite centrée sur le prix actuel)
df['smile_curve'] = (df['M'] - 1.0)**2 

# 2. Le Moneyness au carré (Alternative classique)
df['M_squared'] = df['M']**2 

# 3. La racine carrée du temps (La norme de diffusion de Black-Scholes)
df['tau_sqrt'] = np.sqrt(df['tau'])
# ==========================================
# 3. CRÉATION DES FEATURES SPOT (Dynamique)
# ==========================================
# Assurons-nous que df_spot a bien la colonne date en datetime
df_spot['date'] = pd.to_datetime(df_spot['date'])
df_spot = df_spot.sort_values('date').reset_index(drop=True)

df_spot['ret_1m'] = log_return(df_spot['close'])
annualization_factor = np.sqrt(365.25 * 24 * 60)

# A. Dynamique de base (RV 30m et Momentum)
df_spot['RV_30m'] = df_spot['ret_1m'].rolling(window=30).std() * annualization_factor
df_spot['Mom_30m'] = df_spot['close'] / df_spot['close'].shift(30) - 1.0

# B. L'Accélération de la Volatilité (RV 5m / RV 30m)
df_spot['RV_5m'] = df_spot['ret_1m'].rolling(window=5).std() * annualization_factor
df_spot['RV_Ratio'] = df_spot['RV_5m'] / df_spot['RV_30m']
# Sécurité vitale pour les NNs et Arbres : corriger les divisions par zéro
df_spot['RV_Ratio'].replace([np.inf, -np.inf], 1.0, inplace=True)
df_spot['RV_Ratio'].fillna(1.0, inplace=True)

# C. Volatilité de Parkinson (Les mèches cachées sur 30m)
df_spot['high_30m'] = df_spot['high'].rolling(window=30).max()
df_spot['low_30m'] = df_spot['low'].rolling(window=30).min()
# Log du ratio High/Low pour détecter l'étirement maximal du carnet d'ordre
df_spot['Parkinson_30m'] = np.log(df_spot['high_30m'] / df_spot['low_30m'])

# On nettoie les NaN créés par le rolling
df_spot = df_spot.dropna().reset_index(drop=True)


# ==========================================
# 4. LA FUSION (Merge_AsOf)
# ==========================================
print("🔗 Fusion avec le DataFrame des options Deribit...")

df = df.sort_values('date')
df_spot = df_spot.sort_values('date')

# On sélectionne toutes nos super-features du Spot
cols_to_merge = ['date', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m']

df_BTC = pd.merge_asof(
    df, 
    df_spot[cols_to_merge], 
    on='date',
    direction='backward'
)

# Exportation sans l'index pour garder un CSV propre
df_BTC.to_csv(f"df_4h_feature.csv", index=False)
print("✅ Fusion réussie ! Le dataset suprême est prêt pour tes modèles.")

# Aperçu des colonnes que tu passeras dans la variable `features` de tes modèles
features_list = ['M', 'tau', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m']
print("\nAperçu de tes Features :")
print(df_BTC[features_list].head())