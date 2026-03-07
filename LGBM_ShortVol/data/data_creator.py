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

df['Hour_of_day'] = df['date'].dt.hour
df['Day_of_week'] = df['date'].dt.dayofweek

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
df_spot['RV_Ratio'] = df_spot['RV_Ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)

# C. Volatilité de Parkinson (Les mèches cachées sur 30m)
df_spot['high_30m'] = df_spot['high'].rolling(window=30).max()
df_spot['low_30m'] = df_spot['low'].rolling(window=30).min()
df_spot['Parkinson_30m'] = np.log(df_spot['high_30m'] / df_spot['low_30m'])

# --- NOUVEAU : LES YEUX MACRO (MAX 1 HEURE) ---
# 15m et 60m

# 1. Momentum Macro
df_spot['Mom_15m'] = df_spot['close'] / df_spot['close'].shift(15) - 1.0
df_spot['Mom_1h'] = df_spot['close'] / df_spot['close'].shift(60) - 1.0

# 2. Distance aux Moyennes Mobiles
df_spot['sma_15m'] = df_spot['close'].rolling(window=15).mean()
df_spot['Dist_SMA_15m'] = (df_spot['close'] - df_spot['sma_15m']) / df_spot['sma_15m']

df_spot['sma_1h'] = df_spot['close'].rolling(window=60).mean()
df_spot['Dist_SMA_1h'] = (df_spot['close'] - df_spot['sma_1h']) / df_spot['sma_1h']

# 3. RSI 1h (Calculé minute par minute sur les 60 dernières)
delta = df_spot['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=60).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=60).mean()
rs = gain / loss
df_spot['RSI_1h'] = 100 - (100 / (1 + rs))

# 4. Contexte de Volatilité 1h
df_spot['RV_1h'] = df_spot['ret_1m'].rolling(window=60).std() * annualization_factor
df_spot['Vol_Ratio_30m_1h'] = df_spot['RV_30m'] / df_spot['RV_1h']
df_spot['Vol_Ratio_30m_1h'] = df_spot['Vol_Ratio_30m_1h'].replace([np.inf, -np.inf], 1.0).fillna(1.0)


# Nettoyage
df_spot = df_spot.dropna().reset_index(drop=True)

# ==========================================
# 4. LA FUSION (Merge_AsOf)
# ==========================================
print("🔗 Fusion avec le DataFrame des options Deribit...")

df = df.sort_values('date')
df_spot = df_spot.sort_values('date')

cols_to_merge = [
    'date', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m',
    'Mom_15m', 'Mom_1h', 'Dist_SMA_15m', 'Dist_SMA_1h', 
    'RSI_1h', 'RV_1h', 'Vol_Ratio_30m_1h'
]

df_BTC = pd.merge_asof(
    df, 
    df_spot[cols_to_merge], 
    on='date',
    direction='backward'
)
df_BTC = df_BTC.dropna()

df_BTC.to_csv(f"df_4h_feature_v3.csv", index=False)
print("✅ Fusion réussie ! Le dataset est prêt.")

# 🚨 LA LISTE EXACTE AVEC R 🚨
features_list = [
    'M', 'tau', 'r', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m',
    'Mom_15m', 'Mom_1h', 'Dist_SMA_15m', 'Dist_SMA_1h', 'RSI_1h', 
    'RV_1h', 'Vol_Ratio_30m_1h', 'Hour_of_day', 'Day_of_week'
]
print("\nAperçu de tes Features :")
print(df_BTC[features_list].head())