import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. RECRÉER LE DATASET DE TEST EXACT
# ==========================================
print("⏳ Chargement des données de test...")
df_BTC = pd.read_csv("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/df_4h_feature.csv")
MAX_TAU = 1.0 / 4 / 24 / 365.25 # 4h
df_BTC = df_BTC[(df_BTC['tau'] <= MAX_TAU)]
features = ['M', 'tau','r', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m','smile_curve','M_squared','tau_sqrt']
target = 'iv'

# Même mélange et nettoyage que lors de l'entraînement
df_shuffled = df_BTC.sample(frac=1, random_state=91).reset_index(drop=True)
df_clean = df_shuffled.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

# On récupère X_test et y_test (les données jamais vues par les modèles)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. CHARGEMENT DES MODÈLES OPTIMISÉS
# ==========================================
print("🤖 Chargement des modèles champions...")
live_xgb = xgb.XGBRegressor()
live_xgb.load_model("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/best_xgb_model_r_4h.json")

live_lgb = joblib.load("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/best_lgbm_model_r_4h.pkl")

# ==========================================
# 3. PRÉDICTIONS ET MÉTRIQUES
# ==========================================
print("⚡ Calcul des prédictions...")
y_pred_xgb = live_xgb.predict(X_test)
y_pred_lgb = live_lgb.predict(X_test)

# Métriques
metrics = {
    'MAE': [mean_absolute_error(y_test, y_pred_xgb), mean_absolute_error(y_test, y_pred_lgb)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_xgb)), np.sqrt(mean_squared_error(y_test, y_pred_lgb))],
    'R2 Score': [r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_lgb)]
}

df_metrics = pd.DataFrame(metrics, index=['XGBoost', 'LightGBM'])
print("\n" + "="*50)
print("📊 RÉSULTATS DES PERFORMANCES (SUR DONNÉES DE TEST)")
print("="*50)
print(df_metrics.to_string())
print("="*50)

# ==========================================
# 4. LE TABLEAU DE BORD VISUEL (4 PLOTS)
# ==========================================
print("\n🎨 Génération du tableau de bord d'évaluation...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Plot 1 : Réalité vs Prédiction (Dispersion) ---
axes[0, 0].scatter(y_test, y_pred_xgb, alpha=0.3, color='orange', label='XGBoost', s=10)
axes[0, 0].scatter(y_test, y_pred_lgb, alpha=0.3, color='blue', label='LightGBM', s=10)
min_val = min(y_test.min(), y_pred_xgb.min(), y_pred_lgb.min())
max_val = max(y_test.max(), y_pred_xgb.max(), y_pred_lgb.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait (y=x)')
axes[0, 0].set_title('Précision Globale : Réalité vs Prédiction')
axes[0, 0].set_xlabel('Vraie IV (Deribit)')
axes[0, 0].set_ylabel('IV Prédite')
axes[0, 0].legend()

# --- Plot 2 : Distribution des Erreurs (Résidus) ---
# Un bon modèle doit avoir une erreur centrée sur 0 (courbe en cloche)
erreurs_xgb = y_pred_xgb - y_test
erreurs_lgb = y_pred_lgb - y_test
sns.kdeplot(erreurs_xgb, ax=axes[0, 1], color='orange', fill=True, label=f'XGBoost (Mean: {erreurs_xgb.mean():.4f})')
sns.kdeplot(erreurs_lgb, ax=axes[0, 1], color='blue', fill=True, label=f'LightGBM (Mean: {erreurs_lgb.mean():.4f})')
axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
axes[0, 1].set_title('Distribution des Erreurs de Prédiction (Résidus)')
axes[0, 1].set_xlabel('Erreur (IV Prédite - IV Réelle)')
axes[0, 1].legend()

# --- Plot 3 : Reconstruction du Smile (Moneyness vs IV) ---
M_test = X_test['M'].values
axes[1, 0].scatter(M_test, y_test, alpha=0.2, color='gray', label='Vraie IV', s=15)
axes[1, 0].scatter(M_test, y_pred_xgb, alpha=0.4, color='orange', label='XGBoost', s=10)
axes[1, 0].scatter(M_test, y_pred_lgb, alpha=0.4, color='blue', label='LightGBM', s=10)
axes[1, 0].set_title('Bataille des Smiles (T ≤ 15 min)')
axes[1, 0].set_xlabel('Moneyness (M = S/K)')
axes[1, 0].set_ylabel('Implied Volatility (IV)')
axes[1, 0].legend()

# --- Plot 4 : Feature Importance (Ce que le modèle regarde vraiment) ---
# On affiche l'importance des variables calculée par LightGBM
importance_lgb = live_lgb.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_lgb})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title("Importance des Variables (LightGBM)")

plt.tight_layout()
plt.show()