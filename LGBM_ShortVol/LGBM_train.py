import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import optuna
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore") # Cache les warnings inutiles

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================
df_BTC = pd.read_csv("/home/ryan/GitProject/IV_Model/LGBM_ShortVol/df_4h_feature_v3.csv")

features = [
    'M', 'tau', 'abs_dist_ATM', 'RV_30m', 'Mom_30m', 'RV_Ratio', 'Parkinson_30m',
    'Mom_15m', 'Mom_1h', 'Dist_SMA_15m', 'Dist_SMA_1h', 'RSI_1h', 
    'RV_1h', 'Vol_Ratio_30m_1h', 'Hour_of_day', 'Day_of_week'
]
target = 'iv'

# Mélange et nettoyage
df_shuffled = df_BTC.sample(frac=1, random_state=91).reset_index(drop=True)
df_clean = df_shuffled.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

# Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Données prêtes ! Train: {len(X_train)} lignes | Test: {len(X_test)} lignes")

# ==========================================
# 2. FONCTIONS OBJECTIVES OPTUNA
# ==========================================

def objective_xgb(trial):
    """ Espace de recherche pour XGBoost """
    param = {
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'early_stopping_rounds': 200, # XGBoost gère l'early stopping directement ici
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

def objective_lgb(trial):
    """ Espace de recherche pour LightGBM """
    param = {
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'num_leaves': trial.suggest_int('num_leaves', 40, 512),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_jobs': -1,
        'random_state': 42,
        'verbose':-1
    }
    
    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(stopping_rounds=200, verbose=False), log_evaluation(0)]
    )
    
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# ==========================================
# 3. LANCEMENT DES ÉTUDES OPTUNA
# ==========================================
N_TRIALS = 30  # Monte à 50 ou 100 si tu as le temps

print("\n" + "="*50)
print("🧠 DÉBUT OPTIMISATION : XGBoost")
print("="*50)
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)

# print("\n" + "="*50)
# print("🧠 DÉBUT OPTIMISATION : LightGBM")
# print("="*50)
# study_lgb = optuna.create_study(direction='minimize')
# study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)

# ==========================================
# 4. ENTRAÎNEMENT FINAL DES CHAMPIONS & SAUVEGARDE
# ==========================================
print("\n" + "="*50)
print("🏆 ENTRAÎNEMENT ET SAUVEGARDE DES MEILLEURS MODÈLES")
print("="*50)

# --- Sauvegarde XGBoost ---
print(f"🥇 Meilleur MAE XGBoost : {study_xgb.best_value:.6f}")
best_xgb_params = study_xgb.best_params
best_xgb_params['n_estimators'] = 10000
best_xgb_params['early_stopping_rounds'] = 200
best_xgb_params['n_jobs'] = -1
best_xgb_params['random_state'] = 42

final_xgb = xgb.XGBRegressor(**best_xgb_params)
final_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# XGBoost a une fonction native pour sauvegarder en JSON (très léger et rapide)
xgb_model_path = "/home/ryan/GitProject/IV_Model/LGBM_ShortVol/best_xgb_model_r_4h_v3_s.json"
final_xgb.save_model(xgb_model_path)
print(f"💾 Modèle XGBoost sauvegardé sous : {xgb_model_path}")

# --- Sauvegarde LightGBM ---
# print(f"\n🥇 Meilleur MAE LightGBM : {study_lgb.best_value:.6f}")
# best_lgb_params = study_lgb.best_params
# best_lgb_params['n_estimators'] = 10000
# best_lgb_params['n_jobs'] = -1
# best_lgb_params['random_state'] = 42

# final_lgb = lgb.LGBMRegressor(**best_lgb_params)
# final_lgb.fit(
#     X_train, y_train, 
#     eval_set=[(X_test, y_test)],
#     callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)]
# )

# On utilise joblib pour sauvegarder LightGBM
# lgb_model_path = "/home/ryan/GitProject/IV_Model/LGBM_ShortVol/best_lgbm_model_r_4h.pkl"
# joblib.dump(final_lgb, lgb_model_path)
# print(f"💾 Modèle LightGBM sauvegardé sous : {lgb_model_path}")

# print("\n✅ PIPELINE TERMINÉE AVEC SUCCÈS !")