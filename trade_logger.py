import pandas as pd
import os
from datetime import datetime

class TradeRecorder:
    def __init__(self, filename="trade_logs/trade_journal.csv"):
        self.filename = filename
        # Les colonnes exactes que tu as demandées
        self.columns = [
            "timestamp_entry", "timestamp_exit", "direction", 
            "entry_price", "exit_price", 
            "entry_spot", "exit_spot", 
            "pnl_usd", "roi_pct", "stake", 
            "sigma_pred", "bid_entry", "ask_entry", 
            "x_live_vector", "exit_reason"
        ]
        
        # Si le fichier n'existe pas, on le crée avec les headers
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def record(self, trade_data):
        """
        Prend un dictionnaire, le transforme en DataFrame et l'ajoute au CSV.
        """
        try:
            # Création d'un DataFrame d'une seule ligne
            df_new = pd.DataFrame([trade_data],columns=self.columns)
            
            # Sauvegarde en mode 'append' (a), sans réécrire le header
            df_new.to_csv(self.filename, mode='a', header=False, index=False)
            # print(f"📝 Trade enregistré dans {self.filename}")
            
        except Exception as e:
            print(f"⚠️ Erreur enregistrement CSV: {e}")