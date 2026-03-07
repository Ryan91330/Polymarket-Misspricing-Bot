import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Configuration visuelle
plt.style.use('bmh')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

CSV_FILE = "trade_logs/trade_journal.csv"

def load_data():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Le fichier {CSV_FILE} n'existe pas encore.")
        return None
    
    try:
        df = pd.read_csv(CSV_FILE)
        df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'])
        df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'])
        
        # Nettoyage des spots à 0 et doublons éventuels
        df = df[df['entry_spot'] > 1000] 
        return df
    except Exception as e:
        print(f"⚠️ Erreur de lecture CSV : {e}")
        return None

def clean_exit_reason(reason):
    if not isinstance(reason, str): return "Inconnu"
    if "FV REVERSAL" in reason: return "📉 FV REVERSAL (Modèle Pessimiste)"
    elif "FV INVALIDATION" in reason: return "💀 FV INVALIDATION (Thèse Invalidée)"
    elif "TAKE PROFIT" in reason: return "💎 TAKE PROFIT"
    elif "LOCK PROFIT" in reason: return "🔐 LOCK PROFIT"
    elif "STOP LOSS" in reason: return "🛑 STOP LOSS"
    elif "EDGE GONE" in reason: return "😐 EDGE GONE"
    elif "BREAK-EVEN" in reason: return "🛡️ BREAK-EVEN"
    elif "FORCE CLOSE" in reason or "Watchdog" in reason: return "⏰ FORCE CLOSE (Expiration)"
    else: return reason.split('(')[0].strip()

def analyze():
    df = load_data()
    if df is None or len(df) < 2:
        print("📭 Pas assez de données pour analyser.")
        return

    df['clean_reason'] = df['exit_reason'].apply(clean_exit_reason)

    print(f"\n📊 ANALYSE DE LA SESSION ({len(df)} Trades)")
    print("="*80)

    # Stats Globales
    total_pnl = df['pnl_usd'].sum()
    win_rate = (len(df[df['pnl_usd'] > 0]) / len(df)) * 100
    print(f"💰 PnL Total      : {total_pnl:+.2f} $")
    print(f"🎯 Win Rate       : {win_rate:.1f} %")
    
    # --- ANALYSE DE CORRELATION BTC ---
    df['btc_move'] = (df['exit_spot'] - df['entry_spot']) / df['entry_spot']
    correlation = df['pnl_usd'].corr(df['btc_move'])
    print(f"🔗 Corrélation PnL/BTC : {correlation:.2f} (0=Neutre, >0.5=Long, <-0.5=Short)")
    print("-" * 80)

    # --- NOUVEAU : ANALYSE PAR DIRECTION (UP vs DOWN) ---
    print("\n⚖️  COMPARATIF UP vs DOWN")
    
    # On groupe par direction
    dir_stats = df.groupby('direction').agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'roi_pct': 'mean'
    })
    
    # On calcule le WinRate pour chaque direction manuellement
    up_trades = df[df['direction'] == 'UP']
    down_trades = df[df['direction'] == 'DOWN']
    
    win_rate_up = (len(up_trades[up_trades['pnl_usd'] > 0]) / len(up_trades) * 100) if len(up_trades) > 0 else 0
    win_rate_down = (len(down_trades[down_trades['pnl_usd'] > 0]) / len(down_trades) * 100) if len(down_trades) > 0 else 0
    
    # Affichage propre
    dir_stats.columns = ['Nb Trades', 'PnL Total ($)', 'PnL Moyen ($)', 'ROI Moyen (%)']
    dir_stats['Win Rate (%)'] = [win_rate_down if idx == 'DOWN' else win_rate_up for idx in dir_stats.index]
    
    print(dir_stats)
    
    # Petit check du biais
    nb_up = len(up_trades)
    nb_down = len(down_trades)
    total = nb_up + nb_down
    print(f"\n🧠 Biais du Modèle : {nb_down/total*100:.1f}% DOWN vs {nb_up/total*100:.1f}% UP")
    if nb_down > nb_up * 1.5:
        print("⚠️ ALERTE : Ton modèle shorte massivement (Biais Bearish détecté).")

    print("-" * 80)

    # --- GENERATION DES GRAPHIQUES ---
    generate_charts(df)

def generate_charts(df):
    """Génère un dashboard visuel complet (3 rangées)"""
    # On agrandit la figure pour ajouter la rangée du bas
    fig = plt.figure(figsize=(16, 14))
    # 3 Lignes, 2 Colonnes
    gs = fig.add_gridspec(3, 2)

    # ==========================================================
    # 1. EQUITY CURVE + BTC OVERLAY (Haut - Largeur totale)
    # ==========================================================
    ax_equity = fig.add_subplot(gs[0, :]) 
    
    df_sorted = df.sort_values('timestamp_exit')
    df_sorted['cumulative_pnl'] = df_sorted['pnl_usd'].cumsum()
    
    btc_points_entry = df[['timestamp_entry', 'entry_spot']].rename(columns={'timestamp_entry': 'ts', 'entry_spot': 'price'})
    btc_points_exit = df[['timestamp_exit', 'exit_spot']].rename(columns={'timestamp_exit': 'ts', 'exit_spot': 'price'})
    btc_curve = pd.concat([btc_points_entry, btc_points_exit]).sort_values('ts').drop_duplicates('ts')

    color_eq = 'tab:green'
    line_eq = ax_equity.plot(df_sorted['timestamp_exit'], df_sorted['cumulative_pnl'], color=color_eq, linewidth=2, label='Equity ($)')
    ax_equity.fill_between(df_sorted['timestamp_exit'], df_sorted['cumulative_pnl'], color=color_eq, alpha=0.1)
    ax_equity.set_ylabel('PnL Cumulé ($)', color=color_eq, fontsize=12)
    ax_equity.tick_params(axis='y', labelcolor=color_eq)
    ax_equity.set_title(f'Equity Curve vs BTC Price (Correlation: {df["pnl_usd"].corr(df["btc_move"]):.2f})', fontsize=14)
    ax_equity.grid(True, linestyle='--', alpha=0.5)

    ax_btc = ax_equity.twinx()
    color_btc = 'black'
    line_btc = ax_btc.plot(btc_curve['ts'], btc_curve['price'], color=color_btc, linewidth=1, linestyle='--', alpha=0.6, label='BTC Price')
    ax_btc.set_ylabel('Bitcoin Price ($)', color=color_btc, fontsize=12)
    
    lines = line_eq + line_btc
    labels = [l.get_label() for l in lines]
    ax_equity.legend(lines, labels, loc='upper left')

    # ==========================================================
    # 2. PNL PAR RAISON (Milieu Gauche)
    # ==========================================================
    ax_reason = fig.add_subplot(gs[1, 0])
    reason_sum = df.groupby('clean_reason')['pnl_usd'].sum().sort_values()
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in reason_sum.values]
    reason_sum.plot(kind='barh', ax=ax_reason, color=colors)
    ax_reason.set_title('PnL Net par Raison de Sortie')
    ax_reason.set_xlabel('PnL ($)')

    # ==========================================================
    # 3. IMPACT VOLATILITÉ (Milieu Droite)
    # ==========================================================
    ax_scatter = fig.add_subplot(gs[1, 1])
    sc = ax_scatter.scatter(df['sigma_pred'], df['pnl_usd'], 
                            c=df['pnl_usd'], cmap='RdYlGn', alpha=0.8, edgecolor='black')
    ax_scatter.set_title('Performance vs Volatilité Prédite (Sigma)')
    ax_scatter.set_xlabel('Sigma (Volatilité)')
    ax_scatter.set_ylabel('PnL Trade ($)')
    ax_scatter.axhline(0, color='black', linestyle='--')

    # ==========================================================
    # 4. PNL PAR DIRECTION (Bas Gauche) - NOUVEAU
    # ==========================================================
    ax_dir_pnl = fig.add_subplot(gs[2, 0])
    dir_sum = df.groupby('direction')['pnl_usd'].sum()
    colors_dir = ['#d62728' if x < 0 else '#2ca02c' for x in dir_sum.values]
    
    # Petit trick pour centrer les barres
    dir_sum.plot(kind='bar', ax=ax_dir_pnl, color=colors_dir, rot=0)
    ax_dir_pnl.set_title('PnL Net : UP vs DOWN')
    ax_dir_pnl.set_ylabel('PnL ($)')
    ax_dir_pnl.axhline(0, color='black', linewidth=1)

    # ==========================================================
    # 5. VOLUME PAR DIRECTION (Bas Droite) - NOUVEAU
    # ==========================================================
    ax_dir_count = fig.add_subplot(gs[2, 1])
    dir_counts = df['direction'].value_counts()
    
    # Pie chart pour bien voir la proportion
    dir_counts.plot(kind='pie', ax=ax_dir_count, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax_dir_count.set_ylabel('') # Enlever le label y moche
    ax_dir_count.set_title('Fréquence des Paris (Biais du Modèle)')

    plt.tight_layout()
    try:
        plt.savefig('rapport_trading_btc.png')
        print("\n📸 Graphique complet sauvegardé sous 'rapport_trading_btc.png'")
    except:
        pass
    plt.show()

if __name__ == "__main__":
    analyze()