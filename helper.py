import asyncio
import time

async def expiration_watchdog(portfolio):
    """
    Force la fermeture 10s avant la fin en utilisant le dernier prix connu.
    """
    print("👮 Watchdog d'expiration activé.")
    
    while True:
        await asyncio.sleep(1)
        now = time.time()
        
        # Fin du cycle de 15 minutes (ex: 14:00, 14:15...)
        current_window_end = (int(now) // 900 + 1) * 900
        seconds_remaining = current_window_end - now
        
        # ZONE DE DANGER : Moins de 10 secondes
        if seconds_remaining < 10.0:
            
            if portfolio.positions:
                print(f"⏰ URGENT : Fin du cycle dans {seconds_remaining:.1f}s ! Force Close...")
                
                active_directions = list(portfolio.positions.keys())
                
                for direction in active_directions:
                    # 1. On récupère la position pour voir son dernier prix
                    pos_data = portfolio.positions.get(direction)
                    
                    if pos_data:
                        # 2. On utilise le dernier prix vu par le bot (mis à jour à chaque tick)
                        # Si jamais le prix est None (bug), on met 0 par sécurité pour ne pas inventer d'argent
                        exit_price = pos_data.get("last_price", 0.0)
                        
                        portfolio.close_position(direction, exit_price, "FORCE CLOSE (Watchdog)")
                
                portfolio.print_stats() # On affiche les stats après le nettoyage
                print("🧹 Portefeuille nettoyé.")
                
            await asyncio.sleep(10)

def get_optimal_maker_price(direction, current_bid, current_ask):
    """
    Calcule le prix limite optimal pour un ordre Post-Only.
    """
    # 1. Détermination du Tick Size
    if current_ask < 0.04 or current_bid > 0.96:
        tick_size = 0.001 
    else:
        tick_size = 0.01
        
    # 2. Calcul du Prix Plafond (Ceiling)
    if direction == "UP":
        target = current_bid + tick_size
        max_maker_price = current_ask - tick_size # On ne doit pas toucher l'Ask
        final_price = min(target, max_maker_price)
        
        # Sécurité Spread inversé ou nul
        if final_price < current_bid: final_price = current_bid
        return round(final_price, 4)

    elif direction == "DOWN":
        target = current_ask - tick_size
        min_maker_price = current_bid + tick_size
        final_price = max(target, min_maker_price)
        
        if final_price > current_ask: final_price = current_ask
        return round(final_price, 4)
    
    return current_bid # Fallback