def get_dynamic_thresholds(entry_price):
    """
    Renvoie les seuils de TP et SL adaptés au prix d'entrée.
    Retourne (take_profit_price, stop_loss_price)
    """
    # ZONE 1 : LOTTO (Prix très bas < 0.10$)
    # Ici, le Stop Loss est suicidaire à cause du spread.
    # On accepte de perdre la mise (SL à 0.005) pour viser un doublement (TP x2).
    if entry_price < 0.10:
        tp_price = entry_price * 2.0   # +100%
        sl_price = 0.005               # Quasi 0 (Hold to death)
        return tp_price, sl_price

    # ZONE 3 : STANDARD (> 0.30$)
    # Gestion classique conservatrice.
    else:
        tp_price = entry_price * 1.20  # +20%
        sl_price = entry_price * 0.85  # -15%
        return tp_price, sl_price