import yfinance as yf
import pandas as pd
from binance.client import Client
import numpy as np

# Create a Binance client (you can use empty keys for public endpoints)
binance = Client("", "")


def clean_yf(df):
    """
    Ensures the DataFrame returned by yfinance is in classic OHLCV format
    with a DatetimeIndex and no MultiIndex columns.
    """
    # If MultiIndex columns → flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only standard OHLCV columns
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]]

    # Optional: delete Adj Close if you don’t want it
    df = df.drop(columns=[c for c in ["Adj Close"] if c in df.columns])

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    df.rename(columns={"Open":"open","High":"high","Low":"low","Volume":"volume","Close":"close"},inplace=True)
    df.index.name = "date"
    
    return df

def load_ohlc(market: str, ticker: str, start_date: str, end_date: str, interval: str):
    """
    Load OHLC data for STOCK or CRYPTO.
    Returns a clean DataFrame with datetime index and OHLCV columns.
    
    Parameters:
        market:     "STOCK" or "CRYPTO"
        ticker:     e.g. "AAPL", "BTCUSDT"
        start_date: "2023-01-01"
        end_date:   "2023-12-31"
        interval:   yfinance interval or Binance interval
        
    Returns:
        pd.DataFrame
    """

    # Validate market argument
    if market not in ["STOCK", "CRYPTO"]:
        raise ValueError("Market must be 'STOCK' or 'CRYPTO'")

    # ------------------------------------------------------------
    # STOCK MARKET (via yfinance)
    # ------------------------------------------------------------
    if market == "STOCK":
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False
        )

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        return clean_yf(df)


    # ------------------------------------------------------------
    # CRYPTO MARKET (via Binance)
    # ------------------------------------------------------------
    if market == "CRYPTO":
        # Convert dates to Binance format
        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        klines = binance.get_historical_klines(
            ticker,
            interval,
            start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Build DataFrame
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])

        # Convert timestamp
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')

        # Keep only OHLCV
        df = df[["open_time", "open", "high", "low", "close", "volume"]]

        # Set datetime index
        df.set_index("open_time", inplace=True)

        # Convert numeric columns
        df = df.astype(float)

        df.index.name = "date"
        return df

def log_return(serie):
    return np.log(serie/serie.shift(1))

def log_return_data(market: str, ticker: str, start_date: str, end_date: str, interval: str):
    data = load_ohlc(market, ticker, start_date, end_date, interval)
    data['return'] = log_return(data['close'])
    return data.dropna()
    
    