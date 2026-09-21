# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **paper-trading bot** that exploits mispricing of Polymarket's *BTC 15-minute Up/Down* binary options. It models the underlying implied volatility (IV) with a gradient-boosted tree (XGBoost/LightGBM) or a neural net, prices the binary with Black-Scholes, and compares that "fair value" against Polymarket's live order book. When the model's fair value beats the market price by a configurable edge, it simulates a maker/taker entry and manages the position with a layered exit engine.

Nothing here touches real funds: `PaperTrader` (in `trade.py`) is a full in-memory simulation. There are no wallets, private keys, or Polymarket order-placement calls. Comments, logs, and print statements are in **French**.

## Commands

All scripts use **hardcoded absolute paths** (`/home/ryan/GitProject/IV_Model/...`) — update them if the repo lives elsewhere. There is no `requirements.txt`; dependencies are installed ad hoc (`xgboost`, `lightgbm`, `optuna`, `torch`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `websockets`, `requests`, `matplotlib`, `seaborn`, `joblib`, `python-binance`, `yfinance`).

```bash
# Run the live bot (from repo root) — needs internet for Deribit/Binance/Polymarket WebSockets
python main.py

# Rebuild the training dataset (merges Deribit option IVs with Binance spot features)
python LGBM_ShortVol/data/data_creator.py      # -> df_4h_feature_v3.csv

# Train + Optuna-tune the IV model, saves best_xgb_model_r_4h_v3_s.json
python LGBM_ShortVol/LGBM_train.py

# Evaluate a trained model on held-out data (MAE/RMSE/R², smile reconstruction plots)
python LGBM_ShortVol/model_eval.py

# Analyze the trade journal -> rapport_trading_btc.png (equity curve, PnL by exit reason, UP/DOWN bias)
python analyse_strat.py
```

There is no test suite, linter, or build step.

## Live architecture (`main.py`)

The bot is a single asyncio process built around one central `asyncio.Queue`. Multiple producer coroutines feed the queue; one consumer loop reacts:

- **Producers**
  - `FeatureEngine.watch_binance_1m` (`feature_engine.py`) — Binance 1m klines → rolling volatility/momentum/RSI features held in memory.
  - `FeatureEngine.watch_live_r` — hourly Deribit REST call to derive the risk-free rate `r` from the front future's basis.
  - `fetch_options_dataset` (`deribit_live.py`) — snapshots the Deribit BTC option book every 10 s. In the current `main.py` this dataset only acts as a **"market is alive" gate**; the IV actually used for pricing comes from the model, not from Deribit's quoted IV.
  - `watch_spot_price` (`polymarket_live.py`) — Polymarket RTDS Chainlink feed → global `current_spot_price`.
  - `watch_clob_market` (`polymarket_live.py`) — Polymarket CLOB WebSocket for the current 15m market's Up/Down order book.

- **Consumer loop** (`orchestrator`) reacts to `("polymarket", ...)` events: builds the live feature vector, predicts σ, prices the binary, then runs entry logic (with a confirmation delay + per-direction cooldown) and a multi-block exit engine.

### The market cycle (Polymarket 15m windows)
`polymarket_live.py` derives the active market from the wall clock: the slug is `btc-updown-15m-<start_ts>` where `start_ts` is the current UTC time floored to a 15-minute boundary. **The strike `K` is the Binance 1m open of that window** (`fetch_strike_K`). `tau` = seconds to the window end. `watch_clob_market` rotates to the new slug automatically each window, and `expiration_watchdog` (`helper.py`) force-closes any open position ~10 s before window end using the last known price.

### Order-book hygiene
`watch_clob_market` applies two filters before trusting a quote: (1) reject "dirty" prices with >2 decimals (`is_clean_price`), (2) reject books with spread > 0.10. It maintains `last_valid_book` and runs a **fast path** (`process_pending_orders`, the maker-fill matching engine) on every valid tick and a throttled **slow path** (model pricing) at ~1 Hz.

### Pricing (`pricing.py`)
`BS_Bin` returns the fair Up/Down probabilities as the discounted risk-neutral `N(d2)` / `1−N(d2)` of a binary option. `BS_Bin_Skew` additionally adjusts for a volatility-skew slope via the vanilla Vega (available but not wired into `main.py`).

### Trade lifecycle (`trade.py` `PaperTrader`)
- **Entry** is maker/post-only: `place_post_only_order` locks funds immediately; `process_pending_orders` fills pessimistically (only when the opposite ask crosses the limit) and auto-refunds orders after a 5 s TTL.
- **Sizing** uses fractional Kelly on the edge (`calculate_trade_signal`).
- **Exits** are a priority ladder in `orchestrator` (take-profit near fair value, FV-invalidation, FV-trailing stop, edge-compression, break-even, hard stop-loss), several of which arm a per-direction cooldown.
- **Fees**: entries are treated as fee-free maker; active exits pay the Polymarket taker fee `shares · p · 0.25 · (p(1−p))²`.
- Every closed trade is appended to `trade_logs/trade_journal.csv` via `TradeRecorder` (`trade_logger.py`).

## The feature contract (most important gotcha)

The live feature vector produced by `FeatureEngine.get_live_vector` **must match, in exact column order, the vector the model was trained on** (`LGBM_train.py` / `data_creator.py`). A mismatch silently produces garbage predictions. There are two incompatible variants in the tree:

| | Root (active, `main.py`) | `bot_poly_taker_v3/` (NN experiment) |
|---|---|---|
| Model | XGBoost `best_xgb_model_r_4h_v3_s.json` | PyTorch `Enhanced_NN_Reg` (`deep_model.py`) |
| Feature engine | `feature_engine.py` | `feature_engine_nn.py` |
| Feature count | **16**, `r` excluded | **17**, `r` included at index 2 |
| Macro timeframes | 15m / 1h | 4h / 24h |
| Preprocessing | none (trees) | requires a fitted `StandardScaler` (`nn_scaler_v2.pkl`) — the NN outputs nonsense without it |

When changing features, update **all three**: `data_creator.py` (dataset build), `LGBM_train.py` (`features` list), and `get_live_vector` (live assembly).

## Repository layout

- **Repo root** — the *active* live bot (XGBoost, 15m/1h features). This is what `main.py` runs on the `xg_v3_taker_nn` branch.
- `bot_poly_taker/` — an earlier v1 snapshot of the bot (self-contained copy).
- `bot_poly_taker_v3/` — in-progress NN variant (24h features, PyTorch, scaler); currently untracked/experimental.
- `LGBM_ShortVol/` — the model pipeline: `data/data_downloader.py` (Binance/yfinance OHLC) → `data/data_creator.py` (merge_asof of Deribit option IVs with spot features) → `LGBM_train.py` (Optuna tuning) → `models/*.json|*.pkl` → `model_eval.py`.
- `trade_logs/` — trade journals (CSV) written live and consumed by `analyse_strat.py`.
- `rapport/`, `*.png` — generated performance dashboards.
- `Papier/` — reference research papers (IV surfaces, SABR).
- `GP_Vol/` — empty placeholder (Gaussian-Process vol idea, unimplemented).

## Conventions & gotchas

- **Git LFS** tracks `*.csv`, `*.pkl`, `*.json` (see `.gitattributes`) — the multi-hundred-MB models and datasets are LFS pointers, not raw files. Ensure `git lfs` is installed before cloning/committing model artifacts.
- Absolute paths are hardcoded in every script (model loads, CSV reads/writes). Relocating the repo requires editing them.
- The live pipeline is entirely simulated; to go live you would need to add real Polymarket order placement and settlement.
- Many tuning constants (edge thresholds, cooldown durations, Kelly fraction, spread caps, confirmation delay) are inline literals in `main.py` and `trade.py` — there is no config file.
