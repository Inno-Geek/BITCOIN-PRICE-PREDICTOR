# BTC/USDT Multi-Horizon Forecaster — Streamlit App

Live inference UI for `btc_multihorizon_model.keras` (CNN-BiLSTM + Multi-Head
Attention, 3 heads: +2h / +4h / +24h log-return forecasts).

## Files
- `streamlit_app.py` — the app (UI, live data fetch, charts)
- `model_utils.py` — feature engineering (ported 1:1 from `btc_cnn_lstm_fixed.ipynb`),
  scaling, and inference logic
- `btc_multihorizon_model.keras` — your trained model (place in this folder)
- `btc_scaler_params.json` — your scaler params (place in this folder)
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this folder (including the `.keras` and `.json` files) to a GitHub repo.
2. On share.streamlit.io, point a new app at `streamlit_app.py`.
3. No secrets needed — the app reads BTC/USDT candles directly from Binance's
   public REST API via `ccxt` (no API key required for public market data).

## How it works
1. Fetches the last 500 hourly BTC/USDT candles from Binance.
2. Recomputes all 40 model features exactly as in training (RSI, MACD,
   Bollinger, ATR, Stochastic, MA ratios, rolling volatility, momentum,
   volume-lag ratios, cyclical time encodings).
3. Scales the last 48h window with the saved `MinMaxScaler` params
   (`X_min` / `X_scale`).
4. Runs the model, inverse-scales the 3 log-return outputs per horizon,
   and reconstructs price as `anchor_close * exp(predicted_log_return)`.
5. Optionally runs 30 stochastic forward passes with dropout active
   (MC-Dropout) to sketch a rough uncertainty band — this is a heuristic,
   not a calibrated confidence interval.
6. Derives a BUY / HOLD / SELL signal as a transparent, rule-based read
   of the model's own 3 forecasts — not a separate trained classifier.
   It's a weighted composite of predicted % change (2h weighted most,
   24h least), thresholded against sidebar-adjustable buy/sell cutoffs,
   and auto-downgraded to HOLD if MC-Dropout uncertainty swamps the
   signal. This is not financial advice — it's a readable summary of
   what the forecasts already say, with the math shown in the UI.

## Notes / things to double check
- **Network access**: the sandbox this was built in blocks `api.binance.com`,
  so live fetching couldn't be tested end-to-end here. The full pipeline
  (feature engineering → scaling → inference → MC-Dropout) was verified
  offline against synthetic OHLCV data and produced numerically sane
  results with correct feature ordering. Confirm the live Binance fetch
  works in your actual deployment environment.
- If Binance is geo-blocked in your deployment region, swap
  `ccxt.binance(...)` in `streamlit_app.py` for `ccxt.binanceus(...)` or
  another supported exchange with a `BTC/USDT` 1h market.
- `FETCH_LIMIT = 500` covers the 168h moving-average warmup plus the 48h
  input window with margin; don't drop it below ~220 or `MA_168_ratio`
  will be all-NaN for recent rows.
