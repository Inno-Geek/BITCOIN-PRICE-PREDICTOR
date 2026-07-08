"""
Core utilities for the BTC multi-horizon forecaster.

engineer_features() below is a direct, line-for-line port of the
`engineer_features` function in btc_cnn_lstm_fixed.ipynb (cell 7),
so live features are computed exactly the way the model was trained -
same rolling windows, same normalization by Close, same column order
via feature_cols in btc_scaler_params.json.
"""

import json
import numpy as np
import pandas as pd

SEQ_LEN_FALLBACK = 48


# ----------------------------------------------------------------------
# Scaler params
# ----------------------------------------------------------------------
def load_scaler_params(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Feature engineering — exact port from the notebook
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and time features on hourly OHLCV data.

    Mirrors btc_cnn_lstm_fixed.ipynb exactly. `df` must have a UTC
    DatetimeIndex and columns Open, High, Low, Close, Volume.
    """
    out = df.copy()

    # -- Price-derived (already relative) ------------------------------
    out["Returns"] = out["Close"].pct_change()
    out["Log_Returns"] = np.log(out["Close"] / out["Close"].shift(1))
    out["High_Low_Pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["Volume_Change"] = out["Volume"].pct_change()
    out["HL_CO_norm"] = ((out["High"] + out["Low"]) / 2 - out["Close"]) / out["Close"]

    # -- Moving averages: keep raw MA_w for charting, ratio for modelling --
    for w in [4, 8, 24, 48, 168]:
        out[f"MA_{w}"] = out["Close"].rolling(w).mean()          # display only
        out[f"MA_{w}_ratio"] = out["Close"] / out[f"MA_{w}"]      # model feature

    # -- Rolling volatility (already stationary: std of pct returns) ----
    for w in [4, 8, 24, 48]:
        out[f"Vol_{w}h"] = out["Returns"].rolling(w).std()

    # -- Momentum (already a ratio) --------------------------------------
    for lag in [2, 4, 8, 24]:
        out[f"Mom_{lag}h"] = out["Close"] / out["Close"].shift(lag) - 1

    # -- RSI (14-period, bounded 0-100) ----------------------------------
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # -- MACD: keep raw ($ scale) for charting, normalize by Close for model --
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26                                    # display only
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()  # display only
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]            # display only
    out["MACD_norm"] = out["MACD"] / out["Close"]                  # model feature
    out["MACD_signal_norm"] = out["MACD_signal"] / out["Close"]    # model feature
    out["MACD_hist_norm"] = out["MACD_hist"] / out["Close"]        # model feature

    # -- Bollinger Bands (already ratios) --------------------------------
    ma20 = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    out["BB_width"] = (bb_upper - bb_lower) / out["Close"]
    bb_range = (bb_upper - bb_lower).replace(0, 1e-9)
    out["BB_pos"] = (out["Close"] - bb_lower) / bb_range

    # -- ATR: keep raw ($ scale) for charting, normalize by Close for model --
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - out["Close"].shift()).abs(),
        (out["Low"] - out["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()          # display only
    out["ATR_norm"] = out["ATR"] / out["Close"]  # model feature

    # -- Stochastic Oscillator (14/3, bounded 0-100) ---------------------
    low14 = out["Low"].rolling(14).min()
    high14 = out["High"].rolling(14).max()
    out["Stoch_K"] = 100 * (out["Close"] - low14) / (high14 - low14 + 1e-9)
    out["Stoch_D"] = out["Stoch_K"].rolling(3).mean()

    # -- Lag features: volume lags as ratios -----------------------------
    for lag in [1, 2, 4, 8, 24]:
        out[f"Volume_lag_{lag}h_ratio"] = out["Volume"].shift(lag) / out["Volume"].replace(0, 1e-9)

    # -- Cyclical time encoding (already bounded) ------------------------
    idx_utc = out.index
    out["hour_sin"] = np.sin(2 * np.pi * idx_utc.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx_utc.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx_utc.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx_utc.dayofweek / 7)

    # -- Sanitise ---------------------------------------------------------
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    return out


# ----------------------------------------------------------------------
# Scaling (manual re-implementation of sklearn MinMaxScaler.transform)
# ----------------------------------------------------------------------
def scale_X(X: np.ndarray, X_min: np.ndarray, X_scale: np.ndarray) -> np.ndarray:
    """Replicates MinMaxScaler.transform: X_scaled = (X - data_min_) * scale_"""
    return (X - X_min) * X_scale


def inv_scale_y(scaled: float, y_min: float, y_scale: float) -> float:
    """Replicates MinMaxScaler.inverse_transform for a single scalar target."""
    return scaled / y_scale + y_min


# ----------------------------------------------------------------------
# Live forecast
# ----------------------------------------------------------------------
def make_live_forecast(model, df_feat: pd.DataFrame, scaler_params: dict,
                        mc_dropout_samples: int = 0):
    """
    Produce a forecast from the tail of df_feat (already engineered).
    Predicts log-returns anchored at the last known close, reconstructs
    an absolute price via anchor * exp(predicted_return).

    If mc_dropout_samples > 0, also runs that many stochastic forward
    passes (Dropout active) to estimate a rough uncertainty band.
    """
    feature_cols = scaler_params["feature_cols"]
    seq_len = scaler_params.get("seq_len", SEQ_LEN_FALLBACK)
    horizons = scaler_params["horizons"]
    X_min = np.array(scaler_params["X_min"], dtype=np.float32)
    X_scale = np.array(scaler_params["X_scale"], dtype=np.float32)
    y_min = scaler_params["y_min"]
    y_scale = scaler_params["y_scale"]

    live_feat = df_feat[feature_cols].values[-seq_len:].astype(np.float32)
    live_scaled = scale_X(live_feat, X_min, X_scale)
    live_input = live_scaled[np.newaxis, ...]  # (1, seq_len, n_features)

    anchor = float(df_feat["Close"].iloc[-1])
    current_time = df_feat.index[-1]

    preds_sc = model.predict(live_input, verbose=0)  # list of 3 arrays (1,1)
    keys = list(horizons.keys())  # order matches model output heads: 2h, 4h, 1d

    result = {"current_price": anchor, "current_time": current_time, "horizons": {}}

    for i, key in enumerate(keys):
        log_ret = inv_scale_y(float(preds_sc[i][0][0]), y_min[key], y_scale[key])
        price = anchor * np.exp(log_ret)
        result["horizons"][key] = {
            "hours": horizons[key],
            "log_return": log_ret,
            "price": price,
            "change_pct": (price - anchor) / anchor * 100,
        }

    if mc_dropout_samples and mc_dropout_samples > 0:
        import tensorflow as tf
        input_tensor = tf.convert_to_tensor(live_input)
        mc_prices = {key: [] for key in keys}
        for _ in range(mc_dropout_samples):
            out = model(input_tensor, training=True)
            for i, key in enumerate(keys):
                scaled_val = float(np.array(out[i]).reshape(-1)[0])
                log_ret = inv_scale_y(scaled_val, y_min[key], y_scale[key])
                mc_prices[key].append(anchor * np.exp(log_ret))
        for key in keys:
            arr = np.array(mc_prices[key])
            result["horizons"][key]["mc_mean"] = float(arr.mean())
            result["horizons"][key]["mc_std"] = float(arr.std())
            result["horizons"][key]["mc_low"] = float(np.percentile(arr, 10))
            result["horizons"][key]["mc_high"] = float(np.percentile(arr, 90))

    return result


# ----------------------------------------------------------------------
# Buy / Hold / Sell signal
# ----------------------------------------------------------------------
def compute_signal(forecast: dict, buy_threshold: float = 0.15,
                    sell_threshold: float = -0.15,
                    weights: dict = None) -> dict:
    """
    Derive a simple directional signal from the 3 horizon forecasts.

    This is a transparent, rule-based read of the model's own outputs -
    NOT a separate trained classifier, and not financial advice. It:
      1. Takes a weighted average of the 3 horizons' predicted % change
         (shorter horizons weighted more heavily - they're generally
         more reliable and more actionable for near-term decisions).
      2. Compares that composite score against +/- buy_threshold /
         sell_threshold (in percent).
      3. If MC-Dropout uncertainty is available and the composite score
         is smaller than the average predicted std (i.e. the model's own
         noise swamps the signal), the call is downgraded to HOLD
         regardless of the raw score - the model isn't confident enough
         to justify BUY/SELL.
      4. Checks horizon agreement (do 2h/4h/1d all point the same way?)
         as an extra transparency signal, shown but not used to gate.
    """
    if weights is None:
        weights = {"2h": 0.5, "4h": 0.3, "1d": 0.2}

    horizons = forecast["horizons"]
    changes = {k: horizons[k]["change_pct"] for k in weights}
    composite = sum(changes[k] * weights[k] for k in weights)

    directions = [1 if changes[k] > 0 else (-1 if changes[k] < 0 else 0) for k in weights]
    agree = all(d == directions[0] and d != 0 for d in directions)

    has_uncertainty = "mc_std" in horizons["2h"]
    avg_std_pct = None
    confidence = "N/A"
    downgraded = False

    if has_uncertainty:
        std_pcts = []
        for k in weights:
            anchor = forecast["current_price"]
            std_pcts.append(horizons[k]["mc_std"] / anchor * 100)
        avg_std_pct = sum(std_pcts[i] * list(weights.values())[i] for i in range(len(std_pcts)))
        if abs(composite) < avg_std_pct:
            confidence = "LOW"
        elif abs(composite) < avg_std_pct * 2:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

    signal = "HOLD"
    if composite >= buy_threshold:
        signal = "BUY"
    elif composite <= sell_threshold:
        signal = "SELL"

    if has_uncertainty and confidence == "LOW" and signal != "HOLD":
        downgraded = True
        signal = "HOLD"

    return {
        "signal": signal,
        "composite_pct": composite,
        "changes": changes,
        "agree_all_horizons": agree,
        "confidence": confidence,
        "avg_uncertainty_pct": avg_std_pct,
        "downgraded_for_uncertainty": downgraded,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
    }

