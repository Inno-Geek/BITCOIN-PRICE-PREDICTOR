"""
BTC / USDT — MULTI-HORIZON FORECASTER
CNN-BiLSTM + Multi-Head Attention, live inference on Binance 1h candles.

Run:
    streamlit run streamlit_app.py
"""

import time
from datetime import timedelta

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from model_utils import engineer_features, load_scaler_params, make_live_forecast, compute_signal

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
MODEL_PATH = "btc_multihorizon_model.keras"
SCALER_PATH = "btc_scaler_params.json"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
FETCH_LIMIT = 500          # candles pulled from Binance (covers MA_168 warmup + SEQ_LEN)
DATA_TTL_SECONDS = 60      # re-fetch cadence

INK = "#0A0A0A"
PAPER = "#F3F1EA"
ACID = "#FBE626"
LINE = "#0A0A0A"
UP = "#1F7A3F"
DOWN = "#C22B1E"
MUTE = "#6B6A64"

st.set_page_config(
    page_title="BTC/USDT — MULTI-HORIZON FORECASTER",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# STYLE — OFF-WHITE / brutalist
# ------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>
:root {{
    --ink: {INK}; --paper: {PAPER}; --acid: {ACID}; --up: {UP}; --down: {DOWN}; --mute: {MUTE};
}}
html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
.stApp {{
    background-color: var(--paper);
    color: var(--ink);
}}
[data-testid="stSidebar"] {{
    background-color: var(--ink);
    color: var(--paper);
    border-right: 4px solid var(--ink);
}}
[data-testid="stSidebar"] * {{ color: var(--paper) !important; }}
[data-testid="stSidebar"] hr {{ border-color: #3a3a36; }}

.industrial-tag {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--mute);
    border: 1.5px solid var(--ink);
    display: inline-block;
    padding: 2px 8px;
    background: var(--paper);
}}
.brute-header {{
    border: 4px solid var(--ink);
    background: var(--ink);
    color: var(--paper);
    padding: 22px 26px;
    margin-bottom: 18px;
    position: relative;
}}
.brute-header h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.0;
    text-transform: uppercase;
}}
.brute-header .quote {{
    font-family: 'Space Mono', monospace;
    color: var(--acid);
}}
.brute-header p {{
    font-family: 'Space Mono', monospace;
    color: #b8b6ad;
    margin-top: 8px;
    font-size: 13px;
    letter-spacing: 0.04em;
}}
.price-strip {{
    border: 3px solid var(--ink);
    background: var(--acid);
    padding: 14px 22px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 22px;
    font-family: 'Space Mono', monospace;
}}
.price-strip .big {{
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
}}
.forecast-card {{
    border: 3px solid var(--ink);
    background: white;
    padding: 16px 18px 18px 18px;
    margin-bottom: 14px;
    position: relative;
}}
.forecast-card .h-label {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--mute);
}}
.forecast-card .h-price {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    margin: 4px 0 2px 0;
}}
.forecast-card .h-change {{
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 700;
}}
.forecast-card .h-time {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--mute);
    margin-top: 6px;
}}
.up {{ color: var(--up); }}
.down {{ color: var(--down); }}
.section-label {{
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 3px solid var(--ink);
    padding-bottom: 6px;
    margin: 26px 0 14px 0;
}}
.signal-stamp {{
    border: 4px solid var(--ink);
    padding: 20px 24px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}}
.signal-stamp .signal-word {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    transform: rotate(-2deg);
    display: inline-block;
    padding: 2px 14px;
    border: 4px solid currentColor;
}}
.signal-buy {{ background: rgba(31,122,63,0.10); }}
.signal-buy .signal-word {{ color: var(--up); }}
.signal-sell {{ background: rgba(194,43,30,0.10); }}
.signal-sell .signal-word {{ color: var(--down); }}
.signal-hold {{ background: rgba(107,106,100,0.10); }}
.signal-hold .signal-word {{ color: var(--mute); }}
.signal-meta {{
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: var(--ink);
    text-align: right;
    min-width: 260px;
}}
.signal-meta .k {{ color: var(--mute); }}
.footer-note {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--mute);
    border-top: 2px solid var(--ink);
    padding-top: 10px;
    margin-top: 30px;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights...")
def get_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource(show_spinner=False)
def get_scaler_params():
    return load_scaler_params(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def get_exchange():
    return ccxt.binance({"enableRateLimit": True})


@st.cache_data(ttl=DATA_TTL_SECONDS, show_spinner="Pulling live candles from Binance...")
def fetch_ohlcv(_exchange, symbol, timeframe, limit, _cache_bucket):
    raw = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


# ------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="industrial-tag">"CONTROL PANEL"</span>', unsafe_allow_html=True)
    st.markdown("### SETTINGS")
    chart_hours = st.slider("Chart lookback (hours)", 24, 336, 96, step=12)
    show_uncertainty = st.checkbox("Estimate uncertainty (MC-Dropout)", value=True,
                                    help="Runs 30 stochastic forward passes with dropout "
                                         "active to sketch a rough confidence band. Slower.")
    show_features = st.checkbox("Show feature heatmap", value=True)
    st.markdown("---")
    st.markdown("### SIGNAL")
    buy_threshold = st.slider("Buy threshold (composite %)", 0.05, 1.0, 0.15, step=0.05)
    sell_threshold = -st.slider("Sell threshold (composite %, abs)", 0.05, 1.0, 0.15, step=0.05)
    st.caption(
        "Composite = 0.5×(+2h) + 0.3×(+4h) + 0.2×(+24h) predicted % change. "
        "Shorter horizons weighted more heavily."
    )
    st.markdown("---")
    if st.button("↻ FORCE REFRESH", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:11px;line-height:1.6;">'
        'MODEL — CNN + BiLSTM + Multi-Head Attention<br>'
        'INPUT — 48h window · 40 features<br>'
        'TARGET — log-return, anchored at last close<br>'
        'SOURCE — Binance, live, ' + TIMEFRAME + ' candles<br>'
        'REFRESH — every ' + str(DATA_TTL_SECONDS) + 's'
        '</div>', unsafe_allow_html=True
    )

# ------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------
st.markdown(f"""
<div class="brute-header">
    <span class="industrial-tag" style="background:{ACID}; border-color:{ACID}; color:{INK};">"NOT FINANCIAL ADVICE"</span>
    <h1>BTC / USDT<br><span class="quote">"FORECAST"</span></h1>
    <p>CNN-BiLSTM &middot; MULTI-HEAD ATTENTION &middot; +2H / +4H / +24H &middot; TRAINED ON LOG-RETURNS</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# LOAD MODEL + DATA
# ------------------------------------------------------------------------
try:
    model = get_model()
    scaler_params = get_scaler_params()
except Exception as e:
    st.error(f"Could not load model or scaler params: {e}")
    st.stop()

cache_bucket = int(time.time() // DATA_TTL_SECONDS)  # rotates the cache key every TTL window

try:
    exchange = get_exchange()
    raw_df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, FETCH_LIMIT, cache_bucket)
except Exception as e:
    st.error(
        f"Could not fetch live data from Binance ({e}). "
        "If this environment blocks binance.com, try a proxy exchange or check your network settings."
    )
    st.stop()

min_required = scaler_params.get("seq_len", 48) + 168 + 30  # seq_len + longest rolling window + buffer
if len(raw_df) < min_required:
    st.warning(
        f"Only {len(raw_df)} candles fetched; need ~{min_required} for the 168h moving-average "
        "warmup plus the 48h input window. Predictions may be degraded or unavailable."
    )

df_feat = engineer_features(raw_df)

if len(df_feat) < scaler_params.get("seq_len", 48):
    st.error("Not enough clean rows after feature engineering to build a 48h input window. "
             "Try increasing FETCH_LIMIT.")
    st.stop()

forecast = make_live_forecast(
    model, df_feat, scaler_params,
    mc_dropout_samples=30 if show_uncertainty else 0,
)
signal = compute_signal(forecast, buy_threshold=buy_threshold, sell_threshold=sell_threshold)

# ------------------------------------------------------------------------
# PRICE STRIP
# ------------------------------------------------------------------------
current_price = forecast["current_price"]
current_time = forecast["current_time"]
st.markdown(f"""
<div class="price-strip">
    <div>
        <div class="industrial-tag" style="background:{INK};color:{PAPER};border-color:{INK};">LIVE · BINANCE</div>
        <div class="big">${current_price:,.2f}</div>
    </div>
    <div style="text-align:right;">
        <div class="industrial-tag">AS OF</div>
        <div style="font-size:14px; margin-top:4px;">{current_time.strftime('%Y-%m-%d %H:%M UTC')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# SIGNAL STAMP
# ------------------------------------------------------------------------
sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal["signal"]]
agree_txt = "ALL HORIZONS AGREE" if signal["agree_all_horizons"] else "HORIZONS MIXED"
conf_txt = f'CONFIDENCE: {signal["confidence"]}' if signal["confidence"] != "N/A" else "CONFIDENCE: N/A (enable MC-Dropout)"
downgrade_txt = ""
if signal["downgraded_for_uncertainty"]:
    downgrade_txt = '<div class="k">↓ downgraded to HOLD — signal within model\'s own noise band</div>'

st.markdown(f"""
<div class="signal-stamp {sig_class}">
    <div class="signal-word">{signal['signal']}</div>
    <div class="signal-meta">
        <div><span class="k">composite score:</span> {signal['composite_pct']:+.3f}%
            <span class="k">(buy &ge; {signal['buy_threshold']:+.2f}% &middot; sell &le; {signal['sell_threshold']:+.2f}%)</span></div>
        <div><span class="k">2h:</span> {signal['changes']['2h']:+.3f}%
            &nbsp;<span class="k">4h:</span> {signal['changes']['4h']:+.3f}%
            &nbsp;<span class="k">24h:</span> {signal['changes']['1d']:+.3f}%
            &nbsp;<span class="k">&middot;</span> {agree_txt}</div>
        <div>{conf_txt}</div>
        {downgrade_txt}
    </div>
</div>
""", unsafe_allow_html=True)
st.caption(
    "⚠️ Rule-based read of this model's own forecasts, not a trading system and not "
    "financial advice. Do your own research before acting on it."
)

# ------------------------------------------------------------------------
# FORECAST CARDS
# ------------------------------------------------------------------------
st.markdown('<div class="section-label">FORECAST — NEXT 3 HORIZONS</div>', unsafe_allow_html=True)
cols = st.columns(3)
horizon_order = [("2h", "+2 HOURS"), ("4h", "+4 HOURS"), ("1d", "+24 HOURS")]

for col, (key, label) in zip(cols, horizon_order):
    h = forecast["horizons"][key]
    direction = "up" if h["change_pct"] >= 0 else "down"
    arrow = "▲" if h["change_pct"] >= 0 else "▼"
    target_time = current_time + timedelta(hours=h["hours"])
    band = ""
    if "mc_low" in h:
        band = (f'<div class="h-time">80% band: ${h["mc_low"]:,.0f} – ${h["mc_high"]:,.0f} '
                f'(σ ≈ ${h["mc_std"]:,.0f})</div>')
    with col:
        st.markdown(f"""
        <div class="forecast-card">
            <div class="h-label">{label}</div>
            <div class="h-price">${h['price']:,.2f}</div>
            <div class="h-change {direction}">{arrow} {h['change_pct']:+.2f}%</div>
            <div class="h-time">target: {target_time.strftime('%Y-%m-%d %H:%M UTC')}</div>
            {band}
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------
# CANDLESTICK + FORECAST TRAJECTORY
# ------------------------------------------------------------------------
st.markdown('<div class="section-label">PRICE ACTION &amp; PROJECTED PATH</div>', unsafe_allow_html=True)

hist = raw_df.tail(chart_hours)
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
    increasing_line_color=UP, decreasing_line_color=DOWN,
    increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
    name="BTC/USDT",
))

traj_x = [current_time] + [current_time + timedelta(hours=forecast["horizons"][k]["hours"]) for k, _ in horizon_order]
traj_y = [current_price] + [forecast["horizons"][k]["price"] for k, _ in horizon_order]
fig.add_trace(go.Scatter(
    x=traj_x, y=traj_y, mode="lines+markers+text",
    line=dict(color=INK, width=3, dash="dot"),
    marker=dict(size=10, color=ACID, line=dict(color=INK, width=2)),
    text=["now", "+2h", "+4h", "+24h"],
    textposition="top center",
    textfont=dict(family="Space Mono", size=11, color=INK),
    name="Forecast path",
))

if "mc_low" in forecast["horizons"]["2h"]:
    band_x = traj_x[1:] + traj_x[1:][::-1]
    band_hi = [forecast["horizons"][k]["mc_high"] for k, _ in horizon_order]
    band_lo = [forecast["horizons"][k]["mc_low"] for k, _ in horizon_order]
    band_y = band_hi + band_lo[::-1]
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, fill="toself",
        fillcolor="rgba(251,230,38,0.25)", line=dict(color="rgba(0,0,0,0)"),
        name="80% band", showlegend=True,
    ))

fig.update_layout(
    plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    font=dict(family="Space Mono", color=INK, size=12),
    xaxis=dict(showgrid=False, linecolor=INK, linewidth=2),
    yaxis=dict(showgrid=True, gridcolor="#DEDBCF", linecolor=INK, linewidth=2, title="USDT"),
    height=520,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------
# FEATURE / INDICATOR PANEL
# ------------------------------------------------------------------------
if show_features:
    st.markdown('<div class="section-label">MODEL INPUT — LAST 48H FEATURE WINDOW</div>', unsafe_allow_html=True)
    window = df_feat[scaler_params["feature_cols"]].tail(scaler_params.get("seq_len", 48))
    norm_window = (window - window.min()) / (window.max() - window.min() + 1e-9)

    heat = go.Figure(data=go.Heatmap(
        z=norm_window.T.values,
        x=[t.strftime("%m-%d %H:%M") for t in norm_window.index],
        y=norm_window.columns,
        colorscale=[[0, PAPER], [0.5, ACID], [1, INK]],
        showscale=False,
    ))
    heat.update_layout(
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        font=dict(family="Space Mono", color=INK, size=9),
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickangle=45, nticks=12),
    )
    st.plotly_chart(heat, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    latest = df_feat.iloc[-1]
    c1.metric("RSI (14)", f"{latest['RSI']:.1f}")
    c2.metric("Stoch %K", f"{latest['Stoch_K']:.1f}")
    c3.metric("BB position", f"{latest['BB_pos']:.2f}")
    c4.metric("ATR (norm)", f"{latest['ATR_norm']*100:.2f}%")

# ------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------
st.markdown(f"""
<div class="footer-note">
METHODOLOGY — model predicts log-returns anchored at the last completed hourly close,
reconstructed as price = anchor &times; exp(predicted log-return). Scalers are per-horizon
MinMax fits from training data only, loaded from {SCALER_PATH}. Uncertainty band (if enabled)
comes from {30 if show_uncertainty else 0} stochastic forward passes with dropout active — a rough
sketch of model disagreement, not a calibrated confidence interval. This tool is for research
and educational purposes; it is not financial advice.
</div>
""", unsafe_allow_html=True)
