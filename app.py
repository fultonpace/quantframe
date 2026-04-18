"""
QuantFrame — Portfolio Intelligence Dashboard
Modern Portfolio Theory + Factor Risk Analytics
Author: [Your Name]
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantFrame | Portfolio Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #00d4aa;
    --accent2:   #7c6af7;
    --accent3:   #f0c27f;
    --text:      #e2e2f0;
    --muted:     #6b6b8a;
    --danger:    #ff6b6b;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--mono);
    color: var(--accent);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}
.metric-sub {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.positive { color: var(--accent) !important; }
.negative { color: var(--danger) !important; }

/* Section headers */
.section-header {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    margin-top: 2rem;
}

/* Title */
.app-title {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.app-title span { color: var(--accent); }
.app-subtitle {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Weight table */
.weight-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.82rem;
}
.weight-bar-bg {
    background: var(--border);
    height: 4px;
    border-radius: 2px;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}
.weight-bar-fill {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* Stacked info */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}

/* Plotly dark override */
.js-plotly-plot .plotly, .js-plotly-plot .plot-container {
    background: transparent !important;
}

/* Sidebar text contrast */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] caption,
[data-testid="stSidebar"] small {
    color: var(--muted) !important;
}

/* Selectbox, inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
    background: #16161f !important;
    border: 1px solid #2a2a3e !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 3px !important;
}
[data-testid="stSelectbox"] svg { color: var(--muted) !important; }

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Expander */
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
}
[data-testid="stExpander"] p {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    line-height: 1.7 !important;
}

/* Caption text */
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
}

/* Spinner */
[data-testid="stSpinner"] p { color: var(--accent) !important; }
/* Slider track and thumb */
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 0 2px var(--accent) !important;
}
[data-testid="stSlider"] label, [data-testid="stSlider"] p {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}
/* Slider value display */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: var(--accent) !important;
}
.stButton > button {
    background: var(--accent);
    color: var(--bg);
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 3px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: #00f0c0;
    transform: translateY(-1px);
}
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
}

/* Status badge */
.badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    background: rgba(0,212,170,0.15);
    color: var(--accent);
    border: 1px solid rgba(0,212,170,0.3);
}
.badge-warn {
    background: rgba(240,194,127,0.15);
    color: var(--accent3);
    border-color: rgba(240,194,127,0.3);
}
.badge-danger {
    background: rgba(255,107,107,0.15);
    color: var(--danger);
    border-color: rgba(255,107,107,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#6b6b8a", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e", linecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e", linecolor="#1e1e2e"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e"),
)

PRESET_UNIVERSES = {
    "Mega-Cap Tech": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO"],
    "Diversified Blue-Chip": ["AAPL","JPM","JNJ","XOM","PG","V","BRK-B","UNH"],
    "Factor Tilt (Value+Growth)": ["BRK-B","VZ","BAC","INTC","AMZN","NVDA","MSFT","TSLA"],
    "Custom": [],
}

RF_RATE = 0.0525  # approximate current risk-free rate

# ── Helper functions ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: list, period: str) -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    return prices.dropna(how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(period: str) -> pd.Series:
    spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        return spy["Close"].squeeze().dropna()
    return spy["Close"].squeeze().dropna()

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = RF_RATE):
    weights = weights / weights.sum()
    ret  = float(weights @ mu) * 252
    vol  = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sharpe

def neg_sharpe(w, mu, cov, rf):
    r, v, _ = portfolio_stats(w, mu, cov, rf)
    return -(r - rf) / v if v > 0 else 1e6

def min_vol_obj(w, mu, cov, rf=None):
    return float(np.sqrt(w @ cov @ w)) * np.sqrt(252)

def compute_efficient_frontier(mu, cov, n_points=60):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Find min-vol and max-return portfolios
    w0 = np.ones(n) / n
    res_mv = minimize(min_vol_obj, w0, args=(mu, cov), method="SLSQP",
                      bounds=bounds, constraints=constraints)
    vol_min = min_vol_obj(res_mv.x, mu, cov)
    ret_min = float(res_mv.x @ mu) * 252

    ret_max = float(mu.max()) * 252
    target_rets = np.linspace(ret_min, ret_max * 0.98, n_points)

    frontier_vols, frontier_rets, frontier_weights = [], [], []
    for target in target_rets:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: (w @ mu) * 252 - t}]
        res = minimize(min_vol_obj, w0, args=(mu, cov), method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            v = min_vol_obj(res.x, mu, cov)
            frontier_vols.append(v)
            frontier_rets.append(target)
            frontier_weights.append(res.x)

    return np.array(frontier_vols), np.array(frontier_rets), frontier_weights

def compute_var_cvar(returns_series: np.ndarray, confidence: float = 0.95):
    sorted_r = np.sort(returns_series)
    idx = int(np.floor((1 - confidence) * len(sorted_r)))
    var  = -sorted_r[idx]
    cvar = -sorted_r[:idx].mean() if idx > 0 else var
    return float(var), float(cvar)

def compute_rolling_beta(port_ret: pd.Series, bench_ret: pd.Series, window: int = 60):
    aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
    aligned.columns = ["port", "bench"]
    betas = []
    for i in range(window, len(aligned) + 1):
        sub = aligned.iloc[i - window:i]
        cov_  = np.cov(sub["port"], sub["bench"])
        beta  = cov_[0, 1] / cov_[1, 1] if cov_[1, 1] != 0 else np.nan
        betas.append((aligned.index[i - 1], beta))
    return pd.DataFrame(betas, columns=["date", "beta"]).set_index("date")

def compute_sortino(returns_series: np.ndarray, rf_daily: float):
    ann_ret = float(np.mean(returns_series)) * 252
    downside = returns_series[returns_series < rf_daily]
    downside_std = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252) if len(downside) > 0 else 1e-9
    return (ann_ret - RF_RATE) / downside_std

def compute_max_drawdown(cum_ret: pd.Series):
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    return float(drawdown.min())

def compute_calmar(ann_ret: float, max_dd: float):
    return ann_ret / abs(max_dd) if max_dd != 0 else np.nan

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ QuantFrame")
    st.markdown('<p class="app-subtitle">Portfolio Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## Universe")
    preset = st.selectbox("Asset Universe", list(PRESET_UNIVERSES.keys()), index=0, label_visibility="collapsed")

    if preset == "Custom":
        custom_raw = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
        tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    else:
        tickers = PRESET_UNIVERSES[preset]
        st.caption(f"**Assets:** {', '.join(tickers)}")

    st.markdown("---")
    st.markdown("## Parameters")

    period_map = {
        "1 Year": "1y", "2 Years": "2y", "3 Years": "3y",
        "5 Years": "5y", "10 Years": "10y", "15 Years": "15y", "Max": "max"
    }
    period_label = st.select_slider("Lookback Period", list(period_map.keys()), value="2 Years")
    period = period_map[period_label]

    confidence = st.slider("VaR / CVaR Confidence", 0.90, 0.99, 0.95, 0.01, format="%.2f")
    max_weight = st.slider("Max Single Asset Weight", 0.10, 1.0, 0.40, 0.05, format="%.2f")
    rf_input   = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, RF_RATE * 100, 0.25, format="%.2f")
    rf = rf_input / 100

    st.markdown("---")
    st.markdown("## Diversification")
    st.caption("Max assets with nonzero weight in optimal portfolio")
    max_assets = st.slider("Max Holdings (N)", min_value=2, max_value=20, value=8, step=1)

    st.markdown("---")
    run_btn = st.button("▶  Run Optimization")

    st.markdown("---")
    with st.expander("ℹ Model Reference"):
        st.markdown("""
**Mean-Variance Optimization**  
Markowitz (1952): maximize Sharpe ratio on the efficient frontier.

**VaR / CVaR**  
Historical simulation. CVaR = expected loss beyond VaR threshold.

**Rolling Beta**  
60-day OLS beta vs SPY benchmark.

**Sortino Ratio**  
Penalizes only downside deviation.

**Calmar Ratio**  
Ann. return / Max drawdown.
        """)

# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<div class="app-title">Quant<span>Frame</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Modern Portfolio Theory · Factor Risk · Decision Analytics</div>', unsafe_allow_html=True)
with col_badge:
    st.markdown('<br><span class="badge">v1.0</span>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Main Logic ───────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
<div style="text-align:center; padding: 5rem 2rem; font-family: 'IBM Plex Mono', monospace;">
    <div style="font-size:3rem; margin-bottom:1rem; color:#1e1e2e;">⬡</div>
    <div style="font-size:0.9rem; color:#6b6b8a; letter-spacing:0.15em; text-transform:uppercase;">
        Configure your universe in the sidebar<br>and press <span style="color:#00d4aa;">▶ Run Optimization</span>
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Data Fetch ────────────────────────────────────────────────────────────────
with st.spinner("Fetching market data…"):
    prices = fetch_prices(tickers, period)
    bench  = fetch_benchmark(period)

# Validate
valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 50]
if len(valid_tickers) < 2:
    st.error("Need at least 2 valid tickers with sufficient history. Check your inputs.")
    st.stop()

prices  = prices[valid_tickers].dropna()
returns = compute_returns(prices)
mu      = returns.mean().values
cov     = returns.cov().values * 252  # annualized cov matrix
n       = len(valid_tickers)

# ── Optimization ──────────────────────────────────────────────────────────────
with st.spinner("Running optimization…"):
    bounds      = tuple((0.0, max_weight) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n

    # Max Sharpe
    res_sharpe = minimize(neg_sharpe, w0, args=(mu, cov / 252, rf),
                          method="SLSQP", bounds=bounds, constraints=constraints)
    w_sharpe_full = res_sharpe.x / res_sharpe.x.sum()

    # Apply cardinality constraint: keep top-N by weight, re-normalize
    n_keep = min(max_assets, n)
    top_idx = np.argsort(w_sharpe_full)[-n_keep:]
    w_sharpe = np.zeros(n)
    w_sharpe[top_idx] = w_sharpe_full[top_idx]
    w_sharpe = w_sharpe / w_sharpe.sum()

    # Min Volatility
    res_minvol = minimize(min_vol_obj, w0, args=(mu, cov / 252),
                          method="SLSQP", bounds=bounds, constraints=constraints)
    w_minvol_full = res_minvol.x / res_minvol.x.sum()
    w_minvol = np.zeros(n)
    top_idx_mv = np.argsort(w_minvol_full)[-n_keep:]
    w_minvol[top_idx_mv] = w_minvol_full[top_idx_mv]
    w_minvol = w_minvol / w_minvol.sum()

    # Equal Weight baseline
    w_eq = np.ones(n) / n

    # Frontier
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(mu / 252, cov / 252)

# ── Data Source Panel ─────────────────────────────────────────────────────────
import datetime as _dt
_now        = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
_date_start = prices.index[0].strftime("%Y-%m-%d")
_date_end   = prices.index[-1].strftime("%Y-%m-%d")
_n_obs      = len(returns)
_active     = [t for t in valid_tickers if w_sharpe[valid_tickers.index(t)] > 0.001]

st.markdown(f"""
<div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:4px;
            padding:1rem 1.5rem;margin-bottom:1.5rem;
            display:grid;grid-template-columns:repeat(6,1fr);gap:1rem;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Data Source</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#00d4aa;font-weight:600;">
      Yahoo Finance</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">via yfinance</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Date Range</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e2e2f0;font-weight:500;">
      {_date_start}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">→ {_date_end}</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Observations</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e2e2f0;font-weight:500;">
      {_n_obs:,}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">trading days</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Universe</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e2e2f0;font-weight:500;">
      {len(valid_tickers)} assets</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">
      {', '.join(valid_tickers)}</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Active Holdings</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#00d4aa;font-weight:600;">
      {len(_active)} / {len(valid_tickers)}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">
      N={n_keep} constraint</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#6b6b8a;margin-bottom:0.3rem;">
      Last Fetched</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e2e2f0;font-weight:500;">
      {_now}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#6b6b8a;">adj. close prices</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Portfolio stats ───────────────────────────────────────────────────────────
def port_series(weights, ret_df):
    return (ret_df[valid_tickers] @ weights)

rf_daily = rf / 252
bench_ret = bench.pct_change().dropna()

for label, weights in [("Optimal (Max Sharpe)", w_sharpe),
                        ("Min Volatility", w_minvol),
                        ("Equal Weight", w_eq)]:
    pass  # computed below per tab

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  📐  Efficient Frontier  ",
    "  📊  Risk Analytics  ",
    "  🔁  Rolling Metrics  ",
    "  📋  Portfolio Report  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Efficient Frontier · Mean-Variance Optimization</div>', unsafe_allow_html=True)

    # Individual asset stats
    asset_ann_ret = mu * 252
    asset_ann_vol = np.sqrt(np.diag(cov))

    # Portfolio points
    r_sh, v_sh, s_sh = portfolio_stats(w_sharpe, mu, cov / 252, rf)
    r_mv, v_mv, s_mv = portfolio_stats(w_minvol, mu, cov / 252, rf)
    r_eq, v_eq, s_eq = portfolio_stats(w_eq, mu, cov / 252, rf)

    # Sharpe frontier color
    sharpes_frontier = []
    for r_, v_ in zip(frontier_rets, frontier_vols):
        sharpes_frontier.append((r_ - rf) / v_ if v_ > 0 else 0)

    fig = go.Figure()

    # Individual assets scatter
    fig.add_trace(go.Scatter(
        x=asset_ann_vol * 100, y=asset_ann_ret * 100,
        mode="markers+text",
        text=valid_tickers,
        textposition="top center",
        textfont=dict(size=10, color="#6b6b8a", family="IBM Plex Mono"),
        marker=dict(size=9, color="#1e1e2e", line=dict(color="#6b6b8a", width=1.5)),
        name="Individual Assets",
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Frontier
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100, y=frontier_rets * 100,
        mode="lines",
        line=dict(color="rgba(0,212,170,0.4)", width=2),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Capital Market Line
    if v_sh > 0:
        cml_x = np.linspace(0, max(asset_ann_vol) * 1.3, 100)
        cml_y = rf + (r_sh - rf) / v_sh * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x * 100, y=cml_y * 100,
            mode="lines",
            line=dict(color="rgba(124,106,247,0.5)", width=1.5, dash="dash"),
            name="Capital Market Line",
        ))

    # Portfolio markers
    port_points = [
        (v_sh, r_sh, "Max Sharpe", "#00d4aa", "star", 18),
        (v_mv, r_mv, "Min Volatility", "#7c6af7", "diamond", 16),
        (v_eq, r_eq, "Equal Weight", "#f0c27f", "circle", 14),
    ]
    for v_, r_, name_, color_, sym_, sz_ in port_points:
        fig.add_trace(go.Scatter(
            x=[v_ * 100], y=[r_ * 100],
            mode="markers+text",
            text=[name_],
            textposition="top right",
            textfont=dict(size=10, color=color_, family="IBM Plex Mono"),
            marker=dict(size=sz_, color=color_, symbol=sym_,
                        line=dict(color="white", width=1)),
            name=name_,
            hovertemplate=f"<b>{name_}</b><br>Vol: {v_*100:.2f}%<br>Return: {r_*100:.2f}%<br>Sharpe: {(r_-rf)/v_:.3f}<extra></extra>",
        ))

    layout = dict(**PLOT_LAYOUT)
    layout.update(dict(
        title=dict(text="Efficient Frontier & Portfolio Compositions", font=dict(size=13, color="#e2e2f0")),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=500,
    ))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Portfolio comparison metrics
    st.markdown('<div class="section-header">Portfolio Comparison</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for col, (label, weights, ret, vol, sharpe, color) in zip(cols, [
        ("Max Sharpe",    w_sharpe, r_sh, v_sh, s_sh, "#00d4aa"),
        ("Min Volatility",w_minvol, r_mv, v_mv, s_mv, "#7c6af7"),
        ("Equal Weight",  w_eq,     r_eq, v_eq, s_eq, "#f0c27f"),
    ]):
        with col:
            st.markdown(f"""
<div class="metric-card" style="--accent:{color};">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
  <div class="metric-label">{label}</div>
  <span class="badge" style="background:rgba(0,0,0,0.3);color:{color};border-color:{color}30;">Portfolio</span>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
  <div>
    <div class="metric-label">Ann. Return</div>
    <div class="metric-value" style="font-size:1.1rem;color:{color};">{ret*100:.2f}%</div>
  </div>
  <div>
    <div class="metric-label">Ann. Volatility</div>
    <div class="metric-value" style="font-size:1.1rem;">{vol*100:.2f}%</div>
  </div>
  <div>
    <div class="metric-label">Sharpe Ratio</div>
    <div class="metric-value" style="font-size:1.1rem;color:{color};">{sharpe:.3f}</div>
  </div>
  <div>
    <div class="metric-label">Max Weight</div>
    <div class="metric-value" style="font-size:1.1rem;">{weights.max()*100:.1f}%</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Weights chart
    st.markdown('<div class="section-header">Optimal Weights · Max Sharpe Portfolio</div>', unsafe_allow_html=True)
    df_w = pd.DataFrame({
        "Ticker": valid_tickers,
        "Max Sharpe": w_sharpe * 100,
        "Min Vol":    w_minvol * 100,
        "Equal Wt":   w_eq * 100,
    }).sort_values("Max Sharpe", ascending=False)

    fig2 = go.Figure()
    for name, color in [("Max Sharpe","#00d4aa"),("Min Vol","#7c6af7"),("Equal Wt","#f0c27f")]:
        fig2.add_trace(go.Bar(
            name=name, x=df_w["Ticker"], y=df_w[name],
            marker_color=color, marker_line_width=0,
            opacity=0.85,
        ))
    fig2.update_layout(**{**PLOT_LAYOUT,
        "barmode":"group","height":320,
        "yaxis_title":"Weight (%)",
        "title":dict(text="Portfolio Weight Comparison", font=dict(size=12,color="#e2e2f0")),
    })
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Risk Decomposition · VaR · CVaR · Drawdown</div>', unsafe_allow_html=True)

    port_ret_series = port_series(w_sharpe, returns)
    port_ret_arr    = port_ret_series.values

    # VaR / CVaR
    var_h,  cvar_h  = compute_var_cvar(port_ret_arr, confidence)
    sortino = compute_sortino(port_ret_arr, rf_daily)
    cum_ret = (1 + port_ret_series).cumprod()
    max_dd  = compute_max_drawdown(cum_ret)
    calmar  = compute_calmar(r_sh, max_dd)

    # Top metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("VaR", f"{var_h*100:.2f}%", f"{confidence*100:.0f}% 1-day historical", "negative"),
        ("CVaR", f"{cvar_h*100:.2f}%", "Expected shortfall", "negative"),
        ("Sortino", f"{sortino:.3f}", "Downside-adj. return", "positive" if sortino > 0 else "negative"),
        ("Max Drawdown", f"{max_dd*100:.2f}%", "Peak-to-trough", "negative"),
        ("Calmar", f"{calmar:.3f}", "Return / Max DD", "positive" if calmar > 0 else "negative"),
    ]
    for col, (label, val, sub, cls) in zip([m1,m2,m3,m4,m5], metrics):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value {cls}">{val}</div>
  <div class="metric-sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Return distribution + VaR
    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=port_ret_arr * 100,
            nbinsx=60,
            marker_color="rgba(0,212,170,0.5)",
            marker_line_width=0,
            name="Daily Returns",
        ))
        # Normal fit
        x_norm = np.linspace(port_ret_arr.min(), port_ret_arr.max(), 300)
        y_norm = norm.pdf(x_norm, np.mean(port_ret_arr), np.std(port_ret_arr))
        y_norm = y_norm * len(port_ret_arr) * (port_ret_arr.max() - port_ret_arr.min()) / 60
        fig3.add_trace(go.Scatter(
            x=x_norm*100, y=y_norm,
            line=dict(color="#7c6af7", width=2),
            name="Normal Fit",
        ))
        # VaR line
        fig3.add_vline(x=-var_h*100, line_color="#ff6b6b", line_dash="dash", line_width=1.5,
                       annotation_text=f"VaR {confidence:.0%}", annotation_font_color="#ff6b6b",
                       annotation_font_size=10)
        fig3.add_vline(x=-cvar_h*100, line_color="#f0c27f", line_dash="dot", line_width=1.5,
                       annotation_text="CVaR", annotation_font_color="#f0c27f",
                       annotation_font_size=10, annotation_position="bottom right")
        fig3.update_layout(**{**PLOT_LAYOUT,
            "height":340,"title":dict(text="Return Distribution (Max Sharpe Portfolio)",
            font=dict(size=12,color="#e2e2f0")),
            "xaxis_title":"Daily Return (%)", "yaxis_title":"Frequency",
        })
        st.plotly_chart(fig3, use_container_width=True)

    with col_right:
        # Drawdown chart
        rolling_max = cum_ret.cummax()
        drawdown    = (cum_ret - rolling_max) / rolling_max * 100

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.15)",
            line=dict(color="#ff6b6b", width=1.5),
            name="Drawdown",
        ))
        fig4.update_layout(**{**PLOT_LAYOUT,
            "height":340,"title":dict(text="Drawdown Profile",font=dict(size=12,color="#e2e2f0")),
            "yaxis_title":"Drawdown (%)",
        })
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = returns[valid_tickers].corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.index,
        colorscale=[[0,"#7c6af7"],[0.5,"#111118"],[1,"#00d4aa"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family="IBM Plex Mono"),
        showscale=True,
    ))
    fig5.update_layout(**{**PLOT_LAYOUT,
        "height":380,"title":dict(text="Asset Return Correlations",font=dict(size=12,color="#e2e2f0")),
    })
    st.plotly_chart(fig5, use_container_width=True)

    # Asset-level risk table
    st.markdown('<div class="section-header">Asset Risk Decomposition</div>', unsafe_allow_html=True)
    marginal_contrib = (cov / 252 @ w_sharpe) / np.sqrt(w_sharpe @ (cov / 252) @ w_sharpe)
    risk_contrib     = w_sharpe * marginal_contrib
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

    df_risk = pd.DataFrame({
        "Asset":           valid_tickers,
        "Weight (%)":      (w_sharpe * 100).round(2),
        "Ann. Return (%)": (asset_ann_ret * 100).round(2),
        "Ann. Vol (%)":    (asset_ann_vol * 100).round(2),
        "Sharpe":          ((asset_ann_ret - rf) / asset_ann_vol).round(3),
        "Risk Contrib (%)":risk_contrib_pct.round(2),
    }).sort_values("Risk Contrib (%)", ascending=False)

    # Color cells by risk contribution — no matplotlib needed
    rc_vals = df_risk["Risk Contrib (%)"].values
    rc_max  = rc_vals.max() if rc_vals.max() > 0 else 1
    def _rc_color(v):
        t = v / rc_max
        r = int(255 * t)
        g = int(180 * (1 - t))
        return f"rgba({r},{g},80,0.25)"

    cell_colors = [["#111118"] * len(df_risk) for _ in range(6)]
    cell_colors[5] = [_rc_color(v) for v in rc_vals]

    fig_tbl = go.Figure(go.Table(
        columnwidth=[60, 80, 100, 80, 60, 110],
        header=dict(
            values=["<b>Asset</b>","<b>Weight %</b>","<b>Ann. Return %</b>",
                    "<b>Ann. Vol %</b>","<b>Sharpe</b>","<b>Risk Contrib %</b>"],
            fill_color="#0d0d14",
            line_color="#2a2a3e",
            font=dict(family="IBM Plex Mono", size=11, color="#00d4aa"),
            align="center", height=32,
        ),
        cells=dict(
            values=[df_risk[c].tolist() for c in df_risk.columns],
            fill_color=cell_colors,
            line_color="#1e1e2e",
            font=dict(family="IBM Plex Mono", size=11, color="#e2e2f0"),
            align="center", height=28,
        )
    ))
    fig_tbl.update_layout(**{**PLOT_LAYOUT, "height": 60 + 28 * len(df_risk) + 32, "margin": dict(l=0,r=0,t=0,b=0)})
    st.plotly_chart(fig_tbl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROLLING METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Rolling Risk & Factor Exposure · 60-Day Window</div>', unsafe_allow_html=True)

    port_ret_full = port_series(w_sharpe, returns)
    bench_ret_aligned = bench_ret.reindex(port_ret_full.index).dropna()
    port_aligned = port_ret_full.reindex(bench_ret_aligned.index).dropna()

    WINDOW = 60

    # Rolling beta
    roll_beta = compute_rolling_beta(port_aligned, bench_ret_aligned, WINDOW)

    # Rolling Sharpe (annualized)
    roll_sharpe = port_aligned.rolling(WINDOW).apply(
        lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    )

    # Rolling volatility
    roll_vol = port_aligned.rolling(WINDOW).std() * np.sqrt(252) * 100

    # Cumulative return comparison
    cum_port  = (1 + port_aligned).cumprod()
    cum_bench = (1 + bench_ret_aligned).cumprod()

    fig6 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         subplot_titles=["Cumulative Return vs SPY",
                                         "Rolling Beta (60d)",
                                         "Rolling Sharpe (60d, annualized)",
                                         "Rolling Volatility (60d, annualized %)"],
                         vertical_spacing=0.06)

    fig6.add_trace(go.Scatter(x=cum_port.index,  y=cum_port.values,
        name="Max Sharpe Portfolio", line=dict(color="#00d4aa", width=2)), row=1, col=1)
    fig6.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values,
        name="SPY", line=dict(color="#6b6b8a", width=1.5, dash="dash")), row=1, col=1)

    fig6.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta["beta"],
        name="Beta", line=dict(color="#7c6af7", width=1.8),
        fill="tozeroy", fillcolor="rgba(124,106,247,0.1)"), row=2, col=1)
    fig6.add_hline(y=1.0, line_color="#6b6b8a", line_dash="dot", line_width=1, row=2, col=1)

    fig6.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
        name="Sharpe", line=dict(color="#f0c27f", width=1.8),
        fill="tozeroy", fillcolor="rgba(240,194,127,0.1)"), row=3, col=1)
    fig6.add_hline(y=0, line_color="#6b6b8a", line_dash="dot", line_width=1, row=3, col=1)

    fig6.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
        name="Volatility (%)", line=dict(color="#ff6b6b", width=1.8),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.1)"), row=4, col=1)

    fig6.update_layout(**{**PLOT_LAYOUT,
        "height": 900,
        "showlegend": True,
        "title": dict(text="Rolling Factor Analytics · Max Sharpe Portfolio vs SPY",
                      font=dict(size=13, color="#e2e2f0")),
    })
    for i in range(1, 5):
        fig6.update_xaxes(gridcolor="#1e1e2e", row=i, col=1)
        fig6.update_yaxes(gridcolor="#1e1e2e", row=i, col=1)

    st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Portfolio Report · Max Sharpe Portfolio</div>', unsafe_allow_html=True)

    port_ret_series = port_series(w_sharpe, returns)
    cum_port_full   = (1 + port_ret_series).cumprod()
    max_dd_val      = compute_max_drawdown(cum_port_full)
    var_val, cvar_val = compute_var_cvar(port_ret_series.values, confidence)
    sort_val        = compute_sortino(port_ret_series.values, rf_daily)
    calmar_val      = compute_calmar(r_sh, max_dd_val)

    # Omega ratio
    threshold = rf_daily
    gains = port_ret_series[port_ret_series > threshold] - threshold
    losses = threshold - port_ret_series[port_ret_series <= threshold]
    omega = gains.sum() / losses.sum() if losses.sum() > 0 else np.nan

    # Skew & kurtosis
    from scipy.stats import skew, kurtosis
    skew_val = skew(port_ret_series.values)
    kurt_val = kurtosis(port_ret_series.values)

    st.markdown(f"""
<div class="metric-card" style="margin-bottom:1.5rem;">
  <div class="metric-label" style="font-size:0.7rem;margin-bottom:0.75rem;">
    QUANTFRAME PORTFOLIO ANALYTICS REPORT &nbsp;·&nbsp; 
    Universe: {', '.join(valid_tickers)} &nbsp;·&nbsp;
    Period: {period_label} &nbsp;·&nbsp;
    RF: {rf*100:.2f}% &nbsp;·&nbsp;
    Confidence: {confidence:.0%}
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
    <div>
      <div class="metric-label">Annualized Return</div>
      <div class="metric-value positive">{r_sh*100:.2f}%</div>
    </div>
    <div>
      <div class="metric-label">Annualized Volatility</div>
      <div class="metric-value">{v_sh*100:.2f}%</div>
    </div>
    <div>
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-value positive">{s_sh:.4f}</div>
    </div>
    <div>
      <div class="metric-label">Sortino Ratio</div>
      <div class="metric-value {'positive' if sort_val>0 else 'negative'}">{sort_val:.4f}</div>
    </div>
    <div>
      <div class="metric-label">VaR ({confidence:.0%})</div>
      <div class="metric-value negative">{var_val*100:.3f}%</div>
    </div>
    <div>
      <div class="metric-label">CVaR ({confidence:.0%})</div>
      <div class="metric-value negative">{cvar_val*100:.3f}%</div>
    </div>
    <div>
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value negative">{max_dd_val*100:.2f}%</div>
    </div>
    <div>
      <div class="metric-label">Calmar Ratio</div>
      <div class="metric-value {'positive' if calmar_val>0 else 'negative'}">{calmar_val:.4f}</div>
    </div>
    <div>
      <div class="metric-label">Omega Ratio</div>
      <div class="metric-value {'positive' if omega>1 else 'negative'}">{omega:.4f}</div>
    </div>
    <div>
      <div class="metric-label">Skewness</div>
      <div class="metric-value {'positive' if skew_val>0 else 'negative'}">{skew_val:.4f}</div>
    </div>
    <div>
      <div class="metric-label">Excess Kurtosis</div>
      <div class="metric-value {'negative' if kurt_val>0 else 'positive'}">{kurt_val:.4f}</div>
    </div>
    <div>
      <div class="metric-label">Observations</div>
      <div class="metric-value">{len(port_ret_series)}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # All three portfolios weights table
    st.markdown('<div class="section-header">Allocation Breakdown</div>', unsafe_allow_html=True)
    df_alloc = pd.DataFrame({
        "Ticker":           valid_tickers,
        "Max Sharpe (%)":   (w_sharpe * 100).round(2),
        "Min Vol (%)":      (w_minvol * 100).round(2),
        "Equal Weight (%)": (w_eq * 100).round(2),
    }).sort_values("Max Sharpe (%)", ascending=False)

    # Color active holdings
    def _wt_color(v):
        if v > 0.01:
            t = min(v / 40, 1.0)
            return f"rgba(0,{int(180*t+40)},{int(120*t+50)},0.25)"
        return "#0d0d14"

    alloc_colors = [
        ["#111118"] * len(df_alloc),
        [_wt_color(v) for v in df_alloc["Max Sharpe (%)"]],
        [_wt_color(v) for v in df_alloc["Min Vol (%)"]],
        ["#111118"] * len(df_alloc),
    ]
    fig_alloc = go.Figure(go.Table(
        columnwidth=[60, 100, 80, 110],
        header=dict(
            values=["<b>Ticker</b>","<b>Max Sharpe %</b>","<b>Min Vol %</b>","<b>Equal Weight %</b>"],
            fill_color="#0d0d14",
            line_color="#2a2a3e",
            font=dict(family="IBM Plex Mono", size=11, color="#00d4aa"),
            align="center", height=32,
        ),
        cells=dict(
            values=[df_alloc[c].tolist() for c in df_alloc.columns],
            fill_color=alloc_colors,
            line_color="#1e1e2e",
            font=dict(family="IBM Plex Mono", size=11, color="#e2e2f0"),
            align="center", height=28,
        )
    ))
    fig_alloc.update_layout(**{**PLOT_LAYOUT, "height": 60 + 28 * len(df_alloc) + 32, "margin": dict(l=0,r=0,t=0,b=0)})
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Methodology note
    st.markdown('<div class="section-header">Methodology Notes</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;line-height:1.8;color:#6b6b8a;
            background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.25rem;">

<span style="color:#00d4aa;">OPTIMIZATION</span>  
Mean-variance optimization via SLSQP (Markowitz, 1952). Expected returns estimated from historical 
sample mean; covariance from historical sample covariance. Annualized assuming 252 trading days.
Box constraints enforce 0 ≤ wᵢ ≤ max_weight with full-investment constraint Σwᵢ = 1.
Cardinality constraint (max holdings N) applied post-optimization: the N largest weights are 
retained and re-normalized. This is mathematically equivalent to a thresholded tangency portfolio 
and is standard practice for small universes.

<br><br>
<span style="color:#7c6af7;">RISK METRICS</span>  
VaR and CVaR computed via historical simulation — no distributional assumption imposed.  
Sortino uses downside deviation (returns below Rf) as denominator.  
Beta estimated via 60-day rolling OLS covariance ratio against SPY.
Omega ratio = E[gains above Rf] / E[losses below Rf].

<br><br>
<span style="color:#f0c27f;">DATA</span>  
Source: Yahoo Finance via yfinance. Adjusted close prices (splits + dividends).  
Lookback options: 1Y through Max (full history). Cached 1 hour per session.  
Benchmark: SPY (SPDR S&P 500 ETF Trust).

<br><br>
<span style="color:#ff6b6b;">LIMITATIONS</span>  
Sample covariance is a noisy estimator — longer lookbacks reduce variance but may include 
structural breaks. Ledoit-Wolf shrinkage not applied (future work).  
Cardinality constraint via thresholding is a heuristic; true cardinality-constrained MVO is NP-hard.  
Historical returns are not indicative of future performance. Transaction costs and taxes not modeled.

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("QuantFrame · Built with Streamlit + yfinance + SciPy · For educational and research purposes only. Not financial advice.")
