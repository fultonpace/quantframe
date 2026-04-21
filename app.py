"""
QuantFrame v2 — Portfolio Intelligence Dashboard
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
    page_title="QuantFrame v2 | Portfolio Intelligence",
    page_icon="logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #f7f5f0;
    --surface:   #ffffff;
    --border:    #e0d9ce;
    --accent:    #2d6a4f;
    --accent2:   #b5873a;
    --accent3:   #e07b39;
    --text:      #1a1a18;
    --muted:     #8a8072;
    --danger:    #c0392b;
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
    background: #f7f5f0 !important;
    border: 1px solid #c8bfb2 !important;
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
/* ── BUTTONS — default inactive (beige) ───────────────────────────────── */
.stButton > button {
    background: #f7f5f0;
    color: #8a8072;
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid #c8bfb2;
    border-radius: 3px;
    padding: 0.55rem 1.25rem;
    width: 100%;
    cursor: pointer;
    position: relative;
    transition: background 0.15s ease, color 0.15s ease,
                transform 0.1s ease, box-shadow 0.1s ease,
                border-color 0.15s ease;
    box-shadow: 0 3px 0 #b0a898, 0 1px 4px rgba(0,0,0,0.08);
}
.stButton > button:hover {
    background: #f0ece4;
    border-color: #a8a098;
    color: #4a4a45;
    box-shadow: 0 3px 0 #a8a098, 0 2px 8px rgba(0,0,0,0.08);
    transform: translateY(-1px);
}
.stButton > button:active,
.stButton > button:focus:not(:focus-visible) {
    background: #e8e0d4;
    color: #4a4a45;
    border-color: #b0a898;
    transform: translateY(2px);
    box-shadow: 0 0px 0 #a8a098;
}
/* Run Optimization + Run Discovery — filled primary green */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: var(--accent);
    color: var(--bg);
    border-color: var(--accent);
    box-shadow: 0 4px 0 rgba(20,90,60,0.6), 0 2px 8px rgba(45,106,79,0.2);
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: #1e8f62;
    border-color: #1e8f62;
    color: var(--bg);
    box-shadow: 0 4px 0 rgba(25,110,75,0.6), 0 4px 16px rgba(45,106,79,0.25);
    transform: translateY(-1px);
}
[data-testid="stSidebar"] .stButton:last-of-type > button:active {
    background: #1a7a52;
    transform: translateY(3px);
    box-shadow: 0 1px 0 rgba(15,70,45,0.5);
}
/* Toggle */
[data-testid="stToggle"] { accent-color: var(--accent) !important; }
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
    background: rgba(45,106,79,0.15);
    color: var(--accent);
    border: 1px solid rgba(45,106,79,0.3);
}
.badge-warn {
    background: rgba(201,168,76,0.15);
    color: var(--accent3);
    border-color: rgba(201,168,76,0.3);
}
.badge-danger {
    background: rgba(192,57,43,0.15);
    color: var(--danger);
    border-color: rgba(192,57,43,0.3);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(247,245,240,0)",
    plot_bgcolor="rgba(247,245,240,0)",
    font=dict(family="IBM Plex Mono", color="#8a8072", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor="#e0d9ce", zerolinecolor="#e0d9ce", linecolor="#e0d9ce"),
    yaxis=dict(gridcolor="#e0d9ce", zerolinecolor="#e0d9ce", linecolor="#e0d9ce"),
    legend=dict(bgcolor="rgba(247,245,240,0)", bordercolor="#e0d9ce"),
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

def utility_obj(w, mu, cov, lam):
    """Arrow-Pratt utility: U = mu_p - (lambda/2) * sigma_p^2  (daily, annualized internally)"""
    w = w / w.sum()
    mu_p    = float(w @ mu) * 252
    sigma2_p = float(w @ cov @ w) * 252
    return -(mu_p - (lam / 2) * sigma2_p)  # negate for minimization

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ QuantFrame v2")
    st.markdown('<p class="app-subtitle">Portfolio Intelligence</p>', unsafe_allow_html=True)

    if "show_about" not in st.session_state:
        st.session_state.show_about = False
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Portfolio Lab"
    if "app_mode_radio" not in st.session_state:
        st.session_state.app_mode_radio = "analyze"
    if "optimize_weights" not in st.session_state:
        st.session_state.optimize_weights = False
    if "optimize_n" not in st.session_state:
        st.session_state.optimize_n     = False
        st.session_state.suggested_n    = 8
        st.session_state.max_assets_val = 8

    _cur_mode = st.session_state.app_mode_radio
    _analyze_active = _cur_mode == "analyze"
    st.markdown(f"""<style>
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(1) div[data-testid="stColumn"]:nth-child({'1' if _analyze_active else '2'}) button {{
    background: #2d6a4f !important; color: #f7f5f0 !important;
    border-color: #1a5c3a !important;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.4) !important;
    transform: translateY(2px) !important;
}}
</style>""", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        if st.button("ANALYZE", key="btn_mode_analyze", use_container_width=True):
            st.session_state.app_mode_radio = "analyze"
    with col_m2:
        if st.button("DISCOVER", key="btn_mode_discover", use_container_width=True):
            st.session_state.app_mode_radio = "discover"
    _cur_mode = st.session_state.app_mode_radio
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.1rem;margin-bottom:0.4rem;">
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'#2d6a4f' if _cur_mode == 'analyze' else 'transparent'};box-shadow:{'0 0 6px #2d6a4f' if _cur_mode == 'analyze' else 'none'};"></div></div>
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'#2d6a4f' if _cur_mode == 'discover' else 'transparent'};box-shadow:{'0 0 6px #2d6a4f' if _cur_mode == 'discover' else 'none'};"></div></div>
</div>""", unsafe_allow_html=True)

    app_mode = "  🔍  Discovery  " if _cur_mode == "discover" else "  ⬡  Lab  "
    st.session_state.app_mode  = app_mode
    st.session_state._app_mode = app_mode

    _sidebar_mode    = app_mode
    _is_disc_sidebar = _cur_mode == "discover"

    st.markdown("---")

    if not _is_disc_sidebar:
        # ══════════════════════════════════════════════════════════════════════
        # PORTFOLIO LAB SIDEBAR
        # ══════════════════════════════════════════════════════════════════════

        st.markdown("## Universe")
        preset = st.selectbox("Asset Universe", list(PRESET_UNIVERSES.keys()), index=0, label_visibility="collapsed")

        if preset == "Custom":
            custom_raw = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
            tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
            n_typed = len(tickers)
            if n_typed > 20:
                hint_color = "#b5873a"; hint_icon = "⚠"
                hint_text  = f"{n_typed} tickers · Large universes may slow optimization"
            else:
                hint_color = "#8a8072"; hint_icon = "·"
                hint_text  = f"{n_typed} ticker{'s' if n_typed != 1 else ''} · Recommended: 5–20 · No hard limit"
            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:{hint_color};margin-top:0.2rem;letter-spacing:0.04em;">{hint_icon} {hint_text}</div>', unsafe_allow_html=True)
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
        rf_input   = st.slider("Risk-Free Rate (%)", 0.0, 8.0, RF_RATE * 100, 0.25, format="%.2f")
        rf = rf_input / 100

        # ── Weight Constraint ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Weight Constraint")

        wt_opt = st.session_state.optimize_weights
        st.markdown(f"""<style>
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(2) div[data-testid="stColumn"]:nth-child({'1' if wt_opt else '2'}) button {{
    background: #2d6a4f !important; color: #f7f5f0 !important;
    border-color: #1a5c3a !important;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.4) !important;
    transform: translateY(2px) !important;
}}
</style>""", unsafe_allow_html=True)
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            if st.button("OPTIMIZE", key="btn_wt_opt", use_container_width=True):
                st.session_state.optimize_weights = True
        with col_w2:
            if st.button("MANUAL", key="btn_wt_manual", use_container_width=True):
                st.session_state.optimize_weights = False
        wt_opt = st.session_state.optimize_weights
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.1rem;margin-bottom:0.4rem;">
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'#2d6a4f' if wt_opt else 'transparent'};box-shadow:{'0 0 6px #2d6a4f' if wt_opt else 'none'};"></div></div>
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'transparent' if wt_opt else '#2d6a4f'};box-shadow:{'none' if wt_opt else '0 0 6px #2d6a4f'};"></div></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-bottom:0.25rem;background:{'#f0ece4' if wt_opt else 'transparent'};border:1px solid {'#d6cfc4' if wt_opt else 'transparent'};color:{'#8a8072' if wt_opt else 'transparent'};">
  {'◆ Unconstrained — optimizer controls allocation' if wt_opt else '◆'}
</div>""", unsafe_allow_html=True)
        slider_wt = st.slider("Max Single Asset Weight", 0.10, 1.0, value=1.0 if wt_opt else 0.40, step=0.05, format="%.2f", disabled=wt_opt)
        max_weight = 1.0 if wt_opt else slider_wt

        slider_min_wt = st.slider("Min Single Asset Weight", 0.00, 0.20, value=0.0, step=0.01, format="%.2f", disabled=wt_opt,
                                  help="Forces each active holding to carry at least this weight. Prevents token allocations.")
        min_weight = 0.0 if wt_opt else slider_min_wt

        # ── Diversification ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Diversification")
        st.caption("Max assets with nonzero weight in optimal portfolio")

        n_opt = st.session_state.optimize_n
        st.markdown(f"""<style>
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(3) div[data-testid="stColumn"]:nth-child({'1' if n_opt else '2'}) button {{
    background: #2d6a4f !important; color: #f7f5f0 !important;
    border-color: #1a5c3a !important;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.4) !important;
    transform: translateY(2px) !important;
}}
</style>""", unsafe_allow_html=True)
        col_n1, col_n2 = st.columns([1, 1])
        with col_n1:
            if st.button("OPTIMIZE", key="btn_optimize_n", use_container_width=True):
                st.session_state.optimize_n = True
        with col_n2:
            if st.button("MANUAL", key="btn_manual_n", use_container_width=True):
                st.session_state.optimize_n = False
        n_opt = st.session_state.optimize_n
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.1rem;margin-bottom:0.4rem;">
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'#2d6a4f' if n_opt else 'transparent'};box-shadow:{'0 0 6px #2d6a4f' if n_opt else 'none'};"></div></div>
  <div style="text-align:center;"><div style="width:5px;height:5px;border-radius:50%;margin:0 auto;background:{'transparent' if n_opt else '#2d6a4f'};box-shadow:{'none' if n_opt else '0 0 6px #2d6a4f'};"></div></div>
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#8a8072;line-height:1.55;margin:0.4rem 0 0.6rem 0;padding:0.5rem 0.65rem;background:#f7f5f0;border:1px solid #d6cfc4;border-radius:3px;">
  <b style="color:#1a1a18;">OPTIMIZE N</b> sets holdings to the portfolio's
  <b style="color:#2d6a4f;">effective N</b> = 1/Σwᵢ² — the number of assets the optimizer naturally concentrates into.
</div>""", unsafe_allow_html=True)

        n_opt = st.session_state.optimize_n
        sn    = st.session_state.suggested_n

        # ── Constraint linkage: compute hard cap on N BEFORE the slider renders
        # so the slider's max_value physically prevents infeasible combinations.
        # Rule: min_weight × N ≤ 1.0  →  N ≤ floor(1 / min_weight)
        if not wt_opt and not n_opt and min_weight > 0:
            _hard_max_n = max(2, int(np.floor(1.0 / min_weight)))
        else:
            _hard_max_n = 20

        # Also clamp the stored value so the slider doesn't open above its new max
        _stored_n = st.session_state.max_assets_val
        if _stored_n > _hard_max_n:
            st.session_state.max_assets_val = _hard_max_n
            _stored_n = _hard_max_n

        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;padding:0.3rem 0.65rem;background:#f7f5f0;border:1px solid #d6cfc4;border-radius:3px;margin-bottom:0.25rem;min-height:1.55rem;">
  <span style="color:{'#8a8072' if n_opt else 'transparent'};">Effective N = </span>
  <span style="color:{'#2d6a4f' if n_opt else 'transparent'};font-weight:600;">{sn}</span>
  <span style="color:{'#8a8072' if n_opt else 'transparent'};"> (auto)</span>
</div>""", unsafe_allow_html=True)

        slider_n = st.slider("Max Holdings (N)", min_value=2, max_value=_hard_max_n,
                             value=sn if n_opt else _stored_n, step=1, disabled=n_opt)
        if n_opt:
            max_assets = sn
        else:
            max_assets = slider_n
            st.session_state.max_assets_val = slider_n

        # Final safety clamp (covers edge cases like wt_opt/n_opt combos)
        if min_weight > 0 and min_weight * max_assets > 1.0:
            max_assets = max(2, int(np.floor(1.0 / min_weight)))
        if min_weight > max_weight:
            min_weight = max_weight

        # Feasibility indicator
        if not wt_opt:
            _floor_used = min_weight * max_assets * 100
            _color = "#2d6a4f" if _floor_used <= 80 else "#b5873a"
            _icon  = "✓" if _floor_used <= 80 else "⚠"
            st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:{_color};
            background:rgba(45,106,79,0.05);border:1px solid rgba(45,106,79,0.2);
            border-radius:3px;padding:0.45rem 0.65rem;margin-top:0.4rem;">
  {_icon} floor locks {_floor_used:.0f}% · {100-_floor_used:.0f}% free to optimize
</div>""", unsafe_allow_html=True)

        # ── Risk Tolerance ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Risk Tolerance")
        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;line-height:1.6;margin-bottom:0.75rem;">
Selects a portfolio on the efficient frontier via a <b style="color:#1a1a18;">utility function</b>:<br>
<span style="color:#2d6a4f;">max U = μ − (λ/2)σ²</span><br>
where λ is your risk aversion coefficient.<br>
Higher λ → closer to Min Variance.<br>
Lower λ → closer to Max Sharpe (Optimal Risky).
</div>
""", unsafe_allow_html=True)

        RISK_PRESETS = [
            ("NO GUTS — Min Variance Portfolio",      "minvar",   None,  "#2e86ab"),
            ("STEADY — Low Risk  (λ=10)",             "steady",   10.0,  "#2d6a4f"),
            ("AVERAGE — Moderate  (λ=4)",             "average",  4.0,   "#b5873a"),
            ("HIGH ROLLER — Aggressive  (λ=1.5)",     "roller",   1.5,   "#c0392b"),
            ("OPTIMAL RISKY — Tangency / Max Sharpe", "optimal",  None,  "#7b2d8b"),
            ("VARIABLE — Custom λ",                   "custom",   4.0,   "#5a7a6a"),
        ]
        PRESET_LABELS = [p[0] for p in RISK_PRESETS]
        key_to_idx    = {p[1]: i for i, p in enumerate(RISK_PRESETS)}

        if "risk_preset" not in st.session_state:
            st.session_state.risk_preset = "average"
        if "risk_lambda_val" not in st.session_state:
            st.session_state.risk_lambda_val = 4.0

        default_idx   = key_to_idx.get(st.session_state.risk_preset, 2)
        selected_label = st.selectbox("Risk Profile", PRESET_LABELS, index=default_idx, label_visibility="collapsed", key="risk_dropdown")
        active_preset  = RISK_PRESETS[PRESET_LABELS.index(selected_label)]

        if active_preset[1] != "custom" and active_preset[2] is not None:
            st.session_state.risk_lambda_val = active_preset[2]

        st.session_state.risk_preset = active_preset[1]
        risk_color = active_preset[3]
        short_name = active_preset[0].split("—")[0].strip()
        sub_name   = active_preset[0].split("—")[1].strip() if "—" in active_preset[0] else ""
        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;padding:0.45rem 0.75rem;margin-top:0.25rem;margin-bottom:0.5rem;background:#f0ece4;border-left:2px solid {risk_color};border-radius:0 3px 3px 0;">
  <span style="color:{risk_color};font-weight:600;">{short_name}</span>
  <span style="color:#8a8072;"> · {sub_name}</span>
</div>""", unsafe_allow_html=True)

        slider_disabled = active_preset[1] in ("minvar", "optimal")
        slider_val = st.slider("λ (risk aversion coefficient)", 0.5, 15.0,
                               value=st.session_state.risk_lambda_val, step=0.5, format="%.1f",
                               disabled=slider_disabled,
                               help="λ=0.5 → maximum risk. λ=15 → near min variance.")

        if not slider_disabled and active_preset[1] != "custom":
            expected = active_preset[2]
            if expected is not None and abs(slider_val - expected) > 0.01:
                st.session_state.risk_preset     = "custom"
                st.session_state.risk_lambda_val = slider_val
                st.rerun()

        if slider_disabled:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-top:0.25rem;background:#f0ece4;border:1px solid #d6cfc4;color:#8a8072;">◆ λ not applicable for this portfolio</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-top:0.25rem;background:#f0ece4;border:1px solid #d6cfc4;color:#8a8072;">Higher λ → less risk &nbsp;·&nbsp; Lower λ → more risk</div>', unsafe_allow_html=True)

        st.session_state.risk_lambda_val = slider_val

        if active_preset[1] == "minvar":
            effective_lambda = None; lambda_source = "minvar"
        elif active_preset[1] == "optimal":
            effective_lambda = None; lambda_source = "optimal"
        else:
            effective_lambda = slider_val; lambda_source = active_preset[1]

        st.markdown("---")
        if st.button("▶  Run Optimization"):
            st.session_state.run_optimization = True
        run_btn = st.session_state.get("run_optimization", False)

        st.markdown("---")
        with st.expander("ℹ Model Reference"):
            st.markdown("""
**Mean-Variance Optimization**  
Markowitz (1952): maximize Sharpe ratio on the efficient frontier.

**Utility-Based Selection**  
Arrow-Pratt: U = μ − (λ/2)σ². Selects where on the frontier you sit.

**VaR / CVaR**  
Historical simulation. CVaR = expected loss beyond VaR threshold.

**Rolling Beta**  
60-day rolling OLS vs SPY.

**Sortino / Calmar**  
Downside-adjusted return ratios.
            """)

    else:
        # ══════════════════════════════════════════════════════════════════════
        # DISCOVERY MODE SIDEBAR
        # ══════════════════════════════════════════════════════════════════════

        # Provide safe defaults for Portfolio Lab variables not set in this branch
        tickers = PRESET_UNIVERSES["Mega-Cap Tech"]
        period = "2y"; confidence = 0.95; rf = RF_RATE
        max_weight = 1.0; min_weight = 0.0; max_assets = 10
        effective_lambda = None; lambda_source = "optimal"
        active_preset = ("OPTIMAL RISKY — Tangency / Max Sharpe", "optimal", None, "#7b2d8b")
        run_btn = False

        st.markdown("## Discovery")

        _DISC_SECTORS = [
            "All S&P 500 (~490 tickers)",
            "Technology", "Healthcare", "Financials", "Energy",
            "Consumer Staples", "Industrials", "Consumer Discret",
        ]
        _disc_sector = st.selectbox("Sector Filter", _DISC_SECTORS,
                                    index=_DISC_SECTORS.index(st.session_state.get("disc_sector", "All S&P 500 (~490 tickers)")),
                                    label_visibility="collapsed")
        st.session_state.disc_sector = _disc_sector

        st.markdown("---")
        st.markdown("## Portfolio Size")
        _disc_port_size = st.slider("Stocks per combination", 5, 20, st.session_state.get("disc_port_size", 10), 1)
        st.session_state.disc_port_size = _disc_port_size

        st.markdown("---")
        st.markdown("## Iterations")
        _disc_iterations = st.slider("Number of combinations to test", 10, 5000, st.session_state.get("disc_iterations", 50), 10)
        st.session_state.disc_iterations = _disc_iterations

        st.markdown("---")
        st.markdown("## Lookback")
        _disc_start = st.selectbox("Start date", ["2018-01-01","2019-01-01","2020-01-01","2021-01-01"],
                                   index=["2018-01-01","2019-01-01","2020-01-01","2021-01-01"].index(
                                       st.session_state.get("disc_start", "2018-01-01")),
                                   label_visibility="collapsed")
        st.session_state.disc_start = _disc_start

        st.markdown("---")
        if st.button("🔍  Run Discovery", key="btn_discovery"):
            st.session_state.run_discovery = True
        else:
            if "run_discovery" not in st.session_state:
                st.session_state.run_discovery = False

        st.markdown("---")
        with st.expander("ℹ About Discovery"):
            st.markdown("""
**How it works**  
Randomly samples stock combinations from the selected universe and runs Max Sharpe optimization on each one.

**Search space**  
Combinatorially vast — you are sampling a small fraction of all possible portfolios.

**Result**  
The combination with the highest Sharpe ratio across all iterations is returned.

**Limitation**  
No guarantee of global optimality. Results vary across runs.
            """)

# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("""
<div style="margin-bottom:0.5rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:3rem;font-weight:600;
              color:#1a1a18;letter-spacing:-0.03em;line-height:1;">
    Quant<span style="color:#2d6a4f;">Frame</span><span style="color:#8a8072;font-size:1.4rem;font-weight:400;letter-spacing:0;vertical-align:super;margin-left:0.15em;">v2</span>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
              color:#8a8072;letter-spacing:0.18em;text-transform:uppercase;
              margin-top:0.4rem;">
    Modern Portfolio Theory &nbsp;·&nbsp; Factor Risk &nbsp;·&nbsp; Decision Analytics
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
              color:#b0a898;margin-top:0.5rem;letter-spacing:0.04em;">
    Built by&nbsp;
    <span style="color:#2d6a4f;font-weight:600;font-size:0.85rem;">Fulton Pace</span>
    <span style="color:#d6cfc4;margin:0 0.6rem;">·</span>
    Market data via&nbsp;<span style="color:#b5873a;font-weight:500;">Yahoo Finance</span>
    <span style="color:#d6cfc4;margin:0 0.6rem;">·</span>
    <a href="https://github.com/fultonpace/quantframe" target="_blank"
       style="color:#2e86ab;text-decoration:none;font-weight:500;">
      Open source ↗
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
with col_badge:
    st.markdown('<div style="padding-top:1.4rem;text-align:right;"><span class="badge" style="font-size:0.65rem;padding:0.25rem 0.75rem;">v2.0</span></div>', unsafe_allow_html=True)
    if st.button("ℹ  About", key="btn_about_open"):
        st.session_state.show_about = True

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── About Modal ───────────────────────────────────────────────────────────────
if st.session_state.get("show_about", False):
    if st.button("✕  Close", key="btn_about_close"):
        st.session_state.show_about = False
        st.rerun()
    st.markdown("""
<style>
.about-modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
}
.about-modal-title {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.about-modal-sub {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
}
.about-concept {
    margin-bottom: 1.5rem;
}
.about-concept-label {
    font-family: var(--mono);
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.about-concept-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.875rem;
    color: #4a4a45;
    line-height: 1.8;
}
.about-concept-formula {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent2);
    margin-top: 0.35rem;
    padding: 0.3rem 0.75rem;
    background: #f7f5f0;
    border-left: 2px solid var(--accent2);
    border-radius: 0 3px 3px 0;
    display: inline-block;
}
.about-divider-col {
    border-right: 1px solid var(--border);
    padding-right: 2rem;
    margin-right: 2rem;
}
.about-app-addresses {
    background: #f7f5f0;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
}
.about-app-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #ede7dc;
}
.about-app-row:last-child { border-bottom: none; }
.about-app-row-dot {
    width: 6px; height: 6px; border-radius: 50%;
    margin-top: 0.45rem; flex-shrink: 0;
}
.about-app-row-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    color: #4a4a45;
    line-height: 1.65;
}
.about-app-row-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)
    st.markdown(
        '<div class="about-modal">'
        '<div class="about-modal-title">About QuantFrame v2</div>'
        '<div class="about-modal-sub">Modern Portfolio Theory &nbsp;·&nbsp; Factor Risk &nbsp;·&nbsp; Decision Analytics</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2.5rem;">'
        '<div>'
        '<div class="about-concept">'
        '<div class="about-concept-label" style="color:#2d6a4f;">Modern Portfolio Theory</div>'
        '<div class="about-concept-body">'
        "Imagine you're packing a lunch. You could bring three bags of chips (same food, same risk), "
        "or bring chips, an apple, and a sandwich — things that don't all go stale at the same time."
        '<br><br>'
        'Modern Portfolio Theory (Markowitz, 1952) is that lunch logic applied to stocks. '
        'The key insight: <strong>combining assets that don\'t move in lockstep reduces overall risk '
        'without sacrificing return</strong>. The math finds the exact weights that maximize your '
        'reward per unit of risk — the Sharpe ratio — tracing a curve called the '
        '<strong>Efficient Frontier</strong>. Every portfolio on that curve is "optimal"; '
        'everything below it is wasteful.'
        '</div>'
        '<div class="about-concept-formula">max Sharpe = (R\u209a \u2212 R\u1da0) / \u03c3\u209a &nbsp;\u00b7&nbsp; Markowitz (1952)</div>'
        '</div>'
        '<div class="about-concept">'
        '<div class="about-concept-label" style="color:#b5873a;">Factor Risk Analytics</div>'
        '<div class="about-concept-body">'
        "Not all risk is the same. Some risk comes from the whole market moving (you can't "
        'avoid this — it\'s called <em>systematic</em> risk). Other risk is specific to one company '
        'going wrong (this you <em>can</em> diversify away).'
        '<br><br>'
        "Factor risk analytics breaks your portfolio's risk into these parts. "
        '<strong>Beta</strong> measures how much your portfolio amplifies market swings — a beta of 1.2 '
        'means when the S&amp;P 500 drops 10%, you drop 12%. <strong>VaR</strong> answers "on a bad day, '
        'how much could I lose?" and <strong>CVaR</strong> asks "on the worst days, what\'s my '
        'average loss?" Together they let you stress-test a portfolio before you\'re in it.'
        '</div>'
        '<div class="about-concept-formula">\u03b2 = Cov(R\u209a, R\u2098) / Var(R\u2098) &nbsp;\u00b7&nbsp; CVaR = E[loss | loss &gt; VaR]</div>'
        '</div>'
        '<div class="about-concept">'
        '<div class="about-concept-label" style="color:#2e86ab;">Decision Analytics</div>'
        '<div class="about-concept-body">'
        'Even with a perfect frontier and full risk breakdown, you still face a human question: '
        '<em>how much risk should I take?</em> Decision analytics brings your preference into the math.'
        '<br><br>'
        'QuantFrame encodes risk tolerance with a single number — <strong>\u03bb (lambda)</strong> — '
        'from the Arrow-Pratt utility function. Low \u03bb means you chase returns; high \u03bb means you '
        'prefer safety. The optimizer uses your \u03bb to select the one point on the frontier that '
        'matches your actual preference, turning subjective comfort into an objective weight vector.'
        '</div>'
        '<div class="about-concept-formula">U = \u03bc\u209a \u2212 (\u03bb/2)\u03c3\u209a\u00b2 &nbsp;\u00b7&nbsp; Arrow-Pratt utility</div>'
        '</div>'
        '</div>'
        '<div>'
        '<div class="about-concept-label" style="color:#2d6a4f;margin-bottom:0.75rem;">How QuantFrame Addresses Each</div>'
        '<div class="about-app-addresses">'
        '<div class="about-app-row">'
        '<div class="about-app-row-dot" style="background:#2d6a4f;box-shadow:0 0 5px #2d6a4f;"></div>'
        '<div class="about-app-row-text">'
        '<div class="about-app-row-label" style="color:#2d6a4f;">MPT \u2192 Efficient Frontier + Optimizer</div>'
        'Solves the full Markowitz optimization via SLSQP to find the tangency portfolio (max Sharpe), '
        'minimum variance portfolio, and utility-optimal portfolio. Visualizes the entire frontier so '
        'you can see where your allocation sits versus the theoretical best.'
        '</div>'
        '</div>'
        '<div class="about-app-row">'
        '<div class="about-app-row-dot" style="background:#b5873a;box-shadow:0 0 5px #b5873a;"></div>'
        '<div class="about-app-row-text">'
        '<div class="about-app-row-label" style="color:#b5873a;">Factor Risk \u2192 Live Risk Decomposition</div>'
        'Computes VaR and CVaR from actual daily return distributions (no distributional assumptions). '
        'Rolling 60-day beta against SPY shows how your market exposure changes over time. '
        'Sortino and Calmar ratios isolate downside risk specifically.'
        '</div>'
        '</div>'
        '<div class="about-app-row">'
        '<div class="about-app-row-dot" style="background:#2e86ab;box-shadow:0 0 5px #2e86ab;"></div>'
        '<div class="about-app-row-text">'
        '<div class="about-app-row-label" style="color:#2e86ab;">Decision Analytics \u2192 Risk Profile Selector</div>'
        'Six named risk presets (No Guts \u2192 High Roller) map directly to \u03bb values from the utility '
        'function. The Custom \u03bb slider lets you dial in any preference. The optimizer then '
        'automatically selects the mathematically correct portfolio for that exact risk tolerance.'
        '</div>'
        '</div>'
        '<div class="about-app-row">'
        '<div class="about-app-row-dot" style="background:#7b2d8b;box-shadow:0 0 5px #7b2d8b;"></div>'
        '<div class="about-app-row-text">'
        '<div class="about-app-row-label" style="color:#7b2d8b;">Discovery Mode \u2192 Portfolio Search</div>'
        "When you don't know which stocks to use, Discovery samples thousands of random combinations "
        'from the S&amp;P 500 and runs optimization on each \u2014 surfacing the combination with the highest '
        "Sharpe ratio. It's a brute-force search through the combinatorial space of portfolios."
        '</div>'
        '</div>'
        '<div class="about-app-row">'
        '<div class="about-app-row-dot" style="background:#c0392b;box-shadow:0 0 5px #c0392b;"></div>'
        '<div class="about-app-row-text">'
        '<div class="about-app-row-label" style="color:#c0392b;">\u26a0 Important Limitations</div>'
        'All results are backward-looking. Sample covariance is noisy; short lookback windows '
        'amplify estimation error. Transaction costs, taxes, and slippage are not modeled. '
        'Past performance does not predict future returns. This is an analytical tool, not investment advice.'
        '</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:1.5rem;">'
        '<div class="about-concept-label" style="color:#8a8072;margin-bottom:0.6rem;">Data &amp; Sources</div>'
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:#8a8072;line-height:2;">'
        'Prices &nbsp;\u00b7&nbsp; <span style="color:#b5873a;">Yahoo Finance</span> via yfinance \u2014 adjusted close (splits + dividends)<br>'
        'Benchmark &nbsp;\u00b7&nbsp; <span style="color:#1a1a18;">SPY</span> (SPDR S&amp;P 500 ETF Trust)<br>'
        'Optimization &nbsp;\u00b7&nbsp; <span style="color:#1a1a18;">SciPy SLSQP</span> \u2014 Markowitz (1952)<br>'
        'Universe &nbsp;\u00b7&nbsp; ~490 S&amp;P 500 tickers from public GitHub dataset<br>'
        'Cache &nbsp;\u00b7&nbsp; 1 hour per session'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Mode badge + run button ───────────────────────────────────────────────────
if "app_mode_radio" not in st.session_state:
    st.session_state.app_mode_radio = "analyze"

_cur_mode = st.session_state.get("app_mode_radio", "analyze")
app_mode      = "  🔍  Discovery  " if _cur_mode == "discover" else "  ⬡  Lab  "
st.session_state.app_mode  = app_mode
st.session_state._app_mode = app_mode

_is_discovery = _cur_mode == "discover"
_mode_label   = "🔍  Discover" if _is_discovery else "⬡  Analyze"
_mode_color   = "#2e86ab" if _is_discovery else "#2d6a4f"

_col_badge, _col_run = st.columns([6, 1])
with _col_badge:
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;font-weight:600;
            color:{_mode_color};letter-spacing:0.12em;text-transform:uppercase;
            padding:0.45rem 0;border-bottom:2px solid {_mode_color};display:inline-block;">
  {_mode_label}
</div>""", unsafe_allow_html=True)
with _col_run:
    if _is_discovery:
        if st.button("🔍  Run", key="btn_disc_top"):
            st.session_state.run_discovery = True
    else:
        if st.button("▶  Run", key="btn_run_top"):
            st.session_state.run_optimization = True

st.markdown('<hr class="divider" style="margin-top:0.5rem;">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY MODE
# ══════════════════════════════════════════════════════════════════════════════
if _cur_mode == "discover":
    st.session_state.run_optimization = False

    SECTORS = {
        "All S&P 500 (~490 tickers)": None,
        "Technology":      ["AAPL","MSFT","NVDA","AVGO","META","GOOGL","AMZN","AMD","QCOM","AMAT","MU","INTC","KLAC","LRCX","ADI","MCHP","SNPS","CDNS","ORCL","IBM","CRM","ADBE","NOW","INTU","PANW","CRWD","FTNT","ANET","HPE","TXN"],
        "Healthcare":      ["JNJ","UNH","LLY","ABT","MRK","TMO","DHR","ABBV","BMY","AMGN","GILD","MDT","ISRG","SYK","BSX","ZBH","BDX","EW","REGN","VRTX","BIIB","IQV","A","IDXX","RMD"],
        "Financials":      ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","V","MA","USB","PNC","TFC","COF","MTB","FITB","RF","CFG","HBAN","KEY","WBS","FNB","ZION"],
        "Energy":          ["XOM","CVX","COP","EOG","SLB","PXD","MPC","PSX","VLO","OXY","KMI","WMB","EQT","DVN","FANG","HAL","BKR","APA","MRO","HES"],
        "Consumer Staples":["PG","KO","PEP","WMT","COST","MDLZ","CL","KMB","GIS","HSY","MKC","SJM","HRL","CPB","K","CAG","CLX","CHD","SPB","COTY"],
        "Industrials":     ["HON","GE","MMM","CAT","DE","RTX","LMT","BA","GD","NOC","EMR","ITW","ETN","PH","ROK","CMI","IR","XYL","GNRC","JCI"],
        "Consumer Discret":["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","LOW","BKNG","CMG","YUM","DPZ","RCL","CCL","MAR","HLT","EXPE","LVS","MGM","WYNN"],
    }

    # Read all values from session state (set by sidebar)
    disc_sector     = st.session_state.get("disc_sector",     "All S&P 500 (~490 tickers)")
    disc_port_size  = st.session_state.get("disc_port_size",  10)
    disc_iterations = st.session_state.get("disc_iterations", 50)
    disc_start      = st.session_state.get("disc_start",      "2018-01-01")
    run_discovery   = st.session_state.get("run_discovery",   False)

    # ── Placeholder for idle card ─────────────────────────────────────────────
    _disc_top = st.empty()

    # ── Live time estimate ────────────────────────────────────────────────────
    SECS_PER_ITER = 2.5
    est_secs = disc_iterations * SECS_PER_ITER
    est_mins = est_secs / 60
    est_hrs  = est_mins / 60

    if est_secs < 60:
        est_str = f"~{int(est_secs)} seconds"; est_col = "#2d6a4f"; est_msg = "Quick run ☑"
    elif est_mins < 3:
        est_str = f"~{est_mins:.1f} minutes";  est_col = "#2d6a4f"; est_msg = "Grab a sip of water 💧"
    elif est_mins < 7:
        est_str = f"~{est_mins:.0f} minutes";  est_col = "#b5873a"; est_msg = "Go get a coffee ☕"
    elif est_mins < 12:
        est_str = f"~{est_mins:.0f} minutes";  est_col = "#b5873a"; est_msg = "Take a walk outside 🚶"
    elif est_mins < 20:
        est_str = f"~{est_mins:.0f} minutes";  est_col = "#c0392b"; est_msg = "Call your mom 📞"
    elif est_mins < 35:
        est_str = f"~{est_mins:.0f} minutes";  est_col = "#c0392b"; est_msg = "Watch an episode of something 📺"
    elif est_mins < 60:
        est_str = f"~{est_mins:.0f} minutes";  est_col = "#c0392b"; est_msg = "Hit the gym 🏋️ — seriously"
    elif est_hrs < 2:
        est_str = f"~{est_hrs:.1f} hours";     est_col = "#c0392b"; est_msg = "Take a nap. A real one. 😴"
    elif est_hrs < 3:
        est_str = f"~{est_hrs:.1f} hours";     est_col = "#c0392b"; est_msg = "Watch a full movie 🎬"
    else:
        est_str = f"~{est_hrs:.1f} hours";     est_col = "#c0392b"; est_msg = "Read War and Peace 📖"

    universe_size = len(SECTORS[disc_sector]) if SECTORS[disc_sector] else 490

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0;">
  <div class="metric-card">
    <div class="metric-label">Estimated Runtime</div>
    <div class="metric-value" style="font-size:1.1rem;color:{est_col};">{est_str}</div>
    <div class="metric-sub" style="color:{est_col};margin-top:0.3rem;">{est_msg}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Universe Size</div>
    <div class="metric-value" style="font-size:1.2rem;">{universe_size} tickers</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Combinations Tested</div>
    <div class="metric-value" style="font-size:1.2rem;">{disc_iterations:,}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Search Space</div>
    <div class="metric-value" style="font-size:1.2rem;color:#8a8072;">&#8776;10&#185;&#178;</div>
    <div class="metric-sub">combinatorially vast</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Sector Legend ─────────────────────────────────────────────────────────
    SECTOR_COLORS = {
        "Technology": "#2e86ab", "Healthcare": "#2d6a4f", "Financials": "#7b2d8b",
        "Energy": "#c0392b", "Consumer Staples": "#b5873a",
        "Industrials": "#5a7a6a", "Consumer Discret": "#e07b39",
    }
    with st.expander("📋  Sector Universe Reference", expanded=False):
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;color:#8a8072;margin-bottom:1rem;">Tickers in each sector filter. Selecting a sector dramatically reduces runtime by narrowing the search universe from ~490 to ~20–30 stocks.</div>', unsafe_allow_html=True)
        leg_cols = st.columns(2)
        for i, (sname, tlist) in enumerate([(k, v) for k, v in SECTORS.items() if v is not None]):
            with leg_cols[i % 2]:
                scolor = SECTOR_COLORS.get(sname, "#8a8072")
                st.markdown(
                    '<div style="background:#ffffff;border:1px solid #e0d9ce;border-left:3px solid ' + scolor + ';border-radius:0 4px 4px 0;padding:0.75rem 1rem;margin-bottom:0.75rem;">'
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;font-weight:600;color:' + scolor + ';letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">' + sname + ' &middot; ' + str(len(tlist)) + ' tickers</div>'
                    '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;color:#8a8072;line-height:1.8;word-break:break-word;">' + '  &middot;  '.join(tlist) + '</div></div>',
                    unsafe_allow_html=True)

    # ── Now fill the top placeholder with idle card, or run ───────────────────
    if not run_discovery:
        import math as _math
        _univ_size = len(SECTORS[disc_sector]) if SECTORS[disc_sector] else 490
        try:
            _n_combos = _math.comb(_univ_size, disc_port_size)
            if _n_combos > 1e12:   _combo_str = "&#8776; " + f"{_n_combos:.1e}"
            elif _n_combos > 1e9:  _combo_str = "&#8776; " + f"{_n_combos/1e9:.1f}B"
            elif _n_combos > 1e6:  _combo_str = "&#8776; " + f"{_n_combos/1e6:.1f}M"
            else:                  _combo_str = f"{_n_combos:,}"
            _cov_pct = disc_iterations / _n_combos * 100
            _cov_str = f"{_cov_pct:.4f}%" if _cov_pct < 0.01 else f"{_cov_pct:.2f}%"
        except Exception:
            _combo_str = "vast"; _cov_str = "&lt;0.01%"

        _disc_top.markdown(
            '<div style="max-width:640px;margin:0 auto 1rem auto;font-family:\'IBM Plex Mono\',monospace;'
            'background:#ffffff;border:1px solid #e0d9ce;border-radius:6px;padding:2rem 2.25rem;">'
            '<div style="font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">Ready to search</div>'
            '<div style="font-size:1rem;font-weight:600;color:#1a1a18;margin-bottom:1.25rem;">Stochastic Portfolio Search</div>'
            '<div style="font-size:0.72rem;color:#4a4a45;line-height:2;">'
            'Discovery randomly samples combinations of stocks from your chosen universe and optimizes each one for '
            '<b style="color:#2d6a4f;">maximum Sharpe ratio</b>. '
            'The search is a best-effort heuristic — it cannot guarantee the globally optimal portfolio, '
            'but with enough iterations it will surface strong candidates that pure theory alone would not prescribe. '
            'The combinatorial space is astronomically large. You are sampling a small slice of it. '
            'The best combination found is returned with full allocation and performance statistics.'
            '</div>'
            '</div>'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#8a3030;'
            'background:#fdf0ef;border:1px solid #e8c4c0;border-radius:4px;'
            'padding:0.65rem 1rem;margin-top:0.75rem;">'
            '&#9888;&nbsp;&nbsp;<b>Warning:</b> larger runs can take 30+ minutes. '
            'Check estimated run time before executing.'
            '</div>',
            unsafe_allow_html=True)
    else:
        # ── Run discovery ──────────────────────────────────────────────────────
        import random as _random

        # Load universe
        with st.spinner("Loading S&P 500 universe…"):
            if SECTORS[disc_sector]:
                universe = SECTORS[disc_sector]
            else:
                try:
                    sp_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
                    sp500  = pd.read_csv(sp_url)
                    universe = [t for t in sp500['Symbol'].tolist() if "." not in t]
                except:
                    st.error("Could not load S&P 500 universe. Try a sector filter instead.")
                    st.stop()

        if len(universe) < disc_port_size:
            st.error(f"Universe too small ({len(universe)} tickers) for portfolio size {disc_port_size}.")
            st.stop()

        best = {"sharpe": -np.inf, "stocks": None, "weights": None, "ret": None, "vol": None}
        history = []

        progress_bar = st.progress(0, text="Starting discovery…")
        status_col1, status_col2, status_col3 = st.columns(3)
        best_sharpe_display  = status_col1.empty()
        best_stocks_display  = status_col2.empty()
        iter_display         = status_col3.empty()

        rf_disc = 0.0525

        for i in range(disc_iterations):
            stocks = _random.sample(universe, disc_port_size)
            try:
                raw = yf.download(stocks, start=disc_start, end="2024-12-31",
                                  auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    data = raw["Close"]
                else:
                    data = raw
                data = data.dropna()
                if data.shape[0] < 60 or data.shape[1] < 2:
                    continue

                rets = np.log(data / data.shift(1)).dropna()
                noa  = rets.shape[1]
                actual_stocks = list(rets.columns)

                mu_d   = rets.mean().values
                cov_d  = rets.cov().values

                def _neg_sharpe_d(w):
                    r = float(w @ mu_d) * 252
                    v = float(np.sqrt(w @ cov_d @ w)) * np.sqrt(252)
                    return -(r - rf_disc) / v if v > 0 else 1e6

                w0_d   = np.ones(noa) / noa
                bounds_d = tuple((0, 1) for _ in range(noa))
                cons_d   = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
                res_d    = minimize(_neg_sharpe_d, w0_d, method="SLSQP",
                                    bounds=bounds_d, constraints=cons_d)

                if not res_d.success:
                    continue

                w_d    = res_d.x / res_d.x.sum()
                p_ret  = float(w_d @ mu_d) * 252
                p_vol  = float(np.sqrt(w_d @ cov_d @ w_d)) * np.sqrt(252)
                sr     = (p_ret - rf_disc) / p_vol if p_vol > 0 else -np.inf

                history.append({"iter": i + 1, "sharpe": sr})

                if sr > best["sharpe"]:
                    best = {"sharpe": sr, "stocks": actual_stocks,
                            "weights": w_d, "ret": p_ret, "vol": p_vol}

            except Exception:
                continue

            pct = (i + 1) / disc_iterations
            progress_bar.progress(pct, text=f"Iteration {i+1} / {disc_iterations}  ·  Best Sharpe so far: {best['sharpe']:.3f}")
            best_sharpe_display.metric("Best Sharpe", f"{best['sharpe']:.4f}" if best['sharpe'] > -np.inf else "—")
            best_stocks_display.metric("Best Combo", ", ".join(best['stocks'][:3]) + "…" if best['stocks'] else "—")
            iter_display.metric("Iterations", f"{i+1} / {disc_iterations}")

        progress_bar.progress(1.0, text="Discovery complete!")
        st.session_state.run_discovery = False  # reset so it doesn't re-run on next interaction

        if best["stocks"] is None:
            st.error("No valid portfolios found. Try increasing iterations or changing the sector.")
            st.stop()

        # ── Results ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Discovery Results · Best Portfolio Found</div>', unsafe_allow_html=True)

        res_cols = st.columns(5)
        for col, (label, val, color) in zip(res_cols, [
            ("Sharpe Ratio",        f"{best['sharpe']:.4f}",          "#2d6a4f"),
            ("Ann. Return",         f"{best['ret']*100:.2f}%",         "#2d6a4f"),
            ("Ann. Volatility",     f"{best['vol']*100:.2f}%",         "#1a1a18"),
            ("Portfolio Size",      f"{len(best['stocks'])} stocks",   "#b5873a"),
            ("Iterations Run",      f"{disc_iterations}",              "#8a8072"),
        ]):
            with col:
                st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{val}</div>
    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Weights bar chart
        df_disc = pd.DataFrame({
            "Ticker": best["stocks"],
            "Weight (%)": (best["weights"] * 100).round(2)
        }).sort_values("Weight (%)", ascending=False)

        fig_disc = go.Figure(go.Bar(
            x=df_disc["Ticker"], y=df_disc["Weight (%)"],
            marker_color="#2d6a4f", marker_line_width=0, opacity=0.85,
        ))
        fig_disc.update_layout(**{**PLOT_LAYOUT,
            "height": 320,
            "yaxis_title": "Weight (%)",
            "title": dict(text="Optimal Weights · Best Discovered Portfolio",
                          font=dict(size=12, color="#1a1a18")),
        })
        st.plotly_chart(fig_disc, use_container_width=True)

        # Sharpe history line
        if history:
            df_hist = pd.DataFrame(history)
            df_hist["best_so_far"] = df_hist["sharpe"].cummax()
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df_hist["iter"], y=df_hist["sharpe"],
                mode="markers", marker=dict(size=4, color="#e07b39", opacity=0.5),
                name="Each Iteration",
            ))
            fig_hist.add_trace(go.Scatter(
                x=df_hist["iter"], y=df_hist["best_so_far"],
                mode="lines", line=dict(color="#2d6a4f", width=2),
                name="Best So Far",
            ))
            fig_hist.update_layout(**{**PLOT_LAYOUT,
                "height": 300,
                "xaxis_title": "Iteration",
                "yaxis_title": "Sharpe Ratio",
                "title": dict(text="Sharpe Ratio Across Iterations",
                              font=dict(size=12, color="#1a1a18")),
            })
            st.plotly_chart(fig_hist, use_container_width=True)

        # Full allocation table
        st.markdown('<div class="section-header">Optimal Allocation</div>', unsafe_allow_html=True)
        fig_disc_tbl = go.Figure(go.Table(
            columnwidth=[80, 100, 100, 100],
            header=dict(
                values=["<b>Ticker</b>", "<b>Weight (%)</b>", "<b>Sector Filter</b>", "<b>Start Date</b>"],
                fill_color="#f0ece4", line_color="#c8bfb2",
                font=dict(family="IBM Plex Mono", size=11, color="#2d6a4f"),
                align="center", height=32,
            ),
            cells=dict(
                values=[
                    df_disc["Ticker"].tolist(),
                    df_disc["Weight (%)"].tolist(),
                    [disc_sector] * len(df_disc),
                    [disc_start] * len(df_disc),
                ],
                fill_color="#ffffff",
                line_color="#e0d9ce",
                font=dict(family="IBM Plex Mono", size=11, color="#1a1a18"),
                align="center", height=28,
            )
        ))
        fig_disc_tbl.update_layout(**{**PLOT_LAYOUT,
            "height": 60 + 28 * len(df_disc) + 32,
            "margin": dict(l=0, r=0, t=0, b=0)
        })
        st.plotly_chart(fig_disc_tbl, use_container_width=True)

        st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#8a8072;
            padding:0.75rem 1rem;background:#f7f5f0;border:1px solid #e0d9ce;border-radius:4px;margin-top:1rem;">
      <b style="color:#1a1a18;">Best stocks found:</b> {', '.join(best['stocks'])} &nbsp;·&nbsp;
      <b style="color:#1a1a18;">Sharpe:</b> {best['sharpe']:.4f} &nbsp;·&nbsp;
      <b style="color:#1a1a18;">Return:</b> {best['ret']*100:.2f}% &nbsp;·&nbsp;
      <b style="color:#1a1a18;">Vol:</b> {best['vol']*100:.2f}% &nbsp;·&nbsp;
      {disc_iterations} iterations over {disc_sector}
    </div>""", unsafe_allow_html=True)

    st.stop()


# ── PORTFOLIO LAB (existing app) ──────────────────────────────────────────────
# ── Main Logic ───────────────────────────────────────────────────────────────
if not run_btn:
    _n_assets       = len(tickers)
    _wt_pct         = int(max_weight * 100)
    _min_wt_pct     = int(min_weight * 100)
    _rf_pct         = f"{rf*100:.2f}"
    _conf_pct       = int(confidence * 100)
    _effective_n    = min(max_assets, _n_assets)
    if effective_lambda is not None:
        _lam_display = f"λ = {effective_lambda:.1f}"
    elif lambda_source == "minvar":
        _lam_display = "Min Variance"
    else:
        _lam_display = "Max Sharpe"
    _preset_display = active_preset[0].split("—")[0].strip()
    _preset_color   = active_preset[3]

    _html = (
        '<div style="max-width:640px;margin:1.5rem auto 2rem auto;font-family:\'IBM Plex Mono\',monospace;">'
        '<div style="background:#ffffff;border:1px solid #e0d9ce;border-radius:6px;padding:2rem 2.25rem;">'
        '<div style="font-size:0.58rem;letter-spacing:0.18em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">Ready to optimize</div>'
        '<div style="font-size:1rem;font-weight:600;color:#1a1a18;margin-bottom:1.25rem;">Mean-Variance Optimization</div>'
        '<div style="font-size:0.72rem;color:#4a4a45;line-height:2;">'
        'The optimizer will find the allocation across your selected universe that maximizes '
        'risk-adjusted return — specifically, the <b style="color:#2d6a4f;">Sharpe ratio</b>. '
        'It will respect your weight bounds and holdings limit, then deliver three portfolios for comparison: '
        'the <b style="color:#7b2d8b;">tangency portfolio</b> that sits on the efficient frontier at maximum Sharpe, '
        'the <b style="color:#2e86ab;">minimum variance</b> portfolio that takes the least risk for any given return, '
        'and an <b style="color:#e07b39;">equal-weight baseline</b>. '
        f'Your <b style="color:{_preset_color};">selected risk profile</b> determines which of these '
        'is highlighted as your primary portfolio across all tabs.'
        '</div>'
        '</div></div>'
    )
    st.markdown(_html, unsafe_allow_html=True)
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
    # Guard: min_weight × effective holdings must be ≤ 1
    _effective_n_check = min(max_assets, n)
    if min_weight > 0 and min_weight * _effective_n_check > 1.0:
        max_assets = int(np.floor(1.0 / min_weight))
        st.warning(f"Min weight × holdings would exceed 100%. Holdings cap reduced to {max_assets} to maintain feasibility.")

    bounds      = tuple((min_weight, max_weight) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n
    n_keep      = min(max_assets, n)

    def _apply_cardinality(w_full, k):
        top = np.argsort(w_full)[-k:]
        w   = np.zeros(len(w_full))
        w[top] = w_full[top]
        # Enforce min_weight on retained positions
        if min_weight > 0:
            for idx in top:
                if w[idx] < min_weight:
                    w[idx] = min_weight
        total = w.sum()
        return w / total if total > 0 else w

    # ── Max Sharpe (Optimal Risky / Tangency) ─────────────────────────────────
    # Run UNCONSTRAINED first to compute Effective N
    bounds_free   = tuple((0.0, 1.0) for _ in range(n))
    res_sharpe_uc = minimize(neg_sharpe, w0, args=(mu, cov / 252, rf),
                             method="SLSQP", bounds=bounds_free, constraints=constraints)
    w_sharpe_uc   = res_sharpe_uc.x / res_sharpe_uc.x.sum()

    # Effective N = inverse Herfindahl index
    n_eff = int(round(1.0 / float(np.sum(w_sharpe_uc ** 2))))
    n_eff = max(2, min(n_eff, n))
    st.session_state.suggested_n = n_eff

    # If OPTIMIZE N mode, override n_keep with effective N
    if st.session_state.get("optimize_n", False):
        n_keep = n_eff

    res_sharpe    = minimize(neg_sharpe, w0, args=(mu, cov / 252, rf),
                             method="SLSQP", bounds=bounds, constraints=constraints)
    w_sharpe      = _apply_cardinality(res_sharpe.x / res_sharpe.x.sum(), n_keep)

    # ── Min Volatility (No Guts / Min Variance) ───────────────────────────────
    res_minvol    = minimize(min_vol_obj, w0, args=(mu, cov / 252),
                             method="SLSQP", bounds=bounds, constraints=constraints)
    w_minvol      = _apply_cardinality(res_minvol.x / res_minvol.x.sum(), n_keep)

    # ── Utility-based portfolio (risk tolerance selection) ────────────────────
    if lambda_source == "minvar" or lambda_source == "optimal":
        # Preset maps directly to an already-computed portfolio
        w_utility = w_minvol.copy() if lambda_source == "minvar" else w_sharpe.copy()
    else:
        lam_eff = effective_lambda if effective_lambda is not None else 4.0
        res_util = minimize(utility_obj, w0, args=(mu, cov / 252, lam_eff),
                            method="SLSQP", bounds=bounds, constraints=constraints)
        w_utility = _apply_cardinality(res_util.x / res_util.x.sum(), n_keep)

    # ── Equal Weight baseline ─────────────────────────────────────────────────
    w_eq = np.ones(n) / n

    # ── Efficient Frontier ────────────────────────────────────────────────────
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(mu, cov / 252)

    # ── Primary display portfolio = utility selection ─────────────────────────
    # (used in risk analytics, rolling metrics, report tabs)
    w_primary = w_utility

# ── Data Source Panel ─────────────────────────────────────────────────────────
import datetime as _dt
_now        = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
_date_start = prices.index[0].strftime("%Y-%m-%d")
_date_end   = prices.index[-1].strftime("%Y-%m-%d")
_n_obs      = len(returns)
_active     = [t for t in valid_tickers if w_primary[valid_tickers.index(t)] > 0.001]
_preset_lbl = active_preset[0].split("—")[0].strip()
_preset_sub = active_preset[0].split("—")[1].strip() if "—" in active_preset[0] else active_preset[0]
_preset_col = active_preset[3]

st.markdown(f"""
<div style="background:#f0ece4;border:1px solid #e0d9ce;border-radius:4px;
            padding:1rem 1.5rem;margin-bottom:1.5rem;
            display:grid;grid-template-columns:repeat(7,1fr);gap:1rem;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Data Source</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#2d6a4f;font-weight:600;">
      Yahoo Finance</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">via yfinance</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Date Range</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#1a1a18;font-weight:500;">
      {_date_start}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">→ {_date_end}</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Observations</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#1a1a18;font-weight:500;">
      {_n_obs:,}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">trading days</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Universe</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#1a1a18;font-weight:500;">
      {len(valid_tickers)} assets</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">
      {', '.join(valid_tickers)}</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Active Holdings</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#2d6a4f;font-weight:600;">
      {len(_active)} / {len(valid_tickers)}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">
      N={n_keep} constraint</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Risk Profile</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;font-weight:600;
                color:{_preset_col};">{_preset_lbl}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">{_preset_sub}</div>
  </div>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;margin-bottom:0.3rem;">
      Last Fetched</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#1a1a18;font-weight:500;">
      {_now}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;">adj. close prices</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Portfolio stats ───────────────────────────────────────────────────────────
def port_series(weights, ret_df):
    return (ret_df[valid_tickers] @ weights)

rf_daily = rf / 252
bench_ret = bench.pct_change().dropna()

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
    r_sh, v_sh, s_sh   = portfolio_stats(w_sharpe,  mu, cov / 252, rf)
    r_mv, v_mv, s_mv   = portfolio_stats(w_minvol,  mu, cov / 252, rf)
    r_eq, v_eq, s_eq   = portfolio_stats(w_eq,       mu, cov / 252, rf)
    r_ut, v_ut, s_ut   = portfolio_stats(w_primary,  mu, cov / 252, rf)

    sharpes_frontier = [(r_ - rf) / v_ if v_ > 0 else 0
                        for r_, v_ in zip(frontier_rets, frontier_vols)]

    fig = go.Figure()

    # Individual assets scatter — labels in legend/hover only, no on-chart text
    fig.add_trace(go.Scatter(
        x=asset_ann_vol * 100, y=asset_ann_ret * 100,
        mode="markers",
        text=valid_tickers,
        marker=dict(size=9, color="#e0d9ce", line=dict(color="#8a8072", width=1.5)),
        name="Individual Assets",
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Frontier — smooth curve wrapping the portfolio opportunity set
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100, y=frontier_rets * 100,
        mode="lines",
        line=dict(color="rgba(45,106,79,0.85)", width=2.5),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Capital Market Line (always anchored to true tangency = w_sharpe)
    if v_sh > 0:
        cml_x = np.linspace(0, max(asset_ann_vol) * 1.3, 100)
        cml_y = rf + (r_sh - rf) / v_sh * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x * 100, y=cml_y * 100,
            mode="lines",
            line=dict(color="rgba(181,135,58,0.5)", width=1.5, dash="dash"),
            name="Capital Market Line",
        ))

    # Portfolio markers — markers only, no text
    port_points = [
        (v_sh, r_sh, "Optimal Risky (Max Sharpe)", "#7b2d8b", "star",    18),
        (v_mv, r_mv, "Min Variance Portfolio",      "#2e86ab", "diamond", 16),
        (v_eq, r_eq, "Equal Weight",                "#e07b39", "circle",  12),
    ]
    for v_, r_, name_, color_, sym_, sz_ in port_points:
        fig.add_trace(go.Scatter(
            x=[v_ * 100], y=[r_ * 100],
            mode="markers",
            marker=dict(size=sz_, color=color_, symbol=sym_,
                        line=dict(color="#f7f5f0", width=1)),
            name=name_,
            hovertemplate=f"<b>{name_}</b><br>Vol: {v_*100:.2f}%<br>Return: {r_*100:.2f}%<br>Sharpe: {(r_-rf)/v_:.3f}<extra></extra>",
        ))

    # User's selected portfolio (if different from the above)
    is_custom = lambda_source not in ("minvar", "optimal")
    if is_custom:
        fig.add_trace(go.Scatter(
            x=[v_ut * 100], y=[r_ut * 100],
            mode="markers",
            marker=dict(size=16, color=_preset_col, symbol="pentagon",
                        line=dict(color="#f7f5f0", width=1.5)),
            name=f"Selected: {_preset_lbl}",
            hovertemplate=f"<b>Your Portfolio</b><br>λ={effective_lambda}<br>Vol: {v_ut*100:.2f}%<br>Return: {r_ut*100:.2f}%<br>Sharpe: {s_ut:.3f}<extra></extra>",
        ))

    layout = dict(**PLOT_LAYOUT)
    layout.update(dict(
        title=dict(text="Efficient Frontier & Portfolio Compositions", font=dict(size=13, color="#1a1a18")),
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        height=520,
    ))
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Portfolio comparison metrics
    st.markdown('<div class="section-header">Portfolio Comparison</div>', unsafe_allow_html=True)

    # Determine label for each named portfolio — append "(your pick)" if it matches selection
    def _card_label(base_name, w_card):
        is_sel = np.allclose(w_card, w_primary, atol=1e-4)
        return f"{base_name}\n(your pick)" if is_sel else base_name

    card_list = [
        (_card_label("Min Variance",  w_minvol), w_minvol, r_mv, v_mv, s_mv, "#2e86ab"),
        (_card_label("Equal Weight",  w_eq),     w_eq,     r_eq, v_eq, s_eq, "#e07b39"),
        (_card_label("Optimal Risky", w_sharpe), w_sharpe, r_sh, v_sh, s_sh, "#7b2d8b"),
    ]
    # Only add a separate "Your Portfolio" card if it is a custom utility portfolio
    # (i.e. not identical to any of the three named ones)
    if is_custom:
        card_list.insert(2, (f"Your Pick\n({_preset_lbl})", w_primary, r_ut, v_ut, s_ut, _preset_col))

    cols = st.columns(len(card_list))
    for col, (label, weights, ret, vol, sharpe, color) in zip(cols, card_list):
        is_selected = np.allclose(weights, w_primary, atol=1e-4)
        border_style = f"box-shadow:0 0 0 1px {color}; " if is_selected else ""
        active_html = (
            f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.15rem;">'
            f'<span class="badge" style="background:rgba(45,106,79,0.12);color:{color};border-color:{color}60;">ACTIVE</span>'
            f'<span style="font-family:var(--mono);font-size:0.55rem;color:{color};opacity:0.8;text-align:right;">your current<br>risk selection</span>'
            f'</div>'
        ) if is_selected else ""
        with col:
            st.markdown(f"""
<div class="metric-card" style="--accent:{color};{border_style}">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.75rem;">
  <div class="metric-label" style="white-space:pre;">{label}</div>
  {active_html}
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
  <div>
    <div class="metric-label">Ann. Return</div>
    <div class="metric-value" style="font-size:1.1rem;color:{color};">{ret*100:.2f}%</div>
  </div>
  <div>
    <div class="metric-label">Ann. Vol</div>
    <div class="metric-value" style="font-size:1.1rem;">{vol*100:.2f}%</div>
  </div>
  <div>
    <div class="metric-label">Sharpe</div>
    <div class="metric-value" style="font-size:1.1rem;color:{color};">{sharpe:.3f}</div>
  </div>
  <div>
    <div class="metric-label">Max Wt</div>
    <div class="metric-value" style="font-size:1.1rem;">{weights.max()*100:.1f}%</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Numerical weights table
    st.markdown(f'<div class="section-header">Exact Weights · {_preset_lbl} (your selection)</div>', unsafe_allow_html=True)

    _wt_cols = [
        ("Min Variance",  w_minvol,  "#2e86ab",  False),
        ("Equal Weight",  w_eq,      "#e07b39",  False),
        ("Optimal Risky", w_sharpe,  "#7b2d8b",  np.allclose(w_sharpe, w_primary, atol=1e-4)),
    ]
    if is_custom:
        _wt_cols.insert(0, (_preset_lbl, w_primary, _preset_col, True))

    _sort_w   = _wt_cols[0][1]
    _sort_idx = np.argsort(_sort_w)[::-1]
    _sorted_tickers = [valid_tickers[i] for i in _sort_idx]
    _max_wt = max(w.max() * 100 for _, w, _, _ in _wt_cols)

    def _bar_html(val, max_val, color, is_active):
        pct = min(val / max_val * 100, 100) if max_val > 0 else 0
        opacity = "1" if is_active else "0.55"
        return (
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="flex:1;background:#f0ece4;border-radius:2px;height:6px;">'
            f'<div style="width:{pct:.1f}%;height:6px;border-radius:2px;background:{color};opacity:{opacity};"></div>'
            f'</div>'
            f'<span style="min-width:38px;text-align:right;font-weight:{"600" if is_active else "400"};'
            f'color:{"#1a1a18" if is_active else "#5a5248"};">{val:.2f}%</span>'
            f'</div>'
        )

    _header_cells = '<th style="width:72px;text-align:left;padding:10px 14px;background:#f0ece4;border-bottom:2px solid #c8bfb2;font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:#8a8072;font-weight:500">Ticker</th>'
    for lbl, _, color, is_active in _wt_cols:
        _active_badge = f' <span style="font-size:0.55rem;background:{color}22;color:{color};border:1px solid {color}55;border-radius:2px;padding:1px 5px;letter-spacing:0.06em;">ACTIVE</span>' if is_active else ""
        _header_cells += (
            f'<th style="text-align:left;padding:10px 14px;background:{"#f7f5f2" if is_active else "#f0ece4"};'
            f'border-bottom:2px solid {""+color if is_active else "#c8bfb2"};'
            f'border-left:{"3px solid "+color if is_active else "1px solid #e0d9ce"};'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{color};font-weight:600">'
            f'{lbl}{_active_badge}</th>'
        )

    _rows_html = ""
    for row_i, ticker in enumerate(_sorted_tickers):
        orig_i = valid_tickers.index(ticker)
        _row_bg = "#ffffff" if row_i % 2 == 0 else "#faf8f5"
        _row_html = (
            f'<tr style="background:{_row_bg};">'
            f'<td style="padding:9px 14px;font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
            f'font-weight:600;color:#1a1a18;border-bottom:1px solid #ede8e0;'
            f'border-right:1px solid #e0d9ce;white-space:nowrap;">{ticker}</td>'
        )
        for lbl, w_arr, color, is_active in _wt_cols:
            val = w_arr[orig_i] * 100
            _cell_bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)" if is_active else _row_bg
            _row_html += (
                f'<td style="padding:9px 14px;border-bottom:1px solid #ede8e0;'
                f'border-left:{"3px solid "+color if is_active else "1px solid #e8e2d8"};'
                f'background:{_cell_bg};">'
                f'{_bar_html(val, _max_wt, color, is_active)}'
                f'</td>'
            )
        _row_html += "</tr>"
        _rows_html += _row_html

    st.markdown(f"""
<div style="overflow-x:auto;border:1px solid #e0d9ce;border-radius:6px;margin-bottom:1rem;">
  <table style="width:100%;border-collapse:collapse;">
    <thead><tr>{_header_cells}</tr></thead>
    <tbody>{_rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # ── Weights bar chart — no duplicate bars
    # If user's selection matches a named portfolio, just rename that bar.
    # Only show 4 distinct portfolios max; never show same weights twice.
    st.markdown(f'<div class="section-header">Portfolio Weights · Visual Comparison</div>', unsafe_allow_html=True)

    # Build deduplicated bar list
    def _bar_label(base, w_check):
        if np.allclose(w_check, w_primary, atol=1e-4):
            return f"{base} (your pick)"
        return base

    bar_entries = [
        (_bar_label("Min Variance",  w_minvol), w_minvol, "#2e86ab"),
        (_bar_label("Equal Weight",  w_eq),     w_eq,     "#e07b39"),
        (_bar_label("Optimal Risky", w_sharpe), w_sharpe, "#7b2d8b"),
    ]
    # Add custom utility bar only if it's not identical to any named portfolio
    if is_custom:
        bar_entries.insert(0, (f"{_preset_lbl} (your pick)", w_primary, _preset_col))

    df_bar = pd.DataFrame({"Ticker": valid_tickers})
    for lbl, w_arr, _ in bar_entries:
        df_bar[lbl] = w_arr * 100
    df_bar = df_bar.sort_values(bar_entries[0][0], ascending=False)

    fig2 = go.Figure()
    for lbl, _, color in bar_entries:
        fig2.add_trace(go.Bar(
            name=lbl, x=df_bar["Ticker"], y=df_bar[lbl],
            marker_color=color, marker_line_width=0,
            opacity=0.85,
        ))
    fig2.update_layout(**{**PLOT_LAYOUT,
        "barmode": "group", "height": 340,
        "yaxis_title": "Weight (%)",
        "title": dict(text="Portfolio Weight Comparison", font=dict(size=12, color="#1a1a18")),
    })
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">Risk Decomposition · {_preset_lbl} Portfolio · VaR · CVaR · Drawdown</div>', unsafe_allow_html=True)

    port_ret_series = port_series(w_primary, returns)
    port_ret_arr    = port_ret_series.values

    # VaR / CVaR
    var_h,  cvar_h  = compute_var_cvar(port_ret_arr, confidence)
    sortino = compute_sortino(port_ret_arr, rf_daily)
    cum_ret = (1 + port_ret_series).cumprod()
    max_dd  = compute_max_drawdown(cum_ret)
    calmar  = compute_calmar(r_ut, max_dd)

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
            marker_color="rgba(45,106,79,0.5)",
            marker_line_width=0,
            name="Daily Returns",
        ))
        # Normal fit
        x_norm = np.linspace(port_ret_arr.min(), port_ret_arr.max(), 300)
        y_norm = norm.pdf(x_norm, np.mean(port_ret_arr), np.std(port_ret_arr))
        y_norm = y_norm * len(port_ret_arr) * (port_ret_arr.max() - port_ret_arr.min()) / 60
        fig3.add_trace(go.Scatter(
            x=x_norm*100, y=y_norm,
            line=dict(color="#b5873a", width=2),
            name="Normal Fit",
        ))
        # VaR line
        fig3.add_vline(x=-var_h*100, line_color="#c0392b", line_dash="dash", line_width=1.5,
                       annotation_text=f"VaR {confidence:.0%}", annotation_font_color="#c0392b",
                       annotation_font_size=10)
        fig3.add_vline(x=-cvar_h*100, line_color="#e07b39", line_dash="dot", line_width=1.5,
                       annotation_text="CVaR", annotation_font_color="#e07b39",
                       annotation_font_size=10, annotation_position="bottom right")
        fig3.update_layout(**{**PLOT_LAYOUT,
            "height":340,"title":dict(text="Return Distribution (Max Sharpe Portfolio)",
            font=dict(size=12,color="#1a1a18")),
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
            fillcolor="rgba(192,57,43,0.15)",
            line=dict(color="#c0392b", width=1.5),
            name="Drawdown",
        ))
        fig4.update_layout(**{**PLOT_LAYOUT,
            "height":340,"title":dict(text="Drawdown Profile",font=dict(size=12,color="#1a1a18")),
            "yaxis_title":"Drawdown (%)",
        })
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = returns[valid_tickers].corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.index,
        colorscale=[[0,"#b5873a"],[0.5,"#ffffff"],[1,"#2d6a4f"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family="IBM Plex Mono"),
        showscale=True,
    ))
    fig5.update_layout(**{**PLOT_LAYOUT,
        "height":380,"title":dict(text="Asset Return Correlations",font=dict(size=12,color="#1a1a18")),
    })
    st.plotly_chart(fig5, use_container_width=True)

    # Asset-level risk table
    st.markdown('<div class="section-header">Asset Risk Decomposition</div>', unsafe_allow_html=True)
    marginal_contrib = (cov / 252 @ w_primary) / np.sqrt(w_primary @ (cov / 252) @ w_primary)
    risk_contrib     = w_primary * marginal_contrib
    risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

    df_risk = pd.DataFrame({
        "Asset":           valid_tickers,
        "Weight (%)":      (w_primary * 100).round(2),
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

    cell_colors = [["#ffffff"] * len(df_risk) for _ in range(6)]
    cell_colors[5] = [_rc_color(v) for v in rc_vals]

    fig_tbl = go.Figure(go.Table(
        columnwidth=[60, 80, 100, 80, 60, 110],
        header=dict(
            values=["<b>Asset</b>","<b>Weight %</b>","<b>Ann. Return %</b>",
                    "<b>Ann. Vol %</b>","<b>Sharpe</b>","<b>Risk Contrib %</b>"],
            fill_color="#f0ece4",
            line_color="#c8bfb2",
            font=dict(family="IBM Plex Mono", size=11, color="#2d6a4f"),
            align="center", height=32,
        ),
        cells=dict(
            values=[df_risk[c].tolist() for c in df_risk.columns],
            fill_color=cell_colors,
            line_color="#e0d9ce",
            font=dict(family="IBM Plex Mono", size=11, color="#1a1a18"),
            align="center", height=28,
        )
    ))
    fig_tbl.update_layout(**{**PLOT_LAYOUT, "height": 60 + 28 * len(df_risk) + 32, "margin": dict(l=0,r=0,t=0,b=0)})
    st.plotly_chart(fig_tbl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROLLING METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-header">Rolling Risk & Factor Exposure · {_preset_lbl} Portfolio · 60-Day Window</div>', unsafe_allow_html=True)

    port_ret_full = port_series(w_primary, returns)
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
        name=f"{_preset_lbl} Portfolio", line=dict(color=_preset_col, width=2)), row=1, col=1)
    fig6.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values,
        name="SPY", line=dict(color="#8a8072", width=1.5, dash="dash")), row=1, col=1)

    fig6.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta["beta"],
        name="Beta", line=dict(color="#2e86ab", width=1.8),
        fill="tozeroy", fillcolor="rgba(74,124,158,0.1)"), row=2, col=1)
    fig6.add_hline(y=1.0, line_color="#8a8072", line_dash="dot", line_width=1, row=2, col=1)

    fig6.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
        name="Sharpe", line=dict(color="#7b2d8b", width=1.8),
        fill="tozeroy", fillcolor="rgba(107,63,160,0.08)"), row=3, col=1)
    fig6.add_hline(y=0, line_color="#8a8072", line_dash="dot", line_width=1, row=3, col=1)

    fig6.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
        name="Volatility (%)", line=dict(color="#c0392b", width=1.8),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.08)"), row=4, col=1)

    fig6.update_layout(**{**PLOT_LAYOUT,
        "height": 900,
        "showlegend": True,
        "title": dict(text=f"Rolling Factor Analytics · {_preset_lbl} Portfolio vs SPY",
                      font=dict(size=13, color="#1a1a18")),
    })
    for i in range(1, 5):
        fig6.update_xaxes(gridcolor="#e0d9ce", row=i, col=1)
        fig6.update_yaxes(gridcolor="#e0d9ce", row=i, col=1)

    st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-header">Portfolio Report · {_preset_lbl} · {_preset_sub}</div>', unsafe_allow_html=True)

    port_ret_series = port_series(w_primary, returns)
    cum_port_full   = (1 + port_ret_series).cumprod()
    max_dd_val      = compute_max_drawdown(cum_port_full)
    var_val, cvar_val = compute_var_cvar(port_ret_series.values, confidence)
    sort_val        = compute_sortino(port_ret_series.values, rf_daily)
    calmar_val      = compute_calmar(r_ut, max_dd_val)

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
<div class="metric-card" style="margin-bottom:1.5rem;border-left:3px solid {_preset_col};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
    <div class="metric-label" style="font-size:0.7rem;">
      QUANTFRAME ANALYTICS REPORT &nbsp;·&nbsp;
      Universe: {', '.join(valid_tickers)} &nbsp;·&nbsp;
      Period: {period_label} &nbsp;·&nbsp;
      RF: {rf*100:.2f}% &nbsp;·&nbsp;
      Confidence: {confidence:.0%}
    </div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;font-weight:600;
                 color:{_preset_col};white-space:nowrap;padding-left:1rem;">
      {_preset_lbl} · {_preset_sub}
    </span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
    <div>
      <div class="metric-label">Annualized Return</div>
      <div class="metric-value {'positive' if r_ut>0 else 'negative'}">{r_ut*100:.2f}%</div>
    </div>
    <div>
      <div class="metric-label">Annualized Volatility</div>
      <div class="metric-value">{v_ut*100:.2f}%</div>
    </div>
    <div>
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-value {'positive' if s_ut>0 else 'negative'}">{s_ut:.4f}</div>
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
        "Ticker":              valid_tickers,
        f"{_preset_lbl} (%)": (w_primary * 100).round(2),
        "Optimal Risky (%)":  (w_sharpe  * 100).round(2),
        "Min Variance (%)":   (w_minvol  * 100).round(2),
        "Equal Weight (%)":   (w_eq      * 100).round(2),
    }).sort_values(f"{_preset_lbl} (%)", ascending=False)

    # Color active holdings
    def _wt_color(v):
        if v > 0.01:
            t = min(v / 40, 1.0)
            return f"rgba(0,{int(180*t+40)},{int(120*t+50)},0.25)"
        return "#f0ece4"

    user_col = f"{_preset_lbl} (%)"
    alloc_colors = [
        ["#ffffff"] * len(df_alloc),
        [_wt_color(v) for v in df_alloc[user_col]],
        [_wt_color(v) for v in df_alloc["Optimal Risky (%)"]],
        [_wt_color(v) for v in df_alloc["Min Variance (%)"]],
        ["#ffffff"] * len(df_alloc),
    ]
    fig_alloc = go.Figure(go.Table(
        columnwidth=[60, 100, 100, 90, 100],
        header=dict(
            values=[
                "<b>Ticker</b>",
                f"<b>{_preset_lbl}</b>",
                "<b>Optimal Risky</b>",
                "<b>Min Variance</b>",
                "<b>Equal Weight</b>",
            ],
            fill_color="#f0ece4",
            line_color="#c8bfb2",
            font=dict(family="IBM Plex Mono", size=11, color=[
                "#8a8072", _preset_col, "#7b2d8b", "#2e86ab", "#e07b39"
            ]),
            align="center", height=32,
        ),
        cells=dict(
            values=[df_alloc[c].tolist() for c in df_alloc.columns],
            fill_color=alloc_colors,
            line_color="#e0d9ce",
            font=dict(family="IBM Plex Mono", size=11, color="#1a1a18"),
            align="center", height=28,
        )
    ))
    fig_alloc.update_layout(**{**PLOT_LAYOUT, "height": 60 + 28 * len(df_alloc) + 32, "margin": dict(l=0,r=0,t=0,b=0)})
    st.plotly_chart(fig_alloc, use_container_width=True)

    # Methodology note
    st.markdown('<div class="section-header">Methodology Notes</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;line-height:1.8;color:#8a8072;
            background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.25rem;">

<span style="color:#2d6a4f;">OPTIMIZATION</span>  
Mean-variance optimization via SLSQP (Markowitz, 1952). Expected returns estimated from historical 
sample mean; covariance from historical sample covariance. Annualized assuming 252 trading days.
Box constraints enforce 0 ≤ wᵢ ≤ max_weight with full-investment constraint Σwᵢ = 1.
Cardinality constraint (max holdings N) applied post-optimization: the N largest weights are 
retained and re-normalized. This is mathematically equivalent to a thresholded tangency portfolio 
and is standard practice for small universes.

<br><br>
<span style="color:#b5873a;">RISK METRICS</span>  
VaR and CVaR computed via historical simulation — no distributional assumption imposed.  
Sortino uses downside deviation (returns below Rf) as denominator.  
Beta estimated via 60-day rolling OLS covariance ratio against SPY.
Omega ratio = E[gains above Rf] / E[losses below Rf].

<br><br>
<span style="color:#e07b39;">DATA</span>  
Source: Yahoo Finance via yfinance. Adjusted close prices (splits + dividends).  
Lookback options: 1Y through Max (full history). Cached 1 hour per session.  
Benchmark: SPY (SPDR S&P 500 ETF Trust).

<br><br>
<span style="color:#c0392b;">LIMITATIONS</span>  
Sample covariance is a noisy estimator — longer lookbacks reduce variance but may include 
structural breaks. Ledoit-Wolf shrinkage not applied (future work).  
Cardinality constraint via thresholding is a heuristic; true cardinality-constrained MVO is NP-hard.  
Historical returns are not indicative of future performance. Transaction costs and taxes not modeled.

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("QuantFrame v2 · Built with Streamlit + yfinance + SciPy · Created by Fulton Pace")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("ℹ  About QuantFrame v2", key="btn_about_footer"):
        st.session_state.show_about = True
        st.rerun()
