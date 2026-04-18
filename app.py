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
    --bg:        #f7f5f0;
    --surface:   #ffffff;
    --border:    #e0d9ce;
    --accent:    #2d6a4f;
    --accent2:   #b5873a;
    --accent3:   #c9a84c;
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
/* ── BUTTONS — premium press aesthetic ────────────────────────────────── */
.stButton > button {
    background: transparent;
    color: var(--accent);
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid var(--accent);
    border-radius: 3px;
    padding: 0.55rem 1.25rem;
    width: 100%;
    cursor: pointer;
    position: relative;
    transition: background 0.15s ease, color 0.15s ease,
                transform 0.1s ease, box-shadow 0.1s ease,
                border-color 0.15s ease;
    box-shadow: 0 3px 0 rgba(45,106,79,0.35), 0 1px 6px rgba(0,0,0,0.4);
}
.stButton > button:hover {
    background: rgba(45,106,79,0.08);
    border-color: #1e8f62;
    color: #1e8f62;
    box-shadow: 0 3px 0 rgba(30,143,98,0.4), 0 2px 12px rgba(45,106,79,0.15);
    transform: translateY(-1px);
}
.stButton > button:active,
.stButton > button:focus:not(:focus-visible) {
    background: rgba(45,106,79,0.18);
    color: #2d6a4f;
    border-color: var(--accent);
    transform: translateY(2px);
    box-shadow: 0 0px 0 rgba(45,106,79,0.2), 0 1px 4px rgba(0,0,0,0.3);
}
/* Run Optimization — filled primary */
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
    st.markdown("## ⬡ QuantFrame")
    st.markdown('<p class="app-subtitle">Portfolio Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## Universe")
    preset = st.selectbox("Asset Universe", list(PRESET_UNIVERSES.keys()), index=0, label_visibility="collapsed")

    if preset == "Custom":
        custom_raw = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN")
        tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
        n_typed = len(tickers)
        if n_typed > 20:
            hint_color  = "#b5873a"
            hint_icon   = "⚠"
            hint_text   = f"{n_typed} tickers · Large universes may slow optimization"
        else:
            hint_color  = "#8a8072"
            hint_icon   = "·"
            hint_text   = f"{n_typed} ticker{'s' if n_typed != 1 else ''} · Recommended: 5–20 · No hard limit"
        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            color:{hint_color};margin-top:0.2rem;letter-spacing:0.04em;">
  {hint_icon} {hint_text}
</div>""", unsafe_allow_html=True)
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
    rf_input   = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, RF_RATE * 100, 0.25, format="%.2f")
    rf = rf_input / 100

    # ── Max Weight Constraint ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Weight Constraint")

    if "optimize_weights" not in st.session_state:
        st.session_state.optimize_weights = False

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        if st.button("OPTIMIZE", key="btn_wt_opt"):
            st.session_state.optimize_weights = True
    with col_w2:
        if st.button("MANUAL", key="btn_wt_manual"):
            st.session_state.optimize_weights = False

    wt_opt = st.session_state.optimize_weights
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.4rem;margin-bottom:0.5rem;">
  <div style="text-align:center;">
    <div style="width:5px;height:5px;border-radius:50%;margin:0 auto;
                background:{'#2d6a4f' if wt_opt else 'transparent'};
                box-shadow:{'0 0 6px #2d6a4f' if wt_opt else 'none'};
                transition:all 0.2s;"></div>
  </div>
  <div style="text-align:center;">
    <div style="width:5px;height:5px;border-radius:50%;margin:0 auto;
                background:{'transparent' if wt_opt else '#2d6a4f'};
                box-shadow:{'none' if wt_opt else '0 0 6px #2d6a4f'};
                transition:all 0.2s;"></div>
  </div>
</div>""", unsafe_allow_html=True)

    wt_opt = st.session_state.optimize_weights
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-bottom:0.25rem;
            background:{'#f0ece4' if wt_opt else 'transparent'};
            border:1px solid {'#d6cfc4' if wt_opt else 'transparent'};
            color:{'#8a8072' if wt_opt else 'transparent'};">
  {'◆ Unconstrained — optimizer controls allocation' if wt_opt else '◆'}
</div>""", unsafe_allow_html=True)
    slider_wt = st.slider("Max Single Asset Weight", 0.10, 1.0,
                          value=1.0 if wt_opt else 0.40,
                          step=0.05, format="%.2f", disabled=wt_opt)
    max_weight = 1.0 if wt_opt else slider_wt

    # ── Diversification ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Diversification")
    st.caption("Max assets with nonzero weight in optimal portfolio")

    if "optimize_n" not in st.session_state:
        st.session_state.optimize_n       = False
        st.session_state.suggested_n      = 8
        st.session_state.max_assets_val   = 8

    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        if st.button("OPTIMIZE", key="btn_optimize_n"):
            st.session_state.optimize_n = True
    with col_n2:
        if st.button("MANUAL", key="btn_manual_n"):
            st.session_state.optimize_n = False

    n_opt = st.session_state.optimize_n
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.4rem;margin-bottom:0.5rem;">
  <div style="text-align:center;">
    <div style="width:5px;height:5px;border-radius:50%;margin:0 auto;
                background:{'#2d6a4f' if n_opt else 'transparent'};
                box-shadow:{'0 0 6px #2d6a4f' if n_opt else 'none'};
                transition:all 0.2s;"></div>
  </div>
  <div style="text-align:center;">
    <div style="width:5px;height:5px;border-radius:50%;margin:0 auto;
                background:{'transparent' if n_opt else '#2d6a4f'};
                box-shadow:{'none' if n_opt else '0 0 6px #2d6a4f'};
                transition:all 0.2s;"></div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#8a8072;
            line-height:1.55;margin:0.4rem 0 0.6rem 0;padding:0.5rem 0.65rem;
            background:#f7f5f0;border:1px solid #d6cfc4;border-radius:3px;">
  <b style="color:#1a1a18;">OPTIMIZE N</b> sets holdings to the portfolio's
  <b style="color:#2d6a4f;">effective N</b> = 1/Σwᵢ² — the number of
  assets the optimizer naturally concentrates into.
</div>""", unsafe_allow_html=True)

    n_opt = st.session_state.optimize_n
    sn    = st.session_state.suggested_n

    # Status line — always same height, content changes color not presence
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            padding:0.3rem 0.65rem;background:#f7f5f0;border:1px solid #d6cfc4;
            border-radius:3px;margin-bottom:0.25rem;min-height:1.55rem;">
  <span style="color:{'#8a8072' if n_opt else 'transparent'};">Effective N = </span>
  <span style="color:{'#2d6a4f' if n_opt else 'transparent'};font-weight:600;">{sn}</span>
  <span style="color:{'#8a8072' if n_opt else 'transparent'};"> (auto)</span>
</div>""", unsafe_allow_html=True)

    # Slider — always rendered, disabled when optimize is active
    slider_n = st.slider("Max Holdings (N)", min_value=2, max_value=20,
                         value=sn if n_opt else st.session_state.max_assets_val,
                         step=1, disabled=n_opt)
    if n_opt:
        max_assets = sn
    else:
        max_assets = slider_n
        st.session_state.max_assets_val = slider_n

    # ── Risk Tolerance ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Risk Tolerance")
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#8a8072;
            line-height:1.6;margin-bottom:0.75rem;">
Selects a portfolio on the efficient frontier via a <b style="color:#1a1a18;">utility function</b>:<br>
<span style="color:#2d6a4f;">max U = μ − (λ/2)σ²</span><br>
where λ is your risk aversion coefficient.<br>
Higher λ → closer to Min Variance.<br>
Lower λ → closer to Max Sharpe (Optimal Risky).
</div>
""", unsafe_allow_html=True)

    # 5 presets in ascending risk order + Variable option
    RISK_PRESETS = [
        ("NO GUTS — Min Variance Portfolio",      "minvar",   None,  "#4a7c9e"),
        ("STEADY — Low Risk  (λ=10)",             "steady",   10.0,  "#2d6a4f"),
        ("AVERAGE — Moderate  (λ=4)",             "average",  4.0,   "#b5873a"),
        ("HIGH ROLLER — Aggressive  (λ=1.5)",     "roller",   1.5,   "#c0392b"),
        ("OPTIMAL RISKY — Tangency / Max Sharpe", "optimal",  None,  "#6b3fa0"),
        ("VARIABLE — Custom λ",                   "custom",   4.0,   "#5a7a6a"),
    ]
    PRESET_LABELS = [p[0] for p in RISK_PRESETS]
    key_to_idx    = {p[1]: i for i, p in enumerate(RISK_PRESETS)}

    if "risk_preset" not in st.session_state:
        st.session_state.risk_preset = "average"
    if "risk_lambda_val" not in st.session_state:
        st.session_state.risk_lambda_val = 4.0

    default_idx = key_to_idx.get(st.session_state.risk_preset, 2)

    selected_label = st.selectbox(
        "Risk Profile",
        PRESET_LABELS,
        index=default_idx,
        label_visibility="collapsed",
        key="risk_dropdown",
    )
    active_preset = RISK_PRESETS[PRESET_LABELS.index(selected_label)]

    # When dropdown changes to a non-Variable preset, snap slider to that λ
    if active_preset[1] != "custom" and active_preset[2] is not None:
        st.session_state.risk_lambda_val = active_preset[2]
    elif active_preset[1] == "minvar" or active_preset[1] == "optimal":
        pass  # no lambda to snap to

    st.session_state.risk_preset = active_preset[1]
    risk_color = active_preset[3]

    # Active preset pill
    short_name = active_preset[0].split("—")[0].strip()
    sub_name   = active_preset[0].split("—")[1].strip() if "—" in active_preset[0] else ""
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
            padding:0.45rem 0.75rem;margin-top:0.25rem;margin-bottom:0.5rem;
            background:#f0ece4;border-left:2px solid {risk_color};border-radius:0 3px 3px 0;">
  <span style="color:{risk_color};font-weight:600;">{short_name}</span>
  <span style="color:#8a8072;"> · {sub_name}</span>
</div>""", unsafe_allow_html=True)

    # Lambda slider — always visible
    # Disabled (grayed) for minvar and optimal since λ is irrelevant there
    slider_disabled = active_preset[1] in ("minvar", "optimal")

    slider_val = st.slider(
        "λ (risk aversion coefficient)",
        0.5, 15.0,
        value=st.session_state.risk_lambda_val,
        step=0.5,
        format="%.1f",
        disabled=slider_disabled,
        help="λ=0.5 → maximum risk. λ=15 → near min variance. Moving this sets dropdown to VARIABLE.",
    )

    # If slider moved away from preset's λ → snap dropdown to VARIABLE
    if not slider_disabled and active_preset[1] != "custom":
        expected = active_preset[2]
        if expected is not None and abs(slider_val - expected) > 0.01:
            st.session_state.risk_preset      = "custom"
            st.session_state.risk_lambda_val  = slider_val
            st.rerun()

    if slider_disabled:
        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-top:0.25rem;
            background:#f0ece4;border:1px solid #d6cfc4;color:#8a8072;">
  ◆ λ not applicable for this portfolio
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
            padding:0.3rem 0.6rem;min-height:1.55rem;border-radius:3px;margin-top:0.25rem;
            background:#f0ece4;border:1px solid #d6cfc4;color:#8a8072;">
  Higher λ → less risk, closer to Min Variance &nbsp;·&nbsp; Lower λ → more risk, closer to Optimal Risky
</div>""", unsafe_allow_html=True)

    # Store slider value
    st.session_state.risk_lambda_val = slider_val

    # Resolve final lambda + source
    if active_preset[1] == "minvar":
        effective_lambda = None
        lambda_source    = "minvar"
    elif active_preset[1] == "optimal":
        effective_lambda = None
        lambda_source    = "optimal"
    else:
        effective_lambda = slider_val
        lambda_source    = active_preset[1]

    st.markdown("---")
    run_btn = st.button("▶  Run Optimization")

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

# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<div class="app-title">Quant<span>Frame</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Modern Portfolio Theory · Factor Risk · Decision Analytics</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
            color:#b0a898;margin-top:0.35rem;letter-spacing:0.04em;">
  Created by <span style="color:#2d6a4f;font-weight:500;">Fulton Pace</span>
  <span style="color:#d6cfc4;margin:0 0.5rem;">·</span>
  Market data via <span style="color:#b5873a;font-weight:500;">Yahoo Finance</span>
</div>
""", unsafe_allow_html=True)
with col_badge:
    st.markdown('<br><span class="badge">v1.0</span>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Top-level mode switcher ───────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
}
div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    border: 1px solid #e0d9ce !important;
    border-right: none !important;
    cursor: pointer !important;
    color: #8a8072 !important;
    background: #ffffff !important;
    transition: all 0.15s !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] label:first-of-type { border-radius: 3px 0 0 3px !important; }
div[data-testid="stRadio"] label:last-of-type  { border-right: 1px solid #e0d9ce !important; border-radius: 0 3px 3px 0 !important; }
div[data-testid="stRadio"] label[data-selected="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background: #2d6a4f !important;
    color: #ffffff !important;
    border-color: #2d6a4f !important;
}
div[data-testid="stRadio"] input { display: none !important; }
div[data-testid="stRadio"] > label { display: none !important; }
</style>
""", unsafe_allow_html=True)

app_mode = st.radio("Mode", ["  ⬡  Portfolio Lab  ", "  🔍  Discovery Mode  "],
                    horizontal=True, label_visibility="collapsed")

st.markdown('<hr class="divider" style="margin-top:0.75rem;">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY MODE
# ══════════════════════════════════════════════════════════════════════════════
if "Discovery" in app_mode:

    # ── Red warning banner ────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#fdf0ef;border:1.5px solid #c0392b;border-radius:4px;
            padding:0.9rem 1.25rem;margin-bottom:1.5rem;
            display:flex;align-items:flex-start;gap:0.75rem;">
  <span style="font-size:1.2rem;line-height:1;">⚠</span>
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;font-weight:600;
                color:#c0392b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.3rem;">
      Long Runtime Warning
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#8a3030;line-height:1.6;">
      Discovery Mode randomly samples stock combinations from the full S&P 500 universe (~490 tickers)
      and optimizes each one. Each iteration requires a live data fetch + optimization.
      Large sample sizes can take <b>10–30+ minutes</b> to complete.
      Estimated runtime is shown below and updates as you adjust settings.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Discovery controls ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Discovery Settings</div>', unsafe_allow_html=True)

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

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        disc_sector = st.selectbox("Sector Filter", list(SECTORS.keys()), index=0)
    with col_d2:
        disc_port_size = st.slider("Portfolio Size (N stocks)", 5, 20, 10, 1)
    with col_d3:
        disc_iterations = st.slider("Iterations", 10, 500, 50, 10)
    with col_d4:
        disc_start = st.selectbox("Lookback Start", ["2018-01-01","2019-01-01","2020-01-01","2021-01-01"], index=0)

    # ── Live time estimate ────────────────────────────────────────────────────
    SECS_PER_ITER = 2.5
    est_secs  = disc_iterations * SECS_PER_ITER
    est_mins  = est_secs / 60
    if est_secs < 60:
        est_str = f"~{int(est_secs)} seconds"
        est_col = "#2d6a4f"
    elif est_mins < 5:
        est_str = f"~{est_mins:.1f} minutes"
        est_col = "#b5873a"
    elif est_mins < 15:
        est_str = f"~{est_mins:.0f} minutes"
        est_col = "#c0392b"
    else:
        est_str = f"~{est_mins:.0f} minutes — go get coffee ☕"
        est_col = "#c0392b"

    universe_size = len(SECTORS[disc_sector]) if SECTORS[disc_sector] else 490
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0;">
  <div class="metric-card">
    <div class="metric-label">Estimated Runtime</div>
    <div class="metric-value" style="font-size:1.2rem;color:{est_col};">{est_str}</div>
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
    <div class="metric-value" style="font-size:1.2rem;color:#8a8072;">≈10¹²</div>
    <div class="metric-sub">combinatorially vast</div>
  </div>
</div>
""", unsafe_allow_html=True)

    run_discovery = st.button("🔍  Run Discovery", key="btn_discovery")

    if not run_discovery:
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;font-family:'IBM Plex Mono',monospace;">
  <div style="font-size:2rem;color:#e0d9ce;margin-bottom:1rem;">🔍</div>
  <div style="font-size:0.85rem;color:#8a8072;letter-spacing:0.1em;text-transform:uppercase;">
    Configure settings above and press <span style="color:#2d6a4f;">Run Discovery</span>
  </div>
</div>""", unsafe_allow_html=True)
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
                mode="markers", marker=dict(size=4, color="#c9a84c", opacity=0.5),
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
    st.markdown("""
<div style="text-align:center; padding: 5rem 2rem; font-family: 'IBM Plex Mono', monospace;">
    <div style="font-size:3rem; margin-bottom:1rem; color:#e0d9ce;">⬡</div>
    <div style="font-size:0.9rem; color:#8a8072; letter-spacing:0.15em; text-transform:uppercase;">
        Configure your universe in the sidebar<br>and press <span style="color:#2d6a4f;">▶ Run Optimization</span>
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
    n_keep      = min(max_assets, n)

    def _apply_cardinality(w_full, k):
        top = np.argsort(w_full)[-k:]
        w   = np.zeros(len(w_full))
        w[top] = w_full[top]
        return w / w.sum()

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
    frontier_vols, frontier_rets, _ = compute_efficient_frontier(mu / 252, cov / 252)

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

    # Individual assets scatter
    fig.add_trace(go.Scatter(
        x=asset_ann_vol * 100, y=asset_ann_ret * 100,
        mode="markers+text",
        text=valid_tickers,
        textposition="top center",
        textfont=dict(size=10, color="#8a8072", family="IBM Plex Mono"),
        marker=dict(size=9, color="#e0d9ce", line=dict(color="#8a8072", width=1.5)),
        name="Individual Assets",
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # Frontier
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100, y=frontier_rets * 100,
        mode="lines",
        line=dict(color="rgba(45,106,79,0.4)", width=2),
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

    # Portfolio markers — always show Min Var + Optimal Risky, then highlight user's selection
    port_points = [
        (v_sh, r_sh, "Optimal Risky (Max Sharpe)", "#6b3fa0", "star",    18),
        (v_mv, r_mv, "Min Variance Portfolio",      "#4a7c9e", "diamond", 16),
        (v_eq, r_eq, "Equal Weight",                "#c9a84c", "circle",  12),
    ]
    for v_, r_, name_, color_, sym_, sz_ in port_points:
        fig.add_trace(go.Scatter(
            x=[v_ * 100], y=[r_ * 100],
            mode="markers+text",
            text=[name_],
            textposition="top right",
            textfont=dict(size=9, color=color_, family="IBM Plex Mono"),
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
            mode="markers+text",
            text=[f"YOUR PORTFOLIO ({_preset_lbl})"],
            textposition="bottom right",
            textfont=dict(size=9, color=_preset_col, family="IBM Plex Mono"),
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

    # Build card list — insert user portfolio if it's not one of the named ones
    card_list = [
        ("Min Variance",        w_minvol,  r_mv, v_mv, s_mv, "#4a7c9e"),
        ("Equal Weight",        w_eq,      r_eq, v_eq, s_eq, "#c9a84c"),
        ("Optimal Risky",       w_sharpe,  r_sh, v_sh, s_sh, "#6b3fa0"),
    ]
    if is_custom:
        card_list.insert(2, (f"Your Portfolio\n({_preset_lbl})", w_primary, r_ut, v_ut, s_ut, _preset_col))

    cols = st.columns(len(card_list))
    for col, (label, weights, ret, vol, sharpe, color) in zip(cols, card_list):
        is_selected = np.allclose(weights, w_primary, atol=1e-4)
        border_style = f"box-shadow:0 0 0 1px {color}; " if is_selected else ""
        with col:
            st.markdown(f"""
<div class="metric-card" style="--accent:{color};{border_style}">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
  <div class="metric-label" style="white-space:pre;">{label}</div>
  {'<span class="badge" style="background:rgba(45,106,79,0.12);color:' + color + ';border-color:' + color + '60;">ACTIVE</span>' if is_selected else ''}
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

    # ── Weights chart
    st.markdown(f'<div class="section-header">Portfolio Weights · Your Selection: {_preset_lbl}</div>', unsafe_allow_html=True)
    df_w = pd.DataFrame({
        "Ticker":        valid_tickers,
        "Your Portfolio": w_primary * 100,
        "Optimal Risky": w_sharpe  * 100,
        "Min Variance":  w_minvol  * 100,
        "Equal Wt":      w_eq      * 100,
    }).sort_values("Your Portfolio", ascending=False)

    fig2 = go.Figure()
    for name, color in [
        ("Your Portfolio", _preset_col),
        ("Optimal Risky",  "#6b3fa0"),
        ("Min Variance",   "#4a7c9e"),
        ("Equal Wt",       "#c9a84c"),
    ]:
        fig2.add_trace(go.Bar(
            name=name, x=df_w["Ticker"], y=df_w[name],
            marker_color=color, marker_line_width=0,
            opacity=0.85,
        ))
    fig2.update_layout(**{**PLOT_LAYOUT,
        "barmode":"group","height":340,
        "yaxis_title":"Weight (%)",
        "title":dict(text="Portfolio Weight Comparison", font=dict(size=12,color="#1a1a18")),
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
        fig3.add_vline(x=-cvar_h*100, line_color="#c9a84c", line_dash="dot", line_width=1.5,
                       annotation_text="CVaR", annotation_font_color="#c9a84c",
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
        name="Beta", line=dict(color="#4a7c9e", width=1.8),
        fill="tozeroy", fillcolor="rgba(74,124,158,0.1)"), row=2, col=1)
    fig6.add_hline(y=1.0, line_color="#8a8072", line_dash="dot", line_width=1, row=2, col=1)

    fig6.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
        name="Sharpe", line=dict(color="#6b3fa0", width=1.8),
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
                "#8a8072", _preset_col, "#6b3fa0", "#4a7c9e", "#c9a84c"
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
<span style="color:#c9a84c;">DATA</span>  
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
    st.caption("QuantFrame · Built with Streamlit + yfinance + SciPy · For educational and research purposes only. Not financial advice.")
