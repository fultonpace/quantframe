# ⬡ QuantFrame — Portfolio Intelligence Dashboard

> **Mean-Variance Optimization · Historical VaR/CVaR · Rolling Factor Exposure · Risk Decomposition**  
> Built with Python · Streamlit · yfinance · SciPy · Plotly

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

---

## Overview

QuantFrame is a quantitative portfolio analytics tool that implements **Modern Portfolio Theory** and multi-dimensional **risk modeling** in an interactive dashboard. Users input a stock universe, set optimization constraints, and receive a fully decomposed portfolio with Sharpe-maximizing weights, tail-risk metrics, and rolling factor exposures — all rendered in a terminal-aesthetic dark UI.

This project was built as a decision-support tool for portfolio construction and risk assessment, grounded in peer-reviewed financial theory.

---

## Features

| Module | Models & Metrics |
|--------|-----------------|
| **Efficient Frontier** | Markowitz MVO (SLSQP), Capital Market Line, Max Sharpe, Min Volatility, Equal Weight |
| **Risk Analytics** | Historical VaR, CVaR (Expected Shortfall), Max Drawdown, Calmar, Sortino, Omega |
| **Rolling Metrics** | 60-day rolling Beta vs SPY, rolling Sharpe, rolling Volatility, cumulative returns |
| **Portfolio Report** | Full statistics table, allocation breakdown, skewness, excess kurtosis, methodology notes |

---

## Theoretical Background

### Mean-Variance Optimization (Markowitz, 1952)

The optimizer solves:

$$\max_{w} \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}$$

subject to:

$$\sum_i w_i = 1, \quad 0 \leq w_i \leq w_{\max}$$

where $\mu$ is the vector of expected returns (sample mean, annualized), $\Sigma$ is the annualized covariance matrix (sample covariance × 252), and $r_f$ is the user-specified risk-free rate.

Three portfolios are computed: **Max Sharpe** (tangency portfolio), **Min Volatility** (global minimum variance), and **Equal Weight** (1/N baseline). SLSQP is used via `scipy.optimize.minimize`.

### Value at Risk & Conditional VaR

**Historical simulation** — no distributional assumption imposed on the return series:

$$\text{VaR}_\alpha = -F^{-1}(1-\alpha)$$

$$\text{CVaR}_\alpha = -\mathbb{E}[R \mid R \leq -\text{VaR}_\alpha]$$

where $F^{-1}$ is the empirical quantile of the portfolio's daily return series.

### Risk Ratios

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| **Sharpe** | $(R_p - r_f) / \sigma_p$ | Return per unit total risk |
| **Sortino** | $(R_p - r_f) / \sigma_d$ | Return per unit downside risk |
| **Calmar** | $R_p / \|\text{MDD}\|$ | Return per unit max drawdown |
| **Omega** | $\int_{r_f}^{\infty} (1-F(r))dr \;/\; \int_{-\infty}^{r_f} F(r)dr$ | Gains-to-losses ratio above threshold |

### Rolling Beta

Beta is estimated via 60-day rolling OLS:

$$\beta_t = \frac{\text{Cov}(R_p, R_m)_t}{\text{Var}(R_m)_t}$$

where $R_m$ is the SPY benchmark daily return. This captures time-varying market exposure and regime shifts.

### Risk Contribution Decomposition

Marginal risk contribution of asset $i$:

$$MRC_i = \frac{(\Sigma w)_i}{\sqrt{w^\top \Sigma w}}$$

Portfolio risk contribution: $RC_i = w_i \cdot MRC_i$, normalized to percentages. Identifies concentrated risk sources that weight-based views miss.

---

## Data

- **Source:** Yahoo Finance via `yfinance` (adjusted close prices)
- **Frequency:** Daily
- **Universe:** User-defined (presets: Mega-Cap Tech, Diversified Blue-Chip, Factor Tilt, Custom)
- **Lookback:** 1–5 years (user-selectable)
- **Benchmark:** SPY (SPDR S&P 500 ETF)
- **Caching:** 1-hour TTL via `@st.cache_data` to minimize redundant API calls

---

## Installation & Local Development

```bash
# Clone the repo
git clone https://github.com/yourusername/quantframe.git
cd quantframe

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
quantframe/
├── app.py              # Main Streamlit application (964 lines)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Configuration

All parameters are controlled from the sidebar at runtime — no config files needed.

| Parameter | Default | Description |
|-----------|---------|-------------|
| Asset Universe | Mega-Cap Tech | 8-ticker preset or custom input |
| Lookback Period | 2 Years | Historical window for return estimation |
| VaR Confidence | 95% | Quantile for VaR and CVaR |
| Max Weight | 40% | Box constraint per asset |
| Risk-Free Rate | 5.25% | Annualized, used in all ratio calculations |

---

## Limitations

1. **Estimation risk** — Sample mean and covariance are noisy estimators, especially for short lookback windows. Shrinkage estimators (Ledoit-Wolf) or Black-Litterman views would improve robustness.

2. **Return stationarity** — The model assumes returns are drawn from a stationary distribution. Structural breaks, regime changes, and fat tails are not explicitly modeled.

3. **No transaction costs** — Optimal weights are theoretical. Turnover, bid-ask spreads, and market impact are ignored.

4. **Single-period model** — MVO is a single-period framework. Dynamic rebalancing, multi-period utility, or continuous-time formulations (e.g., Merton, 1969) are not implemented.

5. **Historical ≠ Future** — All metrics are backward-looking. Past Sharpe ratios, betas, and drawdowns are not guarantees of future performance.

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business*, 39(1), 119–138.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.
- Sortino, F. A., & van der Meer, R. (1991). Downside Risk. *Journal of Portfolio Management*, 17(4), 27–31.
- Merton, R. C. (1969). Lifetime Portfolio Selection under Uncertainty. *Review of Economics and Statistics*, 51(3), 247–257.

---

## Dependencies

```
streamlit>=1.32.0
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
plotly>=5.20.0
```

---

## License

MIT License. For educational and research purposes only. **Not financial advice.**

---

*Built with Python + Streamlit · Data via Yahoo Finance · Optimization via SciPy*
