# ⬡ QuantFrame v2 — Portfolio Intelligence Dashboard

https://quantframe.streamlit.app/

> **Mean-Variance Optimization · Historical VaR/CVaR · Rolling Factor Exposure · Risk Decomposition · S&P 500 Discovery**  
> Built with Python · Streamlit · yfinance · SciPy · Plotly

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quantframe.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

---

## Overview

QuantFrame is a quantitative portfolio analytics tool that implements **Modern Portfolio Theory** and multi-dimensional **risk modeling** in an interactive dashboard. It operates in two modes: **Portfolio Lab**, where users input a stock universe, set optimization constraints, and receive a fully decomposed portfolio with Sharpe-maximizing weights, tail-risk metrics, and rolling factor exposures; and **Discovery Mode**, which randomly searches thousands of stock combinations from the S&P 500 to surface the best-performing portfolio. All analysis is rendered in a terminal-aesthetic UI.

This project was built as a decision-support tool for portfolio construction and risk assessment, grounded in peer-reviewed financial theory.

---

## Modes

### ⬡ Portfolio Lab
Input a custom or preset stock universe, configure constraints, and run Markowitz optimization. Outputs a full portfolio breakdown across four analysis tabs.

### 🔍 Discovery Mode
Randomly samples stock combinations from the S&P 500 (~490 tickers, filterable by sector) and runs Max Sharpe optimization on each. Returns the combination with the highest Sharpe ratio across all iterations, with a live progress tracker and Sharpe history chart.

---

## Features

| Module | Models & Metrics |
|--------|-----------------|
| **Efficient Frontier** | Markowitz MVO (SLSQP), Capital Market Line, Max Sharpe, Min Volatility, Equal Weight, Utility-Optimal |
| **Risk Analytics** | Historical VaR, CVaR (Expected Shortfall), Max Drawdown, Calmar, Sortino, Omega, Skewness, Excess Kurtosis |
| **Rolling Metrics** | 60-day rolling Beta vs SPY, rolling Sharpe, rolling Volatility, cumulative returns |
| **Portfolio Report** | Full statistics table, allocation breakdown (4-portfolio comparison), risk contribution decomposition, methodology notes |
| **Discovery Mode** | Random S&P 500 universe search, sector filters, live Sharpe history chart, configurable iterations (10–5,000) |
| **About Modal** | Expandable in-app reference explaining MPT, Factor Risk, and Decision Analytics in plain language with formulas |

### Sidebar Controls

| Control | Options | Notes |
|---------|---------|-------|
| Mode | Analyze / Discover | Switches between Portfolio Lab and Discovery |
| Asset Universe | 4 presets + Custom | Mega-Cap Tech, Blue-Chip, Factor Tilt, Custom |
| Lookback Period | 1Y / 2Y / 3Y / 5Y / Max | Historical window for return estimation |
| VaR Confidence | 90% / 95% / 99% | Quantile for VaR and CVaR |
| Max Asset Weight | 10%–100% | Box constraint per asset (or optimizer-controlled) |
| Min Asset Weight | 0%–20% | Floor constraint; auto-caps N to prevent infeasibility |
| Max Holdings (N) | 2–20 | Cardinality constraint; or optimizer-derived Effective N |
| Risk-Free Rate | User-specified | Annualized; default 5.25% |
| Risk Profile | 6 presets + Custom λ | No Guts → High Roller, maps to Arrow-Pratt utility |

---

## Theoretical Background

### Mean-Variance Optimization (Markowitz, 1952)

The optimizer solves:

$$\max_{w} \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}$$

subject to:

$$\sum_i w_i = 1, \quad w_{\min} \leq w_i \leq w_{\max}$$

where $\mu$ is the vector of expected returns (sample mean, annualized), $\Sigma$ is the annualized covariance matrix (sample covariance × 252), and $r_f$ is the user-specified risk-free rate.

Four portfolios are computed: **Max Sharpe** (tangency portfolio), **Min Volatility** (global minimum variance), **Equal Weight** (1/N baseline), and a **Utility-Optimal** portfolio selected by the user's risk tolerance $\lambda$. SLSQP is used via `scipy.optimize.minimize`.

### Utility-Based Portfolio Selection (Arrow-Pratt)

The risk tolerance slider maps directly to the Arrow-Pratt mean-variance utility function:

$$U(w) = \mu_p - \frac{\lambda}{2} \sigma_p^2$$

Higher $\lambda$ selects a more conservative point on the efficient frontier; lower $\lambda$ selects more aggressive. Six named presets cover the range: Min Variance ($\lambda = \infty$), Steady ($\lambda = 10$), Average ($\lambda = 4$), High Roller ($\lambda = 1.5$), Optimal Risky (tangency, $\lambda$-independent), and Custom.

### Cardinality Constraint

A maximum-holdings constraint (N) is enforced post-optimization: the N largest weights are retained and renormalized. This is equivalent to a thresholded tangency portfolio and is standard practice for small universes. A minimum weight floor is also enforced, with N automatically capped to prevent the infeasibility condition $w_{\min} \times N > 1$.

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

### Discovery Mode

Randomly samples $k$-stock combinations from the selected universe and runs Max Sharpe optimization on each. Returns the combination with the highest realized Sharpe ratio. Search space is combinatorially vast (~$10^{12}$ for large universes); results are a statistical sample, not a global optimum. Configurable from 10 to 5,000 iterations.

---

## Data

- **Source:** Yahoo Finance via `yfinance` (adjusted close prices — corrected for splits and dividends)
- **Frequency:** Daily · 252 trading days per year
- **Universe:** User-defined (presets: Mega-Cap Tech, Diversified Blue-Chip, Factor Tilt, Custom) or S&P 500 (~490 tickers, Discovery Mode)
- **Lookback:** 1Y to Max history (user-selectable)
- **Benchmark:** SPY (SPDR S&P 500 ETF)
- **Caching:** 1-hour TTL via `@st.cache_data` to minimize redundant API calls

---

## Installation & Local Development

```bash
# Clone the repo
git clone https://github.com/fultonpace/quantframe.git
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
├── app.py              # Main Streamlit application (~2,370 lines)
├── logo.svg            # App icon
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
| Max Weight | Optimizer-controlled | Box constraint per asset (can also be set manually) |
| Min Weight | 0% | Floor per asset; 0 = unconstrained |
| Max Holdings (N) | Optimizer-derived | Cardinality constraint (Effective N = 1/Σwᵢ²) |
| Risk-Free Rate | 5.25% | Annualized, used in all ratio calculations |
| Risk Profile | Average (λ=4) | Arrow-Pratt utility preset |

---

## Limitations

1. **Estimation risk** — Sample mean and covariance are noisy estimators, especially for short lookback windows. Shrinkage estimators (Ledoit-Wolf) or Black-Litterman views would improve robustness.

2. **Return stationarity** — The model assumes returns are drawn from a stationary distribution. Structural breaks, regime changes, and fat tails are not explicitly modeled.

3. **No transaction costs** — Optimal weights are theoretical. Turnover, bid-ask spreads, and market impact are ignored.

4. **Single-period model** — MVO is a single-period framework. Dynamic rebalancing, multi-period utility, or continuous-time formulations (e.g., Merton, 1969) are not implemented.

5. **Cardinality constraint is heuristic** — True cardinality-constrained MVO is NP-hard. The thresholding + renormalization approach is an approximation.

6. **Discovery is sampling** — 5,000 iterations covers a tiny fraction of all possible portfolios. Results vary across runs and are not globally optimal.

7. **Historical ≠ Future** — All metrics are backward-looking. Past Sharpe ratios, betas, and drawdowns are not guarantees of future performance.

---

## Known Todos

- [ ] Ledoit-Wolf shrinkage covariance (would reduce estimation noise)
- [ ] Confidence interval on best Sharpe found in Discovery Mode
- [ ] Sector legend for "All S&P 500" option in Discovery (currently shows only named-sector tickers; full list is pulled live from GitHub CSV at runtime)

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Sharpe, W. F. (1966). Mutual Fund Performance. *Journal of Business*, 39(1), 119–138.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.
- Sortino, F. A., & van der Meer, R. (1991). Downside Risk. *Journal of Portfolio Management*, 17(4), 27–31.
- Merton, R. C. (1969). Lifetime Portfolio Selection under Uncertainty. *Review of Economics and Statistics*, 51(3), 247–257.
- Arrow, K. J. (1965). *Aspects of the Theory of Risk-Bearing*. Yrjö Jahnsson Foundation.

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
