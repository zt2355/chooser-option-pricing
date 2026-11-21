# Data Requirement Specification

**Project:** Advanced Chooser Option Pricing Model with Real-World Data & Machine Learning
**Week 1 deliverable** · Collection window: **2018-01-01 – 2024-12-31** (daily)

## 1. Purpose

This document defines the data required to (a) replicate the BSM-based chooser
option pricing model on JPM, (b) evaluate it against market prices, and
(c) supply features for the ML volatility/pricing models in Weeks 5–6.

Every series below maps to either a **BSM pricing input** (S₀, σ, r, q, T) or an
**ML feature** addressing BSM's constant-volatility limitation.

## 2. Data Scope

### 2.1 Financial data (underlying asset)

| Series | Source | File | Frequency | Role |
|---|---|---|---|---|
| JPM OHLCV + Adj Close | Yahoo Finance (`yfinance`) | `jpm_daily.csv` | Daily | S₀ (spot); returns → realized volatility σ |
| JPM dividends | Yahoo Finance (same file) | `jpm_daily.csv` | Event | Dividend yield q; dividend growth feature |
| JPM daily prices (recent ~100 sessions) | Alpha Vantage `TIME_SERIES_DAILY` | `jpm_daily_alphavantage.csv` | Daily | Cross-validation of Yahoo prices; real-time backup source for the Week 7 tool |
| S&P 500 (^GSPC) | Yahoo Finance | `sp500_daily.csv` | Daily | Market-regime context; beta/correlation features |

### 2.2 Macroeconomic data (risk-free rate)

| Series | Source | File | Frequency | Role |
|---|---|---|---|---|
| DGS3MO (3M Treasury CMT) | FRED API | `fred_rates_daily.csv` | Daily (business) | Risk-free rate r for short-dated legs |
| DGS1 (1Y Treasury CMT) | FRED API | `fred_rates_daily.csv` | Daily (business) | Risk-free rate r matching T₂ = 1 year |
| DGS10 (10Y Treasury CMT) | FRED API | `fred_rates_daily.csv` | Daily (business) | Rate-momentum / curve-level feature |
| FEDFUNDS (effective FFR) | FRED API | `fred_rates_daily.csv` | Monthly | Policy-regime feature |
| T10Y2Y (10y–2y spread) | FRED API | `fred_rates_daily.csv` | Daily (business) | Yield-curve regime feature |
| ^TNX / ^IRX (Yahoo yield indices) | Yahoo Finance | `treasury_10y_yahoo_daily.csv`, `tbill_13w_yahoo_daily.csv` | Daily | Redundant backup for FRED (source cross-check) |

### 2.3 Sentiment / volatility data

| Series | Source | File | Frequency | Role |
|---|---|---|---|---|
| ^VIX (CBOE Volatility Index) | Yahoo Finance | `vix_daily.csv` | Daily | Market-implied 30d volatility; VIX–JPM correlation feature; high-vol regime flag |
| VIX-derived sentiment scores (0–1) | Derived from ^VIX (Week 2) | `vix_daily.csv` | Daily | Sentiment proxy **per mentor guidance** (replaces NewsAPI): rolling percentile rank, fear-regime flags, VIX momentum. Level = implied-vol expectation; normalized transforms = sentiment state |

### 2.4 Validation benchmark (resolved with mentor, 2026-07-15)

Per mentor guidance, CME transaction prices are **not** required. Model
validation (Week 4) will benchmark the BSM analytical chooser price against
**Monte Carlo simulation** under the same GBM dynamics (convergence check on
simulated payoffs at the choice date). No additional external data source is
needed for validation; MC is generated in-house from the calibrated parameters
(S₀, σ, r, q from the series above).

## 3. Collection Infrastructure

- **Primary feed:** Yahoo Finance via `yfinance` (no key, no rate limit at daily granularity).
- **Rates feed:** FRED REST API (free key, 120 req/min limit — we use 5 series).
- **Backup feed:** Alpha Vantage (free tier: 25 req/day, `compact` = latest ~100 sessions only; full history is a premium feature, hence backup-only role).
- Keys are stored in `.env` (git-ignored); see `.env.example`.
- One-shot pull: `python -m src.collect_all`. Automated scheduling via GitHub Actions is a Week 2 deliverable.

## 4. Known Data Caveats

1. **FRED missing values**: market holidays are encoded as `"."` → parsed to NaN; to be interpolated in Week 2 cleaning.
2. **Calendar alignment**: FRED uses business-day calendar (1850 rows) vs. NYSE trading calendar (1760 rows); Week 2 pipeline aligns everything to the NYSE calendar.
3. **Adjusted vs. unadjusted prices**: both retained (`Close`, `Adj Close`); returns/volatility computed on adjusted prices, option moneyness on unadjusted.
4. **VIX is an index-level (S&P 500) implied vol**, used as a market-regime proxy for JPM, not as JPM's own implied vol.
