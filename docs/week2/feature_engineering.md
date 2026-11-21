# Feature Engineering Documentation (v2)

**Week 2 deliverable** · Pipeline: `python -m src.preprocess` · Output: `data/processed/`
*v2 (per mentor feedback): added per-feature selection rationale and correlation/importance analysis (`python -m src.analysis_w2`, figures in `figures/`). Full per-feature meaning/computation/rationale prose is in the Chinese PDF report; the tables below give the compact version.*

## 1. Cleaning decisions

| Step | Decision | Rationale |
|---|---|---|
| Calendar alignment | All series reindexed to the NYSE trading calendar (JPM's 1760 trading days) | Options on JPM trade on this calendar; macro series must not inject phantom rows |
| FRED holiday gaps | Time-interpolation for daily series (DGS3MO/DGS1/DGS10/T10Y2Y) | Rates move slowly; linear-in-time interpolation is standard and unbiased |
| FEDFUNDS (monthly) | Forward-fill (step function) | A policy rate holds between FOMC updates; interpolating would invent policy moves |
| Outliers (IQR, k=3) | **Flagged and retained**, audit trail in `outlier_report.json` | All 21 flagged return days and 18 VIX days are genuine market events (worst: 2020-03-13/16, COVID crash). Deleting them would erase exactly the high-volatility regimes this project studies (Week 4 failure-mode analysis) |

## 2. Feature set (18 features; requirement ≥ 10)

### Returns & realized volatility
| Feature | Definition | Role |
|---|---|---|
| `log_return` | ln(AdjClose_t / AdjClose_{t-1}) | Base series for vol estimation |
| `realized_vol_21d/63d/252d` | Rolling std of log returns × √252 | Multi-horizon vol term structure; direct BSM σ candidates |
| `parkinson_vol_21d` | High-low range estimator × √252 | More efficient intraday-information vol estimate |

### Dividends
| Feature | Definition | Role |
|---|---|---|
| `div_yield_ttm` | Trailing-252d dividend sum / unadjusted close | BSM dividend yield q |
| `div_growth_yoy` | YoY change of TTM dividend sum | Dividend dynamics feature (project spec) |

### Rates
| Feature | Definition | Role |
|---|---|---|
| `r_3m`, `r_1y` | DGS3MO/100, DGS1/100 | Risk-free r (T₁=0.5y and T₂=1y legs) |
| `curve_slope` | T10Y2Y spread | Yield-curve regime |
| `rate_momentum_21d` | 21d change in DGS10 | Rate momentum (project spec) |

### Market context
| Feature | Definition | Role |
|---|---|---|
| `spx_log_return` | S&P 500 daily log return | Market factor |
| `beta_63d` | Rolling cov(r_JPM, r_SPX)/var(r_SPX) | Systematic exposure (mean 0.93 over sample — plausible for a mega-cap bank) |
| `corr_jpm_vix_63d` | Rolling corr(JPM return, ΔVIX) | VIX–JPM correlation (project spec) |

### Sentiment (all VIX-derived — per mentor guidance, VIX replaces the news feed)
| Feature | Definition | Role |
|---|---|---|
| `vix_level` | VIX close | Implied-vol expectation (volatility role) |
| `vix_sentiment_pct` | Rolling 252d percentile rank ∈ [0,1] | The required 0–1 sentiment score (sentiment role) |
| `vix_momentum_5d` | 5d VIX change | Sentiment shift speed |
| `fear_regime` | 1 if percentile > 0.75 or VIX > 30 | High-fear regime flag |

**Dual-role control**: raw VIX level enters as a volatility feature; only normalized transforms (percentile, momentum, regime flag) represent "sentiment", limiting information overlap. VIX is market-wide, not JPM-specific — documented as a known limitation.

## 3. Outputs

| File | Rows | Window | Notes |
|---|---|---|---|
| `features_daily.csv/.parquet` | 1760 | 2018-01-02 – 2024-12-30 | Full history incl. rolling warm-up NaNs |
| `features_daily_model_ready.csv/.parquet` | 1309 | 2019-10-17 – 2024-12-30 | Zero NaN; warm-up rows dropped |
| `outlier_report.json` | — | — | IQR audit trail |

Sanity checks passed: mean 21d realized vol 26.9% (consistent with the paper's σ=28.2%); 2020-03-16 row shows return −16.2%, VIX 82.7, sentiment percentile 1.0, fear_regime 1.

## 4. Correlation & importance analysis (v2 addition)

Script: `python -m src.analysis_w2` → `figures/corr_heatmap.png`, `figures/feature_importance.png`. Sample: 1289 rows (target = forward 21d realized vol, tail trimmed).

**Correlation structure** — three blocks: rates (r_3m~r_1y +0.98; both vs curve_slope −0.87/−0.91, the 2022–23 hiking-cycle flattening), volatility (five vol features intercorrelated 0.78–0.95; expected collinearity — fine for trees, needs regularization/representative selection for linear baselines), and sentiment↔vol linkage (vix_level vs realized vols 0.78–0.82, while the percentile transform drops to 0.40–0.49 — evidence the dual-role separation works).

**Importance** — RF top: vix_level (0.182), corr_jpm_vix_63d (0.172), div_growth_yoy (0.159), vix_sentiment_pct (0.157); MI top: realized_vol_252d (0.113). Forward-looking VIX-family features beat backward-looking realized vols under RF — consistent with the variance-risk-premium literature and direct evidence for the Week 5 route-A design. fear_regime ≈ 0 under RF (absorbed by vix_sentiment_pct, ρ=+0.77); retained for interpretability/tool display. Caveats: div_growth_yoy likely proxies regime timing rather than causation (SHAP check in W6); this analysis is exploratory (full-sample fit) — formal W5/6 training uses the 70/15/15 time-series split.

## 5. Automation (GitHub Actions)

`.github/workflows/update_data.yml` runs the full collect → preprocess pipeline every Friday 22:30 UTC (after US close) and commits refreshed datasets. The scheduled run sets `END_DATE=today` (env override in `src/config.py`) so data stays current, while modeling code slices to ≤ 2024-12-31 for the study window. Setup: add `FRED_API_KEY` and `ALPHA_VANTAGE_API_KEY` as repository secrets; a `workflow_dispatch` button allows manual test runs.
