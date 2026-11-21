# Baseline Performance Benchmark (Week 4)

Reference numbers that the Week 5–6 ML models must beat. Produced by
`python -m src.analysis_w4`; audited by `analysis/week4/01_baseline_evaluation.py`.

## Setup

Daily fresh 1y ATM chooser (t1=0.5, T2=1, K=spot), calibrated per day with
r = 1y Treasury, q = TTM dividend yield, σ = backward-looking estimator.
**Oracle** = same contract priced with the volatility actually realized over
the option's 1y life. Panel: 1310 trading days, 2019-10-17 → 2024-12-31.

## Benchmark table (`baseline_metrics.csv`)

| σ estimator | MAE ($) | RMSE ($) | MAE (%) |
|---|---|---|---|
| 21d realized | 9.80 | 14.40 | 36.1 |
| 63d realized | 9.76 | 13.37 | 37.4 |
| 252d realized | 10.37 | 13.61 | 41.7 |
| **EWMA (λ=0.94)** | **9.44** | **13.52** | **35.3** |

## Failure-mode structure (the key finding)

- **Fast estimators (21d, EWMA) fail in fear regimes** (MAE 11.4 / 10.8 vs 9.3 / 9.0 in calm): they chase the vol spike that forward realized vol then mean-reverts away from.
- **The 252d estimator fails in post-crisis calm** (calm 10.9 > fear 8.6; 2021: $10.34 vs 21d's $6.13): its window drags crisis vol for a full year.
- No single backward window dominates across regimes → direct motivation for the Week 5 ML design (conditionally combining windows + forward-looking VIX features).

## Extended metric suite + weighted composite (v2, per mentor)

Eight metrics (MAE, RMSE, MedAE, Bias, P95AE, RegimeWorstMAE, R², Hit10) and a
weighted composite — lower-better ratios vs the EWMA reference, weights
0.35 average accuracy / 0.30 tail-robustness / 0.10 bias / 0.10 R² / 0.15 usability
(rationale + formula in `src/metrics.py`; full table in `baseline_metrics.csv`).
**Composite < 1 beats the EWMA baseline.** Findings: the four backward baselines
are a statistical tie (0.98–1.07, ranking weight-sensitive, 21d always last), and
**all baselines have negative R² (~−1.1) against oracle prices** — worse than
predicting the mean, the sharpest statement of constant-vol failure. The W5–6 ML
goal is therefore: positive R² first, composite < 1 across ALL weighting schemes.

## Honest caveats

1. Late-2019 errors (~$32) stem from the COVID crash inside the forward window — unforecastable by any backward method; ML should be expected to improve *predictable* vol dynamics, not black swans.
2. EWMA over-reacts to single-day jumps (2024-11 election day, JPM +11.5% → transient price spike).
3. MC implementation validation (paper params): MAE 0.051 at 100k paths (0.18% of price), 1/√n convergence confirmed.

## Rule-cost recheck on real dates (W3 follow-up)

Moneyness-rule gap: 14.4 bp at r=0.07% (2021-06-30) → 32.2 bp at r=5.49% (2023-10-02). Rational rule used throughout this project.
