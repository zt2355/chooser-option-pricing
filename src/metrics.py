"""Evaluation metric suite + weighted composite score (mentor request, W4).

Eight metrics on pricing errors e_t = P_model - P_oracle (rel_t = e_t / P_oracle):

  Average accuracy   MAE, RMSE, MedAE
  Direction          Bias (mean signed error; sign kept for reporting,
                     |Bias| used in the composite)
  Tail / robustness  P95AE (95th pct of |e|), RegimeWorstMAE
                     (max of calm-MAE and fear-MAE — a model must not buy its
                     average by failing in one regime)
  Explanatory power  R2 (of price levels vs oracle)
  Usability          Hit10 (share of days with |rel error| < 10%)

Composite (lower = better):
  every metric is transformed to lower-better (1-R2, 1-Hit10, |Bias|),
  divided by the reference model's value (ratio: reference == 1), then
  combined as a weighted arithmetic mean. A model beats the reference iff
  its composite < 1. Reference in Week 4: the EWMA baseline.

Weights (sum to 1) and their rationale:
  MAE .20 / RMSE .10 / MedAE .05   average accuracy        (.35)
  P95AE .15 / RegimeWorstMAE .15   tails & regime robust.  (.30)
  |Bias| .10                       systematic mispricing   (.10)
  1-R2  .10                        explanatory power       (.10)
  1-Hit10 .15                      day-level usability     (.15)
Robustness family is deliberately weighted ~equal to average accuracy:
for a pricing model, being right on average while blowing up in one regime
is not acceptable. Weight-sensitivity of the RANKING is checked in
`weight_sensitivity` (equal / accuracy-heavy / robustness-heavy variants).
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "MAE": 0.20, "RMSE": 0.10, "MedAE": 0.05,
    "P95AE": 0.15, "RegimeWorstMAE": 0.15,
    "AbsBias": 0.10, "OneMinusR2": 0.10, "MissRate10": 0.15,
}

ALT_WEIGHTS = {
    "equal": {k: 1 / 8 for k in WEIGHTS},
    "accuracy-heavy": {"MAE": 0.35, "RMSE": 0.20, "MedAE": 0.10, "P95AE": 0.075,
                       "RegimeWorstMAE": 0.075, "AbsBias": 0.05,
                       "OneMinusR2": 0.05, "MissRate10": 0.10},
    "robustness-heavy": {"MAE": 0.10, "RMSE": 0.05, "MedAE": 0.05, "P95AE": 0.25,
                         "RegimeWorstMAE": 0.25, "AbsBias": 0.10,
                         "OneMinusR2": 0.05, "MissRate10": 0.15},
}


def metric_suite(pred: pd.Series, oracle: pd.Series,
                 regime: pd.Series) -> dict:
    """All eight metrics (reporting form: Bias signed, R2 and Hit10 as-is)."""
    e = pred - oracle
    rel = e / oracle
    calm_mae = float(e[regime == 0].abs().mean())
    fear_mae = float(e[regime == 1].abs().mean())
    ss_res = float((e**2).sum())
    ss_tot = float(((oracle - oracle.mean()) ** 2).sum())
    return {
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e**2).mean())),
        "MedAE": float(e.abs().median()),
        "Bias": float(e.mean()),
        "P95AE": float(e.abs().quantile(0.95)),
        "RegimeWorstMAE": max(calm_mae, fear_mae),
        "R2": 1.0 - ss_res / ss_tot,
        "Hit10": float((rel.abs() < 0.10).mean()),
    }


def _lower_better(m: dict) -> dict:
    return {
        "MAE": m["MAE"], "RMSE": m["RMSE"], "MedAE": m["MedAE"],
        "P95AE": m["P95AE"], "RegimeWorstMAE": m["RegimeWorstMAE"],
        "AbsBias": abs(m["Bias"]),
        "OneMinusR2": 1.0 - m["R2"],
        "MissRate10": 1.0 - m["Hit10"],
    }


def composite_score(m: dict, reference: dict, weights: dict | None = None) -> float:
    """Weighted mean of lower-better metric ratios vs the reference model.
    reference model scores exactly 1.0 against itself."""
    weights = weights or WEIGHTS
    lb, ref = _lower_better(m), _lower_better(reference)
    return float(sum(w * (lb[k] / ref[k]) for k, w in weights.items()))


def weight_sensitivity(all_metrics: dict[str, dict], reference_name: str) -> dict:
    """Composite ranking under the main and three alternative weightings."""
    ref = all_metrics[reference_name]
    out = {}
    for scheme_name, w in {"main": WEIGHTS, **ALT_WEIGHTS}.items():
        scores = {name: composite_score(m, ref, w) for name, m in all_metrics.items()}
        out[scheme_name] = {
            "scores": scores,
            "ranking": sorted(scores, key=scores.get),
        }
    return out
