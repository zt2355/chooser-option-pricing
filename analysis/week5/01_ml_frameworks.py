"""ML Frameworks Walkthrough & Leakage Audits (Week 5) — 佟铸城 (Zhucheng Tong)

Run AFTER `python -m src.analysis_w5`. Re-verifies the pipeline independently:

  audit 1  purge correctness — no training label window crosses the val start
  audit 2  scaler leak-check — StandardScaler statistics equal train-only stats
  audit 3  anchored-decode identity — predicting 0 reproduces the EWMA baseline
  audit 4  test-set discipline — summary contains no test-set metrics
  audit 5  determinism — retrained ridge_anch reproduces identical predictions
  audit 6  composite consistency — summary composites recompute from suites

Run from project root:  python analysis/week5/01_ml_frameworks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import Ridge

from src.metrics import composite_score
from src.ml.dataset import HORIZON, LINEAR_COLS, build, scaled

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    summary = json.load(open(ROOT / "docs" / "week5" / "results_summary.json"))
    data = build()
    panel = data.panel

    print("== headline (validation) ==")
    for name, c in sorted(summary["composites_val"].items(), key=lambda kv: kv[1]):
        print(f"  {name:>12s}: composite {c:.3f}")

    # audit 1: purge — position of last train row + HORIZON < position of val start
    pos = {d: i for i, d in enumerate(panel.index)}
    last_train, val_start = pos[data.idx_train.max()], pos[data.idx_val.min()]
    assert last_train + HORIZON < val_start + HORIZON + 1  # sanity of positions
    assert last_train + HORIZON < pos[data.idx_val.min()] + HORIZON
    gap = val_start - last_train
    assert gap > HORIZON, f"purge violated: only {gap} rows between train end and val start"
    print(f"audit 1 PASS: {gap} rows between last train row and val start (> horizon {HORIZON})")

    # audit 2: scaler stats equal train-only stats
    mu = data.X.loc[data.idx_train].mean().to_numpy()
    assert np.allclose(mu, data.scaler.mean_, atol=1e-10)
    print("audit 2 PASS: scaler fitted on purged train only")

    # audit 3: anchored decode identity — zero prediction == EWMA baseline
    from src.analysis_w5 import price_from_sigma
    idx = data.idx_val
    sigma0 = panel.loc[idx, "sigma_ewma"] * np.exp(0.0)
    p0 = price_from_sigma(panel, idx, sigma0)
    assert np.allclose(p0, panel.loc[idx, "price_sigma_ewma"], atol=1e-9)
    print("audit 3 PASS: anchored model predicting 0 reproduces the EWMA baseline exactly")

    # audit 4: test-set discipline
    dumped = json.dumps(summary)
    assert "pricing_suites_test" not in dumped and "composites_test" not in dumped
    assert summary["splits"]["test_sealed"][2] == len(data.idx_test)
    print("audit 4 PASS: no test-set metrics anywhere in the Week 5 summary (sealed)")

    # audit 5: determinism — retrain ridge_anch, compare to saved-model preds
    import joblib
    saved = joblib.load(ROOT / "models" / "ridge_anch.joblib")
    m = Ridge(alpha=1.0, random_state=42).fit(
        scaled(data, data.idx_train, LINEAR_COLS),
        data.y_sigma_ratio.loc[data.idx_train].to_numpy())
    pv_new = m.predict(scaled(data, data.idx_val, LINEAR_COLS))
    pv_saved = saved.predict(scaled(data, data.idx_val, LINEAR_COLS))
    assert np.allclose(pv_new, pv_saved, atol=1e-10)
    print("audit 5 PASS: ridge_anch retrained from scratch reproduces saved model exactly")

    # audit 6: composite recompute
    ref = summary["ewma_reference_val"]
    for name, suite in summary["pricing_suites_val"].items():
        assert abs(composite_score(suite, ref) - summary["composites_val"][name]) < 1e-9
    print("audit 6 PASS: composites recompute from saved metric suites")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
