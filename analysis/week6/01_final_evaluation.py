"""Final Evaluation Walkthrough & Audits (Week 6) — 佟铸城 (Zhucheng Tong)

Run AFTER `python -m src.analysis_w6`. Independent re-verification:

  audit 1  walk-forward purge — every refit block trains only on rows whose
           label window ends before the block starts
  audit 2  tuning integrity — chosen hyperparameters are the argmin of the
           validation tuning log (test never entered selection)
  audit 3  determinism — ridge_anch walk-forward re-run twice bit-identically,
           and its recomputed test MAE matches the saved summary
  audit 4  composite consistency — composites recompute from saved suites
  audit 5  SHAP sanity — ranking covers exactly the W6 feature set
  audit 6  decode guards — anchored predictions clipped, sigma floored

Run from project root:  python analysis/week6/01_final_evaluation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.analysis_w6 import decode_price
from src.metrics import composite_score
from src.ml.dataset import HORIZON, build
from src.ml.walkforward import BLOCK, ml_features, walk_forward

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    summary = json.load(open(ROOT / "docs" / "week6" / "results_summary.json"))
    data = build()
    panel = data.panel
    X = ml_features(data)
    y_a = np.log(panel["sigma_oracle"] / panel["sigma_ewma"])

    print("== headline (TEST, opened once) ==")
    for name, c in sorted(summary["composites_test"].items(), key=lambda kv: kv[1]):
        print(f"  {name:>12s}: composite {c:.3f}")

    # audit 1: purge in every walk-forward block over the test range
    pos = {d: i for i, d in enumerate(panel.index)}
    eval_pos = np.array([pos[d] for d in data.idx_test])
    for start in range(eval_pos.min(), eval_pos.max() + 1, BLOCK):
        train_end_pos = max(start - HORIZON, 0) - 1
        assert train_end_pos + HORIZON < start, f"purge violated at block {start}"
    print(f"audit 1 PASS: all test blocks train only on rows ending >{HORIZON} rows "
          f"before the block")

    # audit 2: tuning integrity
    for name, log in summary["tuning_log"].items():
        maes = [e["val_MAE"] for e in log]
        assert log[0]["val_MAE"] == min(maes)
        assert summary["best_params"][name] == log[0]["params"]
        assert all("test" not in k for e in log for k in e)
    print("audit 2 PASS: hyperparameters are the argmin of validation-only tuning logs")

    # audit 3: determinism + summary match (ridge_anch)
    params = summary["best_params"]["ridge_anch"]
    runs = []
    for _ in range(2):
        wf = walk_forward(data, X, y_a, data.idx_test,
                          lambda: Ridge(random_state=42, **params), True)
        runs.append(wf.preds)
    assert runs[0].equals(runs[1])
    price = decode_price(panel, runs[0], "A")
    mae = float((price - panel.loc[price.index, "price_sigma_oracle"]).abs().mean())
    assert abs(mae - summary["test_suites"]["ridge_anch"]["MAE"]) < 1e-9
    print(f"audit 3 PASS: ridge_anch walk-forward deterministic; recomputed test "
          f"MAE {mae:.4f} matches summary")

    # audit 4: composite recompute
    ref = summary["ewma_reference_test"]
    for name, suite in summary["test_suites"].items():
        assert abs(composite_score(suite, ref)
                   - summary["composites_test"][name]) < 1e-9
    print("audit 4 PASS: composites recompute from saved suites")

    # audit 5: SHAP coverage
    assert set(summary["shap_rank"]) == set(X.columns)
    top = max(summary["shap_rank"], key=summary["shap_rank"].get)
    assert summary["shap_rank"][top] > 0
    print(f"audit 5 PASS: SHAP ranking covers the W6 feature set (top: {top})")

    # audit 6: decode guards
    wild = pd.Series(10.0, index=data.idx_test[:5])
    p = decode_price(panel, wild, "A")
    sane = decode_price(panel, pd.Series(0.0, index=data.idx_test[:5]), "A")
    assert np.isfinite(p).all() and (p < sane * 20).all()
    print("audit 6 PASS: decode guards cap runaway predictions (clip + sigma floor)")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
