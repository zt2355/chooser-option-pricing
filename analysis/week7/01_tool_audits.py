"""Pricing Tool & Scenario Audits (Week 7) — 佟铸城 (Zhucheng Tong)

Run AFTER `python -m src.analysis_w7`. Verifies the tool's pricing service:

  audit 1  quote internal consistency (primary = closed form; ML price =
           anchored decode; band = regime-conditional W4 MAE)
  audit 2  no-arbitrage ordering at live params: max(C,P) < chooser < straddle
  audit 3  scenario engine: base row == quote; vol+50% row == direct compute;
           price monotone increasing in sigma
  audit 4  stress grid sanity: monotone in sigma at fixed rate; saved min/max
           bracket the base price
  audit 5  determinism: two independent service loads produce identical quotes
  audit 6  UI layer purity: the Streamlit app contains no pricing formulas —
           it only calls src.tool.pricing_service

Week 12: the primary anchor became GARCH-1d (EWMA fallback for short histories),
so the primary-price and scenario checks use state.sigma_anchor and the band of the
quote's own anchor; the ML reference became the GARCH-anchored cross-sectional GBDT,
so the decode check is sigma_ml = sigma_anchor * exp(ratio).

Run from project root:  python analysis/week7/01_tool_audits.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.models.chooser import chooser_closed_form
from src.tool.pricing_service import (BAND_MAE_BY_ANCHOR, load_market_state,
                                      ml_reference_sigma, price_chooser, quote,
                                      scenario_table)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    state = load_market_state()
    q = quote(state)
    print(f"live quote {q['as_of']}: primary {q['primary_price']:.4f}, "
          f"ML ref {q['ml_reference_price']:.4f}, regime {q['regime']}")

    # audit 1
    direct = float(chooser_closed_form(S0=state.S, K=state.S, r=state.r_1y,
                                       q=state.q, sigma=state.sigma_anchor,
                                       t1=0.5, T2=1.0))
    assert abs(direct - q["primary_price"]) < 1e-10
    sigma_ml, ratio = ml_reference_sigma(state)
    assert abs(sigma_ml - state.sigma_anchor * np.exp(ratio)) < 1e-10      # Week 12: GARCH-anchored model
    lo, hi = q["primary_band"]
    assert abs((hi - lo) / 2 - BAND_MAE_BY_ANCHOR[q["anchor"]][q["regime"]]) < 1e-9
    print("audit 1 PASS: quote = closed form; ML decode exact; band = W4 regime MAE")

    # audit 2
    assert max(q["call_leg"], q["put_leg"]) < q["primary_price"] < q["straddle"]
    print("audit 2 PASS: no-arbitrage ordering holds at live parameters")

    # audit 3
    tbl = scenario_table(state)
    base_row = float(tbl.loc[tbl["情景"] == "基准(当前市场)", "价格"].iloc[0])
    assert abs(base_row - round(q["primary_price"], 4)) < 1e-9
    vol_row = float(tbl.loc[tbl["情景"] == "波动率 +50%", "价格"].iloc[0])
    direct_vol = price_chooser(state.S, state.S, state.r_1y, state.q,
                               state.sigma_anchor * 1.5, 0.5, 1.0)
    assert abs(vol_row - round(direct_vol, 4)) < 1e-9
    sigs = np.linspace(0.05, 1.0, 20)
    prices = [price_chooser(state.S, state.S, state.r_1y, state.q, s_, 0.5, 1.0)
              for s_ in sigs]
    assert all(b > a for a, b in zip(prices, prices[1:]))
    print("audit 3 PASS: scenario rows recompute exactly; price strictly "
          "increasing in sigma")

    # audit 4
    summary = json.load(open(ROOT / "docs" / "week7" / "results_summary.json"))
    g = summary["stress_grid"]
    assert g["price_min"] < q["primary_price"] < g["price_max"]
    row = [price_chooser(state.S, state.S, state.r_1y, state.q,
                         state.sigma_anchor * m, 0.5, 1.0)
           for m in g["sigma_mult"]]
    assert all(b > a for a, b in zip(row, row[1:]))
    print("audit 4 PASS: stress grid brackets base price; monotone in sigma")

    # audit 5
    q2 = quote(load_market_state())
    assert q == q2
    print("audit 5 PASS: service fully deterministic across independent loads")

    # audit 6
    app_src = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "pricing_service" in app_src
    for banned in ("norm.cdf", "np.exp((r", "d1", "black"):
        assert banned not in app_src, f"pricing logic leaked into UI: {banned}"
    print("audit 6 PASS: UI layer contains no pricing formulas (thin wrapper)")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
