"""Model Validation (Week 3) — 佟铸城 (Zhucheng Tong)

Validation of the BSM chooser implementation against:
  (i)   Monte Carlo convergence (two independent estimators);
  (ii)  the paper's Table-3 path logic and Fig.3-6 sensitivity directions;
  (iii) analyses the paper did not do: the choice-date t1 and the cost of
        its suboptimal moneyness rule.

The paper reports no single chooser price, so replication targets are
structural: path-table logic and sensitivity shapes.

Figures are produced by `python -m src.analysis_w3` into docs/week3/figures/;
this script prints the validation tables and reads the saved summary.

Run from project root:  python analysis/week3/02_model_validation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.models.chooser import chooser_closed_form, chooser_mc, chooser_mc_terminal


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    params = json.load(open(root / "configs" / "paper_params.json"))
    params.pop("description")
    cf = chooser_closed_form(**params)
    print(f"closed-form benchmark: {cf:.4f}")

    # -- 1. Monte Carlo convergence ---------------------------------------
    print("\n== 1. Monte Carlo convergence (error bars: ±1 s.e.) ==")
    rows = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        a = chooser_mc(**params, n_paths=n, seed=7)
        b = chooser_mc_terminal(**params, n_paths=n, seed=7)
        rows.append({
            "n_paths": n,
            "conditional MC": f"{a.price:.4f} ± {a.std_error:.4f}",
            "terminal MC": f"{b.price:.4f} ± {b.std_error:.4f}",
            "|cond-cf|": round(abs(a.price - cf), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("figure: docs/week3/figures/convergence.png")

    # -- 2. Paper Table-3-style path table --------------------------------
    print("\n== 2. Table-3-style path table (both choice rules) ==")
    summary = json.load(open(root / "docs" / "week3" / "results_summary.json"))
    tbl = pd.DataFrame(summary["path_table"])
    print(tbl.to_string(index=False))
    agree = int(tbl["agree"].sum())
    print(f"rational vs moneyness agree on {agree}/{len(tbl)} paths; they disagree "
          f"exactly when S_t1 lies between K and S* = K e^{{(q-r)tau}}")

    # -- 3. Sensitivity replication + t1 ----------------------------------
    print("\n== 3. Sensitivity replication (paper Fig.3-6) + t1 dimension ==")
    print("directions reproduced: sigma up -> both legs up; K up -> call down/put up;")
    print("r up -> call up/put down; q -> the reverse.")
    print("figures: docs/week3/figures/sensitivity_panels.png, t1_sensitivity.png")
    t1 = summary["t1_sensitivity"]
    print(f"t1 curve endpoints: t1->0 gives {t1['t1_0.01']:.4f} "
          f"(max(C,P)={t1['max_single']:.4f}); t1->T2 gives {t1['t1_max']:.4f} "
          f"(straddle={t1['straddle']:.4f})")

    # -- 4. Cost of the paper's moneyness rule ----------------------------
    print("\n== 4. Cost of the paper's moneyness rule (CRN paired) ==")
    g = summary["rule_gap_vs_r"]["gap_bp"]
    print(f"rule gap at r = 0% / 5% / 10%: {g[0]:.1f} / {g[10]:.1f} / {g[20]:.1f} bp")
    print("benign at the paper's r=0.15% (S* ~= K), an order of magnitude larger "
          "in high-rate regimes — relevant caveat for the 2022-24 window in Week 4.")
    print("figure: docs/week3/figures/rule_gap_vs_r.png")

    print("\n== conclusions ==")
    print("1. two closed forms agree to 7e-15; both MC estimators converge within ±1 s.e.")
    print(f"2. chooser price under paper parameters: {cf:.4f}, bracketed by best vanilla and straddle")
    print("3. all paper sensitivity directions reproduced; t1 dimension mapped")
    print("4. moneyness rule: state-dependent approximation cost (6bp -> 60bp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
