"""BSM Chooser Option Pricing Model (Week 3) — 佟铸城 (Zhucheng Tong)

Implementation walkthrough of the simple chooser pricer replicating
Huang, Wang & Wan (2021), "Exploration of JPMorgan Chooser Option Pricing".

Theory
------
A simple chooser lets the holder decide at t1 whether the option is a
European call or put (common strike K, expiry T2). At t1:

    V(t1) = max( C(S_t1, K, tau), P(S_t1, K, tau) ),   tau = T2 - t1.

Using put-call parity  P = C - S e^{-q tau} + K e^{-r tau}:

    V(t1) = C + max( 0, K e^{-r tau} - S_t1 e^{-q tau} ),

so the chooser is a STATIC PORTFOLIO: one call (K, T2) plus e^{-q tau}
units of a put with strike K* = K e^{-(r-q) tau} expiring at t1
(Rubinstein 1991) — hence a closed form under BSM.

Run from project root:  python analysis/week3/01_bsm_chooser_model.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.bsm import bs_call, bs_put
from src.models.chooser import (
    chooser_closed_form,
    chooser_closed_form_direct,
    chooser_mc,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    params = json.load(open(root / "configs" / "paper_params.json"))
    params.pop("description")
    print("== parameters (paper Table 2) ==")
    for k, v in params.items():
        print(f"  {k:>6s} = {v}")

    # -- closed form: two independent implementations must agree ---------
    print("\n== closed form: two independent implementations ==")
    cf = chooser_closed_form(**params)
    cfd = chooser_closed_form_direct(**params)
    print(f"  decomposition : {cf:.6f}")
    print(f"  direct formula: {cfd:.6f}")
    print(f"  |difference|  : {abs(cf - cfd):.2e}")
    assert abs(cf - cfd) < 1e-10, "closed-form implementations disagree"

    # -- economic sanity: max(C,P) < chooser < straddle -------------------
    print("\n== economic sanity checks ==")
    call = bs_call(params["S0"], params["K"], params["r"], params["q"],
                   params["sigma"], params["T2"])
    put = bs_put(params["S0"], params["K"], params["r"], params["q"],
                 params["sigma"], params["T2"])
    print(f"  vanilla call {call:.4f} | vanilla put {put:.4f} | straddle {call + put:.4f}")
    print(f"  chooser {cf:.4f}")
    assert max(call, put) < cf < call + put, "no-arbitrage ordering violated"
    print(f"  flexibility premium over call: {cf - call:+.4f}")
    print(f"  discount vs straddle         : {cf - (call + put):+.4f}")

    # -- Monte Carlo check (full study in 02_model_validation.py) ---------
    print("\n== Monte Carlo check ==")
    mc_o = chooser_mc(**params, n_paths=1_000_000, rule="optimal")
    mc_m = chooser_mc(**params, n_paths=1_000_000, rule="moneyness")
    print(f"  MC optimal   : {mc_o.price:.4f} ± {mc_o.std_error:.4f}  (closed form {cf:.4f})")
    print(f"  MC moneyness : {mc_m.price:.4f} ± {mc_m.std_error:.4f}")
    gap = mc_o.price - mc_m.price
    print(f"  paired rule gap (same seed): {gap:.4f} = {gap / cf * 1e4:.1f} bp")
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
