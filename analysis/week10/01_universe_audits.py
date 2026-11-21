"""Universe / batch-pipeline audits (Week 10) — 佟铸城 (Zhucheng Tong)

  audit 1   membership: today's point-in-time set == current constituent table;
            no overlapping intervals per ticker; JPM member on every study day;
            daily count within [495, 515] on four probe dates
  audit 2   incremental feature rebuild == full rebuild, bit-for-bit, on a
            5-ticker sample (JPM + 4 random members)
  audit 3   stored sigma_ewma / r_1y_dev == full-history derive() (same sample)
  audit 4   quote_universe row == scalar quote() for the sample (all fields)
  audit 5   scenario_universe == scalar scenario_table for the sample
  audit 6   no-arbitrage ordering max(C,P) <= chooser <= straddle (1e-9) for EVERY
            ticker in the universe quote; band half-width == regime MAE of the
            row's anchor (Week 12: GARCH-1d, EWMA fallback);
            reports how many tickers hit the dividend-yield cap
  audit 7   idempotency: dry-run pipeline twice on the sample -> quotes unchanged
  audit 8   coverage: active tickers with prices >= 95% of current constituents;
            the missing ones are listed with their note
  audit 9   Week 9 online-stack audits still pass (subprocess)
  audit 10  theory line sealed: docs/week10/panel_summary.json says sealed and
            max training date < test start

Run from project root with services up:  python analysis/week10/01_universe_audits.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["S", "sigma_ewma", "sigma_ml", "ml_ratio", "r", "q", "primary_price",
          "ml_reference_price", "call_leg", "put_leg", "straddle"]


def main() -> int:
    from src.db import repository as repo
    from src.db.engine import db_available
    from src.db.features_db import rebuild_full, rebuild_incremental
    from src.service.pipeline import run_daily
    from src.tool import pricing_service as ps
    from src.universe.sp500 import universe_on
    assert db_available(), "PostgreSQL not reachable"

    # audit 1
    mem = repo.read_membership("sp500")
    cur = pd.read_csv(ROOT / "data/universe/sp500_current.csv")
    today = pd.Timestamp.today().normalize()
    assert universe_on(mem, today) == set(cur["ticker"])
    for t, g in mem.groupby("ticker"):
        g = g.sort_values("start")
        ends = g["end"].fillna(pd.Timestamp("2100-01-01")).to_numpy()
        assert (g["start"].to_numpy()[1:] >= ends[:-1]).all(), f"overlapping intervals for {t}"
    jpm = mem[mem["ticker"] == "JPM"]
    assert len(jpm) == 1 and jpm["start"].iloc[0] <= pd.Timestamp("2018-01-02") and pd.isna(jpm["end"].iloc[0])
    counts = {d: len(universe_on(mem, d)) for d in ["2018-01-02", "2020-06-30", "2022-12-30", "2024-12-31"]}
    assert all(495 <= v <= 515 for v in counts.values()), counts
    print(f"audit 1 PASS: membership consistent; {mem['ticker'].nunique()} tickers, counts {counts}")

    # sample
    rng = np.random.default_rng(10)
    active = repo.list_securities()["ticker"].tolist()
    with_prices = [t for t in active if t in repo.last_price_dates()]
    sample = ["JPM"] + [t for t in rng.choice([t for t in with_prices if t != "JPM"], 4, replace=False)]

    # audit 2 + 3
    worst2 = worst3 = 0.0
    for t in sample:
        rebuild_full(t); full = repo.read_features(t)
        rebuild_incremental(t); inc = repo.read_features(t)
        worst2 = max(worst2, float((inc - full).abs().max().max()))
        d = ps.derive(ps.load_history(t, "db")[0])
        worst3 = max(worst3, float((inc["sigma_ewma"] - d["sigma_ewma"]).abs().max()),
                     float((inc["r_1y_dev"] - d["r_1y_dev"]).abs().max()))
    assert worst2 < 1e-9 and worst3 < 1e-9, (worst2, worst3)
    print(f"audit 2 PASS: incremental == full rebuild on {sample} (max diff {worst2:.1e})")
    print(f"audit 3 PASS: stored EWMA / r_1y_dev == derive() (max diff {worst3:.1e})")

    # audit 4 + 5
    uq = ps.quote_universe(sample)
    worst4 = worst5 = 0.0
    for t in sample:
        st = ps.load_market_state(ticker=t, source="db"); q = ps.quote(st, cached=False)
        for k in FIELDS:
            a, b = float(uq.loc[t, k]), q[k]
            assert np.isnan(a) == np.isnan(b), (t, k)          # ML fields are NaN outside ML_SCOPE
            if not np.isnan(a):
                worst4 = max(worst4, abs(a - b))
        assert uq.loc[t, "regime"] == q["regime"]
        # Week 12: the scope is data-defined (point-in-time members with a GARCH anchor), so both
        # paths must agree with each other and with the presence of an ML reference price
        assert bool(uq.loc[t, "ml_in_scope"]) == bool(q["ml_in_scope"]) == (not np.isnan(q["ml_reference_price"]))
        sc = ps.scenario_universe(uq.loc[[t]]).set_index("scenario"); tbl = ps.scenario_table(st, cached=False)
        worst5 = max(worst5, float(np.abs(sc.loc[tbl["情景"], "price"].to_numpy() - tbl["价格"].to_numpy()).max()))
    assert worst4 < 1e-9 and worst5 < 1e-9, (worst4, worst5)
    print(f"audit 4 PASS: quote_universe == scalar quote (max diff {worst4:.1e}); ML reference only inside scope ({ps.ML_SCOPE})")
    print(f"audit 5 PASS: scenario_universe == scalar scenario_table (max diff {worst5:.1e})")

    # audit 6
    all_q = ps.quote_universe()
    assert len(all_q) >= 400, len(all_q)
    # <= with tolerance: when the call is worthless (deep-ITM put, capped high yield) the
    # chooser equals the put to floating-point precision, which is still no-arbitrage-consistent
    tol = 1e-9
    ok = (np.maximum(all_q["call_leg"], all_q["put_leg"]) <= all_q["primary_price"] + tol) & (all_q["primary_price"] <= all_q["straddle"] + tol)
    assert ok.all(), all_q[~ok].index.tolist()[:10]
    capped = int((all_q["q"] >= ps.Q_CAP - 1e-12).sum())
    hw = (all_q["band_hi"] - all_q["band_lo"]) / 2
    # Week 12: each row's band belongs to the anchor that row used (GARCH-1d, EWMA fallback)
    exp = pd.Series([ps.BAND_MAE_BY_ANCHOR[a][g] for a, g in zip(all_q["anchor"], all_q["regime"])], index=all_q.index)
    assert np.allclose(hw, exp)
    print(f"audit 6 PASS: no-arbitrage ordering and regime bands hold for all {len(all_q)} tickers "
          f"({capped} priced with the dividend-yield cap {ps.Q_CAP:.0%})")

    # audit 7
    before = repo.read_universe_quotes().loc[sample]
    r1 = run_daily(sample, dry_run=True); r2 = run_daily(sample, dry_run=True)
    after = repo.read_universe_quotes().loc[sample]
    assert r1.status == r2.status == "ok", (r1.errors, r2.errors)
    num = [c for c in after.columns if after[c].dtype.kind == "f"]
    assert np.allclose(before[num].to_numpy(float), after[num].to_numpy(float), equal_nan=True)   # ML fields NaN outside scope
    print(f"audit 7 PASS: batch pipeline idempotent on the sample ({r2.timings})")

    # audit 8
    sec = repo.list_securities(active_only=False)
    cur_set = set(cur["ticker"]); have = set(with_prices)
    cov = len(cur_set & have) / len(cur_set)
    missing = sec[~sec["active"]][["ticker", "note"]]
    assert cov >= 0.95, cov
    print(f"audit 8 PASS: price coverage of current constituents {cov:.1%}; inactive {len(missing)}: "
          f"{missing['ticker'].tolist()[:12]}{'...' if len(missing) > 12 else ''}")

    # audit 9
    res = subprocess.run([sys.executable, "analysis/week9/01_online_stack_audits.py"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0 and "All audits passed" in res.stdout, res.stdout[-800:] + res.stderr[-800:]
    print("audit 9 PASS: Week 9 online-stack audits still pass")

    # audit 10
    js = json.load(open(ROOT / "docs/week10/panel_summary.json"))
    assert js["ml"]["sealed"] is True
    assert pd.Timestamp(js["ml"]["max_train_date_used"]) < pd.Timestamp(js["ml"]["split"]["test_start"])
    assert pd.Timestamp(js["ml"]["max_train_date_used"]) < pd.Timestamp(js["ml"]["split"]["val_end"])
    print(f"audit 10 PASS: test period sealed (training data ends {js['ml']['max_train_date_used']}, "
          f"test starts {js['ml']['split']['test_start']})")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
