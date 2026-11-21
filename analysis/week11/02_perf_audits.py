"""Performance-optimisation audits (Week 11) — 佟铸城 (Zhucheng Tong)

Every Week 11 optimisation must change WHERE time goes and nothing else.

  audit 1  features_latest == DISTINCT ON scan over features_daily for every
           ticker (same dates, identical values)
  audit 2  batched incremental rebuild (rebuild_many) == per-ticker
           rebuild_incremental == full rebuild on a 5-ticker sample (tol 1e-9)
  audit 3  quote_universe from features_latest == quote_universe from the scan
           path (all fields) and == scalar quote for the sample
  audit 4  concurrent market update (live network) only rewrites the provider
           overlap window: every index/rates value older than 15 days is
           unchanged. Executed LAST -- see the comment at the audit.
  audit 5  planner uses the new (ticker, date DESC) index for a tail read
  audit 6  estimated table counts within 2% of exact counts
  audit 7  scale universe: sp400/sp600 securities carry NO index_membership
           rows and the research panel (build_panel) contains sp500 members only
  audit 8  Week 10 audits still pass (subprocess; includes Week 9)
  audit 9  round-2 theory: selection used no test rows; test opened once
           (max_train_date_selection < test_start; JSON says sealed)

Run from project root with services up:  python analysis/week11/02_perf_audits.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["S", "sigma_ewma", "sigma_ml", "ml_ratio", "r", "q", "primary_price",
          "ml_reference_price", "call_leg", "put_leg", "straddle"]


def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.allclose(a, b, equal_nan=True, atol=1e-9, rtol=0)


def main() -> int:
    from src.db import repository as repo
    from src.db.engine import db_available, get_engine
    from src.db.features_db import rebuild_full, rebuild_incremental, rebuild_many
    from src.tool import pricing_service as ps
    assert db_available()

    # audit 1
    a = repo.read_latest_features(); b = repo.read_latest_features_scan()
    assert set(a.index) == set(b.index), (len(a), len(b))
    num = [c for c in a.columns if a[c].dtype.kind == "f"]
    assert (a.loc[b.index, "date"] == b["date"]).all()
    assert _close(a.loc[b.index, num], b[num])
    print(f"audit 1 PASS: features_latest == DISTINCT ON scan for {len(a)} tickers")

    # audit 2
    rng = np.random.default_rng(11)
    with_prices = [t for t in repo.list_securities()["ticker"] if t in repo.last_price_dates()]
    sample = ["JPM"] + list(rng.choice([t for t in with_prices if t != "JPM"], 4, replace=False))
    worst = 0.0
    for t in sample:
        rebuild_full(t); full = repo.read_features(t)
        rebuild_incremental(t); inc = repo.read_features(t)
        worst = max(worst, float((inc - full).abs().max().max()))
    rebuild_many(sample)
    for t in sample:
        many = repo.read_features(t); full = None
        rebuild_full(t); full = repo.read_features(t)
        worst = max(worst, float((many - full).abs().max().max()))
    assert worst < 1e-9, worst
    print(f"audit 2 PASS: rebuild_many == rebuild_incremental == rebuild_full on {sample} (max diff {worst:.1e})")

    # audit 3
    u_fast = ps.quote_universe(sample)
    latest_scan = repo.read_latest_features_scan(sample)
    # scan path through the same pricing code: temporarily route read_latest_features to the scan
    orig = repo.read_latest_features
    repo.read_latest_features = lambda tickers=None, as_of=None: repo.read_latest_features_scan(tickers, as_of)
    try:
        u_scan = ps.quote_universe(sample)
    finally:
        repo.read_latest_features = orig
    assert _close(u_fast[[f for f in FIELDS if f in u_fast.columns]], u_scan[[f for f in FIELDS if f in u_scan.columns]])
    for t in sample:
        q = ps.quote(ps.load_market_state(ticker=t, source="db"), cached=False)
        for k in FIELDS:
            x, y = float(u_fast.loc[t, k]), q[k]
            assert np.isnan(x) == np.isnan(y) and (np.isnan(x) or abs(x - y) < 1e-9), (t, k)
    print("audit 3 PASS: latest-table quotes == scan-path quotes == scalar quotes")

    # audit 5
    with get_engine().connect() as c:
        plan = c.execute(sa.text("EXPLAIN (FORMAT JSON) SELECT * FROM features_daily WHERE ticker='JPM' ORDER BY date DESC LIMIT 760")).scalar()
        txt = json.dumps(plan)
        assert "ix_features_ticker_date_desc" in txt or "features_daily_pkey" in txt, txt[:300]
        assert "Seq Scan" not in txt
    print("audit 5 PASS: tail reads use an index (no sequential scan)")

    # audit 6
    est, exact = repo.table_counts(), repo.table_counts(exact=True)
    for k in ("features_daily", "prices_daily", "quotes_daily"):
        assert abs(est[k] - exact[k]) <= 0.02 * max(exact[k], 1), (k, est[k], exact[k])
    print(f"audit 6 PASS: estimated counts within 2% of exact ({est['features_daily']} vs {exact['features_daily']})")

    # audit 7
    sec = repo.list_securities(active_only=False)
    mem = repo.read_membership("sp500")
    scale = set(sec[sec["universe"].isin(["sp400", "sp600"])]["ticker"])
    assert not (scale & set(mem["ticker"])), "scale tickers leaked into sp500 membership"
    from src.analysis_w10 import build_panel
    pt = set(build_panel()["ticker"].unique())
    assert pt <= set(mem["ticker"]) and not (pt & scale)
    print(f"audit 7 PASS: {len(scale)} scale-test tickers excluded from membership and from the research panel ({len(pt)} tickers)")

    # audit 8
    res = subprocess.run([sys.executable, "analysis/week10/01_universe_audits.py"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0 and "All audits passed" in res.stdout, res.stdout[-600:] + res.stderr[-600:]
    print("audit 8 PASS: Week 10 (and Week 9) audits still pass")

    # audit 9
    js = json.load(open(ROOT / "docs/week11/ml_round2_summary.json"))
    assert js["sealed_until_selection_done"] is True
    assert pd.Timestamp(js["max_train_date_selection"]) < pd.Timestamp(js["split"]["test_start"])
    assert pd.Timestamp(js["max_train_date_selection"]) < pd.Timestamp(js["split"]["val_end"])
    assert pd.Timestamp(js["max_train_date_test"]) < pd.Timestamp(js["split"]["test_start"])
    print(f"audit 9 PASS: selection trained on data <= {js['max_train_date_selection']}; test opened once from {js['split']['test_start']}")

    # audit 4 -- run LAST on purpose: it is the only audit that touches the network and
    # mutates inputs (index/rates overlap window). Run earlier, fresh data (intraday bar,
    # a FRED print published minutes ago) would leave stored quotes stale and make the
    # Week 10 idempotency audit (run inside audit 8) fail for a reason unrelated to code.
    from src.service.pipeline import RunReport, update_market_tables
    before_idx = {s: repo.read_index(s) for s in ["^VIX", "^GSPC", "^TNX", "^IRX"]}
    before_rates = repo.read_rates()
    rep = RunReport(); update_market_tables(False, rep)          # concurrent path (idempotent upsert)
    after_idx = {s: repo.read_index(s) for s in before_idx}
    after_rates = repo.read_rates()
    for s in before_idx:
        common = before_idx[s].index.intersection(after_idx[s].index)
        assert len(common) >= len(before_idx[s]) - 15
        old = before_idx[s].loc[common[common < common.max() - pd.Timedelta(days=15)], "close"]
        assert _close(old, after_idx[s].loc[old.index, "close"]), s
    common = before_rates.index.intersection(after_rates.index)
    old = common[common < common.max() - pd.Timedelta(days=15)]
    assert _close(before_rates.loc[old], after_rates.loc[old])
    print(f"audit 4 PASS: concurrent market update only touched the overlap window "
          f"({'no source errors' if not rep.errors else rep.errors})")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
