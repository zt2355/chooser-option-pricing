"""Online-stack audits (Week 9) — 佟铸城 (Zhucheng Tong)

Verifies that moving the tool from files to PostgreSQL + Redis changed
WHERE the numbers live and nothing about WHAT they are.

  audit 1  arithmetic parity: the Week 2 FILE pipeline run on the DB's own
           price rows == the DB features (max |diff| < 1e-9, same NaN pattern).
           Week 10 lesson: Yahoo recomputes the entire adjusted-close series at
           every dividend, so live DB and snapshot files are different vintages;
           the vintage gap is printed, not asserted.
  audit 2  quote from the one-row DB state == quote from the full-history
           derivation at the same as-of date (< 1e-9); the file-snapshot
           quote is printed for reference (vintage gap)
  audit 3  Redis: cached quote == fresh recompute; scenario table identical;
           invalidate() removes the ticker's keys
  audit 4  idempotency: running the reprice pipeline twice leaves
           quotes_daily row count and values unchanged
  audit 5  the latest quotes_daily row is reproducible from features_daily
           (older rows are the day's audit trail and may drift after
           provider revisions of adjusted prices — reported, not asserted)
  audit 6  migrations are idempotent (create_all twice; table set fixed)
  audit 7  UI purity (Week 7 audit 6 extended): the app imports only the
           service/cache/repository layers, no pricing formulas
  audit 8  scheduler: cron trigger builds and next fire time is a weekday

Run from project root with services up:  python analysis/week9/01_online_stack_audits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from src import config, features as F
    from src.cache import redis_cache as cache
    from src.db import repository as repo, schema
    from src.db.engine import db_available
    from src.db.migrate import migrate
    from src.tool import pricing_service as ps
    from src.service.pipeline import run_daily
    assert db_available(), "PostgreSQL not reachable (bash scripts/dev_services.sh start)"

    # audit 1 — arithmetic parity on the SAME data vintage. Yahoo recomputes the
    # whole adjusted-close series at every dividend, so DB (live) and file
    # (snapshot) prices are different vintages; comparing features across
    # vintages is not a test of the code. We therefore run the Week 2 FILE
    # pipeline on the DB's own price rows and compare with the DB features.
    from src import preprocess as P
    px = repo.read_prices(config.TICKER).rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                                                         "adj_close": "Adj Close", "volume": "Volume",
                                                         "dividend": "Dividends", "split": "Stock Splits"})
    raw = {"jpm": px, "vix": repo.read_index("^VIX").rename(columns=str.title),
           "sp500": repo.read_index("^GSPC").rename(columns=str.title),
           "rates": repo.read_rates().rename(columns=str.upper)}
    fl = P.engineer_features(P.align_and_clean(raw))
    db = F.to_file_names(repo.read_features(config.TICKER))
    cols = ["jpm_close", "jpm_adj_close", "vix_close"] + P.FEATURE_COLS
    common = db.index.intersection(fl.index)
    assert len(common) >= 2000, len(common)
    d, f = db.loc[common, cols], fl.loc[common, cols]
    assert (d.isna() == f.isna()).all().all()
    diff = float((d - f).abs().max().max())
    assert diff < 1e-9, diff
    snap = pd.read_csv(config.DATA_PROCESSED / "features_daily.csv", index_col="date", parse_dates=True)
    vint = float((db.loc[db.index.intersection(snap.index), "jpm_adj_close"] - snap.loc[db.index.intersection(snap.index), "jpm_adj_close"]).abs().max())
    print(f"audit 1 PASS: file pipeline on DB prices == DB features on {len(common)} dates (max diff {diff:.1e}); "
          f"snapshot-vs-live adj_close vintage gap {vint:.1e} (provider revision, not code)")

    # audit 2
    as_of = str((snap.index.max() - pd.Timedelta(days=15)).date())
    sf = ps.load_market_state(as_of, source="file")
    sd = ps.load_market_state(as_of, source="db")
    qf, qd = ps.quote(sf, cached=False), ps.quote(sd, cached=False)
    fields = ["S", "sigma_ewma", "sigma_ml", "ml_ratio", "r", "q", "primary_price",
              "ml_reference_price", "call_leg", "put_leg", "straddle"]
    mx = max(abs(qf[k] - qd[k]) for k in fields)
    # different vintages again: assert on the file path rebuilt from DB prices instead
    hist_db = F.to_file_names(repo.read_features(config.TICKER))
    sd2 = ps._state_from_row(config.TICKER, ps.derive(hist_db).loc[:as_of].dropna(subset=ps.REQUIRED + ps.ML_FEATURES).iloc[-1], "db")
    qd2 = ps.quote(sd2, cached=False)
    mx2 = max(abs(qd2[k] - qd[k]) for k in fields)
    assert mx2 < 1e-9 and qd2["regime"] == qd["regime"], mx2
    print(f"audit 2 PASS: one-row DB state == full-history derivation at {as_of} (max diff {mx2:.1e}); "
          f"file-snapshot quote differs by {mx:.1e} (vintage)")

    # audit 3
    assert cache.available(), "Redis not reachable"
    state = ps.load_market_state(source="db")
    cache.invalidate(state.ticker)
    q1 = ps.quote(state)               # miss -> set
    q2 = ps.quote(state)               # hit
    q3 = ps.quote(state, cached=False)  # fresh
    assert q1 == q2 and all(abs(q2[k] - q3[k]) < 1e-12 for k in fields)
    t1, t2 = ps.scenario_table(state), ps.scenario_table(state, cached=False)
    assert t1.equals(t2)
    n = cache.invalidate(state.ticker)
    assert n >= 2 and cache.get_json(cache.key("quote", state.ticker, state.date.date(),
                                               round(state.S, 6), 0.5, 1.0, ps.MODEL_VERSION)) is None
    print(f"audit 3 PASS: cache hit == recompute; invalidate removed {n} keys")

    # audit 4
    before = repo.read_quotes(config.TICKER)
    r1 = run_daily([config.TICKER], dry_run=True); r2 = run_daily([config.TICKER], dry_run=True)
    after = repo.read_quotes(config.TICKER)
    assert r1.status == r2.status == "ok"
    assert len(before) == len(after)
    num = after.select_dtypes("number").columns
    assert np.allclose(before[num].to_numpy(dtype=float), after[num].to_numpy(dtype=float), equal_nan=True)
    print(f"audit 4 PASS: reprice pipeline idempotent ({len(after)} quote rows unchanged)")

    # audit 5 — the LATEST stored quote is reproducible from today's features.
    # Older quote rows are the audit trail of what the tool showed on that day;
    # after a provider revision of the adjusted-close history they are NOT
    # expected to match a recomputation (reported, not asserted).
    latest_date = after.index.max()
    s = ps.load_market_state(str(latest_date.date()), source="db")
    q = ps.quote(s, cached=False)
    row = after.loc[latest_date]
    worst = max(abs(q["primary_price"] - row["primary_price"]), abs(q["ml_reference_price"] - row["ml_reference_price"]))
    assert worst < 1e-9, worst
    drift = []
    for d_, r_ in after.iterrows():
        if d_ == latest_date:
            continue
        s_ = ps.load_market_state(str(d_.date()), source="db")
        drift.append(abs(ps.quote(s_, cached=False)["primary_price"] - r_["primary_price"]))
    print(f"audit 5 PASS: latest stored quote ({latest_date.date()}) reproducible (diff {worst:.1e}); "
          f"{len(drift)} older archive rows drift up to {max(drift) if drift else 0:.2f} USD after provider revisions")

    # audit 6
    t_a = migrate(); t_b = migrate()
    assert t_a == t_b == sorted(schema.metadata.tables)
    print(f"audit 6 PASS: migrate idempotent ({len(t_a)} tables)")

    # audit 7
    app_src = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "pricing_service" in app_src
    for banned in ("norm.cdf", "np.exp((r", "d1", "black", "chooser_closed_form", "sqlalchemy", "pg_insert"):
        assert banned not in app_src, f"leak into UI: {banned}"
    print("audit 7 PASS: UI imports only service/cache/repository layers")

    # audit 8
    from apscheduler.triggers.cron import CronTrigger
    import datetime as dt
    trig = CronTrigger(day_of_week="mon-fri", hour=17, minute=30, timezone="America/New_York")
    nxt = trig.get_next_fire_time(None, dt.datetime.now(dt.timezone.utc))
    assert nxt is not None and nxt.weekday() < 5 and (nxt.hour, nxt.minute) == (17, 30)
    print(f"audit 8 PASS: scheduler trigger valid, next run {nxt:%Y-%m-%d %H:%M %Z}")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
