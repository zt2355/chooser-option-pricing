"""Full-stack profiling, dashboard -> service -> database -> data feed (Week 11)
— 佟铸城 (Zhucheng Tong)

Mentor (Week 10 review): performance testing moves up the schedule. This
script is the diagnosis step: it measures WHERE the time and memory go at
universe scale (590 tickers) before any Week 11 optimisation, using only
standard tooling (cProfile, tracemalloc, PostgreSQL EXPLAIN ANALYZE,
Streamlit AppTest). Week 12 copy (writes docs/week12/profile_<tag>.json; the Week 11 folder is frozen),
with three additions: the uncached scalar quote (it now builds the cross-sectional ML
reference), the cross-section itself, and the per-day cost of the incremental GARCH
anchor simulated in memory for one new trading day on every GARCH ticker (no writes).
It writes docs/week12/profile_<tag>.json so the same
script re-run after optimisation gives a like-for-like comparison.

Layers
  L1 dashboard      Streamlit first render (all six tabs) — wall time, peak mem,
                    top functions by cumulative time
  L2 service        quote_universe / scenario_universe — cProfile split into
                    SQL vs numpy vs sklearn
  L3 database       EXPLAIN ANALYZE of the latest-row query (DISTINCT ON), of
                    the per-ticker tail reads used by the incremental rebuild,
                    table/index sizes
  L4 data feed      last daily pipeline run's stage timings; network share
  L5 process/space  RSS after each layer, DB size, Redis memory, image size

Usage: python analysis/week12/02_profile_final.py [--tag final]
"""

import cProfile
import io
import json
import pstats
import resource
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "week12"


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


def top_funcs(prof: cProfile.Profile, n=12, restrict=None):
    s = io.StringIO()
    st = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    st.print_stats(*(restrict or []), n)
    rows = []
    for line in s.getvalue().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[0].replace("/", "").isdigit() and parts[3].replace(".", "").isdigit():
            rows.append({"ncalls": parts[0], "tottime": float(parts[1]), "cumtime": float(parts[3]),
                         "func": parts[5].replace(str(ROOT) + "/", "")[-110:]})
    return rows[:n]


def bucket(prof: cProfile.Profile) -> dict:
    """Share of total time by layer keyword in the function path."""
    st = pstats.Stats(prof)
    tot = st.total_tt
    buckets = {"sqlalchemy/psycopg2": 0.0, "pandas": 0.0, "numpy/scipy": 0.0, "sklearn": 0.0, "src": 0.0, "other": 0.0}
    for (file, _, fn), (cc, nc, tt, ct, callers) in st.stats.items():
        f = file.lower()
        if "sqlalchemy" in f or "psycopg" in f:
            buckets["sqlalchemy/psycopg2"] += tt
        elif "pandas" in f:
            buckets["pandas"] += tt
        elif "numpy" in f or "scipy" in f:
            buckets["numpy/scipy"] += tt
        elif "sklearn" in f:
            buckets["sklearn"] += tt
        elif "/src/" in f:
            buckets["src"] += tt
        else:
            buckets["other"] += tt
    return {k: round(v / tot, 3) if tot else 0 for k, v in buckets.items()} | {"total_s": round(tot, 3)}


def timed_profile(fn, n=3):
    """Median wall ms over n runs + cProfile of the last run + tracemalloc peak."""
    ts = []
    for _ in range(n - 1):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    tracemalloc.start(); prof = cProfile.Profile()
    t = time.perf_counter(); prof.enable(); fn(); prof.disable(); ts.append((time.perf_counter() - t) * 1e3)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    ts.sort()
    return {"median_ms": round(ts[len(ts) // 2], 1), "peak_mb": round(peak / 2**20, 1), "buckets": bucket(prof),
            "top": top_funcs(prof)}, prof


def explain(conn, sql: str) -> dict:
    rows = conn.execute(sa.text("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql)).scalar()
    plan = rows[0]["Plan"] if isinstance(rows, list) else json.loads(rows)[0]["Plan"]
    top = json.loads(json.dumps(rows))[0]
    return {"execution_ms": round(top["Execution Time"], 1), "planning_ms": round(top["Planning Time"], 2),
            "node": plan["Node Type"], "rows": plan.get("Actual Rows"),
            "shared_hit_blocks": plan.get("Shared Hit Blocks"), "shared_read_blocks": plan.get("Shared Read Blocks"),
            "plan_head": _plan_head(plan)}


def _plan_head(plan, depth=0, out=None):
    out = [] if out is None else out
    if depth < 4:
        out.append("  " * depth + f"{plan['Node Type']} rows={plan.get('Actual Rows')} t={plan.get('Actual Total Time', 0):.1f}ms"
                   + (f" idx={plan['Index Name']}" if "Index Name" in plan else ""))
        for p in plan.get("Plans", []):
            _plan_head(p, depth + 1, out)
    return out


def _garch_one_day(repo, back: int = 0) -> dict:
    """Per-day GARCH cost (garch_tail only, incl. its history read on refit days): for every
    ticker with a GARCH state, pretend its last stored
    row is new and run features_db.garch_tail on the 40-row write-back tail (refits
    happen exactly where the schedule says). In memory only."""
    from src import garch_state as gs
    from src.analysis_w9 import MIN_OBS, REFIT
    from src.db.features_db import garch_tail
    latest = repo.read_latest_features()
    tk = sorted(latest.index[latest["sigma_garch_1d"].notna()])
    spent = 0.0; refits = 0; n = 0
    for i in range(0, len(tk), 100):
        batch = tk[i:i + 100]
        stored = repo.read_features_tails(batch, 760)
        cache = {}
        for t in batch:
            st = stored.get(t)
            if st is None or len(st) < 41:
                continue
            if back:
                st = st.iloc[:-back]                            # pretend the data ends `back` rows earlier
            tail = st.iloc[-40:].drop(columns=gs.OUT_COLS)
            prior = st.iloc[:-1]                                # the last row is "new"
            last_i = st["garch_i"].iloc[-1]
            refits += int(pd.notna(last_i) and (int(last_i) + 1 - MIN_OBS) % REFIT == 0)
            def fetch(t=t):
                if not cache:
                    cache.update(repo.read_log_returns(batch))
                return cache[t]
            t0 = time.perf_counter(); garch_tail(tail, prior, fetch); spent += time.perf_counter() - t0; n += 1
    secs = spent                         # only the GARCH step (the stored-tail read already exists in W11)
    return {"tickers": n, "refits": refits, "total_s": round(secs, 2), "per_ticker_ms": round(secs / max(n, 1) * 1e3, 2)}


def main() -> int:
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "final"
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"tag": tag, "rss_mb": {"start": rss_mb()}}
    from src.cache import redis_cache as cache
    from src.db import repository as repo
    from src.db.engine import get_engine
    from src.tool import pricing_service as ps
    res["rss_mb"]["after_imports"] = rss_mb()

    # ---------------- L3 database first (cold-ish cache)
    with get_engine().connect() as c:
        t = repo.list_securities()["ticker"].tolist()
        required = ["close", "sigma_ewma", "r_1y", "div_yield_ttm", "vix_level", "vix_sentiment_pct", "r_1y_dev",
                    "realized_vol_252d", "beta_63d", "corr_vix_63d"]
        cond = " AND ".join(f"{k} IS NOT NULL" for k in required)
        latest_sql = (f"SELECT DISTINCT ON (ticker) * FROM features_daily WHERE {cond} "
                      f"ORDER BY ticker, date DESC")
        res["L3_latest_row_query"] = explain(c, latest_sql)
        # Week 12: the same database-side measure for the Week 11 latest-state table (like-for-like with the scan)
        res["L3_features_latest_explain"] = explain(c, "SELECT * FROM features_latest")
        res["L3_tail_read_JPM"] = explain(c, "SELECT * FROM features_daily WHERE ticker='JPM' ORDER BY date DESC LIMIT 760")
        res["L3_prices_tail_JPM"] = explain(c, "SELECT * FROM prices_daily WHERE ticker='JPM' ORDER BY date DESC LIMIT 760")
        res["L3_universe_quotes_query"] = explain(c, "SELECT DISTINCT ON (q.ticker) q.*, s.name, s.sector FROM quotes_daily q "
                                                     "JOIN securities s ON q.ticker=s.ticker ORDER BY q.ticker, q.date DESC")
        sizes = c.execute(sa.text("""SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) AS total,
                                     pg_size_pretty(pg_relation_size(relid)) AS table_only, n_live_tup
                                     FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC""")).all()
        res["L3_table_sizes"] = [dict(relname=r[0], total=r[1], table=r[2], rows=int(r[3])) for r in sizes]
        res["L3_indexes"] = [r[0] for r in c.execute(sa.text("SELECT indexname FROM pg_indexes WHERE tablename='features_daily'")).all()]
        res["L3_db_size_mb"] = round(c.execute(sa.text("select pg_database_size(current_database())")).scalar() / 2**20, 1)

    # ---------------- L2 service
    tickers = ps.quote_universe().index.tolist()
    r, _ = timed_profile(lambda: ps.quote_universe(tickers))
    res["L2_quote_universe"] = r | {"tickers": len(tickers)}
    uq = ps.quote_universe(tickers)
    r, _ = timed_profile(lambda: ps.scenario_universe(uq))
    res["L2_scenario_universe"] = r
    r, _ = timed_profile(lambda: ps.load_market_state(ticker="JPM", source="db"), n=5)
    res["L2_load_state_one_row"] = r
    # ---------------- Week 12 additions
    r, _ = timed_profile(lambda: repo.read_latest_features(), n=7)
    res["L3_features_latest_read_all"] = r | {"rows": int(len(repo.read_latest_features()))}
    st = ps.load_market_state(ticker="JPM", source="db")
    r, _ = timed_profile(lambda: ps.quote(st, cached=False), n=7)
    res["L2_quote_scalar_uncached_JPM"] = r
    from src.ml import cross_sectional as xs
    r, _ = timed_profile(lambda: xs.cross_section(), n=7)
    res["L2_ml_cross_section"] = r
    sec_ = repo.list_securities()
    sp_now = [t for t in tickers if t in set(sec_[sec_["universe"] == "sp500"]["ticker"])]
    r, _ = timed_profile(lambda: ps.quote_universe(sp_now))
    res["L2_quote_universe_sp500_part"] = r | {"tickers": len(sp_now)}

    res["rss_mb"]["after_service"] = rss_mb()

    # ---------------- L1 dashboard
    from streamlit.testing.v1 import AppTest
    def render():
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300); at.run()
        assert not at.exception, at.exception
    r, prof = timed_profile(render, n=3)
    r["top_src"] = top_funcs(prof, 10, ["src/|app/"])
    res["L1_dashboard_first_render"] = r
    res["rss_mb"]["after_dashboard"] = rss_mb()
    # cold import cost of the service layer (subprocess) and what torch/shap add
    import subprocess
    code = ("import time,sys;sys.path.insert(0,%r);t=time.perf_counter();import %s;print((time.perf_counter()-t)*1e3)")
    for mod in ("src.tool.pricing_service", "torch", "shap", "streamlit"):
        try:
            v = float(subprocess.check_output([sys.executable, "-c", code % (str(ROOT), mod)], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1])
        except Exception:
            v = None
        res.setdefault("L1_cold_import_ms", {})[mod] = round(v, 1) if v else None

    # ---------------- L4 data feed
    runs = repo.read_runs(50)
    daily = runs[(runs["job"] == "daily") & (runs["tickers"] >= 400)].sort_values("id", ascending=False)
    if len(daily):
        rr = daily.iloc[0]; tm = json.loads(rr["timings"]) if rr["timings"] else {}
        secs = (pd.Timestamp(rr["finished_at"]) - pd.Timestamp(rr["started_at"])).total_seconds()
        net = tm.get("market", 0) + tm.get("fetch+prices", 0)
        res["L4_daily_pipeline"] = {"run_id": int(rr["id"]), "tickers": int(rr["tickers"]), "total_s": round(secs, 1),
                                    "stages": {k: v for k, v in tm.items() if k != "sources"},
                                    "network_share": round(net / secs, 3) if secs else None,
                                    "features_share": round(tm.get("features", 0) / secs, 3) if secs else None}

    # ---------------- L5 space
    res["L5_space"] = {"db_size_mb": res["L3_db_size_mb"], "redis": cache.stats(),
                       "features_csv_kb": round((ROOT / "data/processed/features_daily.csv").stat().st_size / 1024, 1),
                       "rss_mb": res["rss_mb"]}
    try:
        img = subprocess.check_output(["/Applications/Docker.app/Contents/Resources/bin/docker", "images",
                                       "ms-chooser-option-app", "--format", "{{.Size}}"]).decode().strip()
        res["L5_space"]["docker_image"] = img
    except Exception:
        pass

    # Week 12: measured LAST so that its memory does not leak into the RSS readings above
    from src.db import repository as repo2
    res["L4b_garch_one_new_day_simulated"] = {"refit_day": _garch_one_day(repo2, back=0),
                                              "ordinary_day": _garch_one_day(repo2, back=1)}
    (OUT / f"profile_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("top", "top_src", "plan_head")})
                      for k, v in res.items() if k.startswith(("L1", "L2", "L3_latest", "L3_universe", "L4", "rss"))},
                     indent=1, ensure_ascii=False, default=str)[:3500])
    print("\nlatest-row plan:", *res["L3_latest_row_query"]["plan_head"], sep="\n  ")
    print("\ndashboard top src/app functions:")
    for f in res["L1_dashboard_first_render"]["top_src"][:8]:
        print(f"  {f['cumtime']:7.3f}s  {f['func']}")
    print("written", OUT / f"profile_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
