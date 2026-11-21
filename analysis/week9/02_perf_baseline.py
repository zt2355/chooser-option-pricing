"""Performance baseline (Week 9) — 佟铸城 (Zhucheng Tong)

Measures time and memory of every stage between the dashboard and the data
feed, BEFORE the Week 12 optimisation pass. Per the mentor, only RELATIVE
improvement will be judged, so this file is the "before" that Week 12 is
compared against; the hardware is recorded alongside the numbers.

Stages (each repeated N times, median reported; peak memory via tracemalloc)
  S1  cold import of the service layer            (subprocess, wall clock)
  S2  feature history load: CSV file vs PostgreSQL
  S3  market-state derivation (EWMA + r_1y_dev) from the loaded history
  S4  quote — Week 7 code path (reference model re-read from disk every call)
  S5  quote — Week 9 code path (model loaded once per process)
  S6  quote — Redis hit
  S7  scenario table: compute vs Redis hit
  S8  daily ATM panel pricing, 1310 days: scalar loop vs vectorised closed form
  S9  reprice pipeline (--dry-run: DB -> features -> quote -> DB, no network)
  S10 real daily pipeline with network, from pipeline_runs (latest 'daily' row)
  S11 Streamlit first render (streamlit.testing AppTest, headless)

Usage (project root, services running):  python analysis/week9/02_perf_baseline.py
Writes docs/week9/perf_baseline.json and perf_baseline.csv.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "week9"
N = 5


def timed(fn, n=N, memory=False):
    """median wall time in ms (and peak tracemalloc MB if memory=True)."""
    times, peaks = [], []
    for _ in range(n):
        if memory:
            tracemalloc.start()
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
        if memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / 2**20)
    row = {"median_ms": round(statistics.median(times), 3),
           "min_ms": round(min(times), 3), "max_ms": round(max(times), 3), "n": n}
    if memory:
        row["peak_mb"] = round(max(peaks), 2)
    return row


def hardware() -> dict:
    mem = None
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()) / 2**30
    except Exception:
        pass
    cpu = None
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "cores": os.cpu_count(), "memory_gb": round(mem, 1) if mem else None,
            "python": platform.python_version()}


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    print("hardware:", hardware())

    # S1 cold import (fresh interpreter each time)
    code = ("import time,sys;sys.path.insert(0,%r);t=time.perf_counter();"
            "from src.tool import pricing_service;print((time.perf_counter()-t)*1e3)" % str(ROOT))
    times = [float(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT).decode().strip().splitlines()[-1])
             for _ in range(N)]
    results["S1_cold_import_service"] = {"median_ms": round(statistics.median(times), 1),
                                         "min_ms": round(min(times), 1), "max_ms": round(max(times), 1), "n": N}
    print("S1", results["S1_cold_import_service"])

    from src.cache import redis_cache as cache
    from src.tool import pricing_service as ps
    from src.db import repository as repo
    from src.db.engine import db_available
    assert db_available(), "PostgreSQL not reachable — start services first"

    # S2 history load
    results["S2_history_load_file"] = timed(lambda: ps.load_history(source="file"), memory=True)
    results["S2_history_load_db"] = timed(lambda: ps.load_history(source="db"), memory=True)
    print("S2", results["S2_history_load_file"], results["S2_history_load_db"])

    # S3 derive + row pick
    hist, _ = ps.load_history(source="db")
    results["S3_derive_state"] = timed(lambda: ps.derive(hist), memory=True)
    state = ps.load_market_state(source="db")

    # S4 quote, Week 7 path (model re-read each call)
    def q_w7():
        ps._reference_bundle.cache_clear()
        return ps.quote(state, cached=False)
    results["S4_quote_w7_path_model_reload"] = timed(q_w7, memory=True)
    # S5 quote, Week 9 path (model resident)
    ps.quote(state, cached=False)  # warm
    results["S5_quote_w9_path_model_resident"] = timed(lambda: ps.quote(state, cached=False), memory=True)
    # S6 redis hit
    cache.invalidate(state.ticker); ps.quote(state)  # populate
    results["S6_quote_redis_hit"] = timed(lambda: ps.quote(state))
    print("S4-6", results["S4_quote_w7_path_model_reload"]["median_ms"],
          results["S5_quote_w9_path_model_resident"]["median_ms"], results["S6_quote_redis_hit"]["median_ms"])

    # S7 scenario table
    results["S7_scenario_compute"] = timed(lambda: ps.scenario_table(state, cached=False))
    ps.scenario_table(state)
    results["S7_scenario_redis_hit"] = timed(lambda: ps.scenario_table(state))

    # S8 panel pricing 1310 days
    from src.analysis_w4 import build_panel, T1, T2
    from src.models.chooser import chooser_closed_form
    panel = build_panel()
    S, r, qd, sig = (panel["jpm_close"].to_numpy(), panel["r_1y"].to_numpy(),
                     panel["div_yield_ttm"].to_numpy(), panel["sigma_ewma"].to_numpy())

    def scalar_loop():
        return [ps.price_chooser(S[i], S[i], r[i], qd[i], sig[i], T1, T2) for i in range(len(S))]

    def vectorised():
        return chooser_closed_form(S0=S, K=S, r=r, q=qd, sigma=sig, t1=T1, T2=T2)
    results["S8_panel_pricing_scalar_loop"] = timed(scalar_loop, n=3, memory=True)
    results["S8_panel_pricing_vectorised"] = timed(vectorised, memory=True)
    results["S8_panel_pricing_scalar_loop"]["rows"] = int(len(S))
    a, b = np.array(scalar_loop()), np.asarray(vectorised())
    assert np.allclose(a, b, atol=1e-9)
    print("S8", results["S8_panel_pricing_scalar_loop"]["median_ms"], results["S8_panel_pricing_vectorised"]["median_ms"])

    # S9 reprice pipeline (no network)
    from src.service.pipeline import run_daily
    results["S9_reprice_pipeline_dry_run"] = timed(lambda: run_daily(["JPM"], dry_run=True), n=3, memory=True)
    print("S9", results["S9_reprice_pipeline_dry_run"])

    # S10 latest real daily run from the log
    runs = repo.read_runs(50)
    daily = runs[(runs["job"] == "daily") & (runs["status"] == "ok")]
    if len(daily):
        row = daily.iloc[0]
        secs = (pd.Timestamp(row["finished_at"]) - pd.Timestamp(row["started_at"])).total_seconds()
        results["S10_daily_pipeline_with_network"] = {"median_ms": round(secs * 1e3, 1), "n": 1,
                                                       "rows_prices": int(row["rows_prices"]),
                                                       "run_id": int(row["id"])}
        print("S10", results["S10_daily_pipeline_with_network"])

    # S11 Streamlit first render (headless)
    try:
        from streamlit.testing.v1 import AppTest
        def first_render():
            at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
            at.run()
            assert not at.exception, at.exception
        results["S11_streamlit_first_render"] = timed(first_render, n=3, memory=True)
        print("S11", results["S11_streamlit_first_render"])
    except Exception as e:  # keep the baseline even if AppTest is unavailable
        results["S11_streamlit_first_render"] = {"error": str(e)[:200]}

    # process footprint + DB / cache sizes
    counts = repo.table_counts()
    results["footprint"] = {"process_rss_mb_end": round(rss_mb(), 1),
                            "db_rows": counts, "cache": cache.stats(),
                            "features_csv_kb": round((ROOT / "data/processed/features_daily.csv").stat().st_size / 1024, 1)}
    try:
        import sqlalchemy as sa
        from src.db.engine import get_engine
        with get_engine().connect() as c:
            results["footprint"]["db_size_mb"] = round(
                c.execute(sa.text("select pg_database_size(current_database())")).scalar() / 2**20, 2)
    except Exception:
        pass

    summary = {"hardware": hardware(), "n_repeats": N, "results": results,
               "measured_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (OUT / "perf_baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    rows = [{"stage": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}}
            for k, v in results.items() if k != "footprint"]
    pd.DataFrame(rows).to_csv(OUT / "perf_baseline.csv", index=False)
    print("\nwritten:", OUT / "perf_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
