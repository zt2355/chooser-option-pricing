"""Optimisation experiments, before vs after (Week 10) — 佟铸城 (Zhucheng Tong)

Each experiment re-measures the Week 9 "before" path in the same process and
hardware as the "after" path, so the ratio is hardware-independent; absolute
numbers are reported alongside for the record (mentor: relative improvement
is what counts).

  E1  feature rebuild for one ticker: full recompute + full write (Week 9)
      vs incremental tail (Week 10)
  E2  market-state load for a quote: full history read + EWMA derivation
      (Week 9) vs one-row read of stored state (Week 10)
  E3  pricing the whole universe: per-ticker scalar path
      (load_market_state + quote, Week 9 style) vs quote_universe (vectorised)
  E4  scenario tables for the universe: scalar loop vs scenario_universe
  E5  daily pipeline, universe scale: extrapolated Week 9 per-ticker cost
      (S10 = 4.5 s/ticker, serial) vs measured Week 10 batch run
      (latest 'daily' row in pipeline_runs, stage timings)

Usage (services up, universe loaded):  python analysis/week10/02_perf_after.py
Writes docs/week10/perf_after.json and perf_before_after.csv
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "week10"
N = 5


def med(fn, n=N):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 3)


def main() -> int:
    from src.db import repository as repo
    from src.db.features_db import rebuild_full, rebuild_incremental
    from src.tool import pricing_service as ps
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.load(open(ROOT / "docs/week9/perf_baseline.json"))
    res = {}

    # E1
    before = med(lambda: rebuild_full("JPM"), n=3)
    after = med(lambda: rebuild_incremental("JPM"), n=5)
    res["E1_feature_rebuild_jpm"] = {"before_ms": before, "after_ms": after, "speedup": round(before / after, 1),
                                     "before_rows": 2190, "after_rows": 40}
    print("E1", res["E1_feature_rebuild_jpm"])

    # E2
    def before_state():
        hist, _ = ps.load_history("JPM", "db"); df = ps.derive(hist)
        return df.dropna(subset=ps.REQUIRED + ps.ML_FEATURES).iloc[-1]
    before = med(before_state); after = med(lambda: ps.load_market_state(ticker="JPM", source="db"))
    res["E2_market_state_load"] = {"before_ms": before, "after_ms": after, "speedup": round(before / after, 1)}
    print("E2", res["E2_market_state_load"])

    # E3
    tickers = ps.quote_universe().index.tolist()     # tickers with a complete latest row
    sample = tickers[:50]
    def scalar_loop(ts):
        for t in ts:
            ps.quote(ps.load_market_state(ticker=t, source="db"), cached=False)
    t = time.perf_counter(); scalar_loop(sample); per_ticker_ms = (time.perf_counter() - t) * 1e3 / len(sample)
    before_all = per_ticker_ms * len(tickers)
    after_all = med(lambda: ps.quote_universe(tickers), n=3)
    res["E3_universe_quotes"] = {"tickers": len(tickers), "before_ms_extrapolated": round(before_all, 1),
                                 "before_per_ticker_ms": round(per_ticker_ms, 2), "after_ms": after_all,
                                 "speedup": round(before_all / after_all, 1)}
    print("E3", res["E3_universe_quotes"])

    # E4
    uq = ps.quote_universe(tickers)
    states = [ps.load_market_state(ticker=t, source="db") for t in sample]
    t = time.perf_counter()
    for s in states:
        ps.scenario_table(s, cached=False)
    per = (time.perf_counter() - t) * 1e3 / len(sample)
    after_sc = med(lambda: ps.scenario_universe(uq), n=3)
    res["E4_universe_scenarios"] = {"before_ms_extrapolated": round(per * len(tickers), 1), "after_ms": after_sc,
                                    "speedup": round(per * len(tickers) / after_sc, 1)}
    print("E4", res["E4_universe_scenarios"])

    # E5
    runs = repo.read_runs(50)
    daily = runs[(runs["job"] == "daily") & (runs["tickers"] >= 400)].sort_values("id", ascending=False)
    if len(daily):
        r = daily.iloc[0]
        secs = (pd.Timestamp(r["finished_at"]) - pd.Timestamp(r["started_at"])).total_seconds()
        w9_per_ticker = base["results"]["S10_daily_pipeline_with_network"]["median_ms"] / 1e3
        res["E5_daily_pipeline_universe"] = {"tickers": int(r["tickers"]), "measured_s": round(secs, 1),
                                             "per_ticker_s": round(secs / int(r["tickers"]), 3),
                                             "w9_serial_extrapolated_s": round(w9_per_ticker * int(r["tickers"]), 1),
                                             "speedup": round(w9_per_ticker * int(r["tickers"]) / secs, 1),
                                             "stage_timings": json.loads(r["timings"]) if r["timings"] else None,
                                             "run_id": int(r["id"]), "rows_prices": int(r["rows_prices"])}
        print("E5", {k: v for k, v in res["E5_daily_pipeline_universe"].items() if k != "stage_timings"})
        print("   stages:", res["E5_daily_pipeline_universe"]["stage_timings"])

    rows = [{"experiment": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}} for k, v in res.items()]
    pd.DataFrame(rows).to_csv(OUT / "perf_before_after.csv", index=False)
    (OUT / "perf_after.json").write_text(json.dumps({"hardware": base["hardware"], "results": res,
                                                     "measured_at": pd.Timestamp.now().isoformat(timespec="seconds")},
                                                    indent=2, ensure_ascii=False))
    print("written", OUT / "perf_after.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
