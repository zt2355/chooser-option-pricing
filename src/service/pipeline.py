"""Daily pipeline (Week 10: multi-ticker, batched, incremental).

    python -m src.service.pipeline                 all active securities
    python -m src.service.pipeline JPM AAPL        explicit tickers
    python -m src.service.pipeline --dry-run       no network: rebuild features + reprice
    python -m src.service.pipeline --full          force full feature rebuild (not incremental)

Stages (each timed; timings are stored as JSON in pipeline_runs.timings):
  market    ^VIX ^GSPC ^TNX ^IRX + FRED, incremental with a 10-day overlap
  fetch     multi-source batch download; tickers with no stored prices get
            their full history, the rest the overlap window
  prices    upsert per ticker (rows without a close are dropped upstream);
            tickers that returned nothing are marked inactive/no_price_history
  features  incremental rebuild per ticker (full on first load)
  quotes    ONE vectorised pass over the universe (quote_universe) + scenarios
  cache     invalidate every touched ticker's keys
No DATABASE_URL -> Week 7 file refresh, so the manual button never dies.
"""

import datetime as dt
import json
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

from src import config
from src.cache import redis_cache as cache

OVERLAP_DAYS = 10
INDEX_SYMBOLS = ["^VIX", "^GSPC", "^TNX", "^IRX"]
INDEX_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}


@dataclass
class RunReport:
    job: str = "daily"
    started: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    status: str = "ok"
    tickers: list = field(default_factory=list)
    rows_prices: int = 0
    rows_features: int = 0
    rows_quotes: int = 0
    rows_rates: int = 0
    rows_index: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)
    mode: str = "db"
    timings: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    def summary(self) -> str:
        if self.mode == "file":
            return f"[file mode] {self.errors[0] if self.errors else '数据文件已更新'}"
        head = (f"{'完成' if self.status == 'ok' else '部分失败'}: {len(self.tickers)} 只股票, "
                f"价格 +{self.rows_prices} 行, 特征 {self.rows_features} 行, 报价 {self.rows_quotes} 行, "
                f"{self.seconds:.1f}s " + " ".join(f"{k}={v:.1f}" for k, v in self.timings.items()))
        if self.missing:
            head += f" | 无价格 {len(self.missing)}"
        return head + ("" if not self.errors else " | " + "; ".join(self.errors)[:300])


def _since(last) -> str:
    if last is None:
        return config.START_DATE
    return (pd.Timestamp(last) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")


def _today() -> str:
    return (dt.date.today() + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive


def _file_mode_refresh(rep: RunReport) -> RunReport:
    import os
    import subprocess
    env = {**os.environ, "END_DATE": dt.date.today().isoformat()}
    for mod in ("src.collect_all", "src.preprocess"):
        res = subprocess.run([sys.executable, "-m", mod], env=env,
                             cwd=config.PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            rep.status, rep.errors = "error", [f"更新失败于 {mod}: {res.stderr[-300:]}"]
            return rep
    rep.errors = [f"数据已更新到 {dt.date.today().isoformat()}"]
    return rep


class _Timer:
    def __init__(self, rep): self.rep = rep
    def __call__(self, name):
        rep = self.rep
        class _C:
            def __enter__(s): s.t = time.perf_counter(); return s
            def __exit__(s, *a): rep.timings[name] = round(time.perf_counter() - s.t, 3)
        return _C()


def update_market_tables(dry_run: bool, rep: RunReport) -> None:
    """Week 11 (perf): the four index series and FRED are independent HTTP
    calls, so they run concurrently (ThreadPoolExecutor) instead of serially."""
    from concurrent.futures import ThreadPoolExecutor
    from src.data_collection import fred, yahoo
    from src.db import repository as repo
    if dry_run:
        return
    end = _today()

    def _index(sym):
        idx = repo.read_index(sym)
        last = idx.index.max() if len(idx) else None
        return sym, yahoo.fetch_index(sym, _since(last), end).rename(columns=INDEX_RENAME)

    def _rates():
        rates_db = repo.read_rates()
        last = rates_db.index.max() if len(rates_db) else None
        r = fred.fetch_all(config.FRED_API_KEY, _since(last), dt.date.today().isoformat())
        r.columns = [c.lower() for c in r.columns]
        return "FRED", r

    jobs = [lambda s=s: _index(s) for s in INDEX_SYMBOLS]
    if config.FRED_API_KEY:
        jobs.append(_rates)
    else:
        rep.errors.append("FRED_API_KEY missing: rates not refreshed")
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = [ex.submit(j) for j in jobs]
        for f in futures:
            try:
                name, df = f.result()
                if name == "FRED":
                    rep.rows_rates += repo.upsert_rates(df)
                else:
                    rep.rows_index += repo.upsert_index(name, df)
            except Exception as e:
                rep.errors.append(f"market: {e}"[:120])


def fetch_and_store_prices(tickers: list[str], rep: RunReport) -> list[str]:
    """Batch download grouped by required window; returns tickers with prices."""
    from src.data_collection.multi_source import fetch_prices
    from src.db import repository as repo
    last = repo.last_price_dates()
    new = [t for t in tickers if t not in last]
    old = [t for t in tickers if t in last]
    end = _today()
    frames, source = {}, {}
    if new:
        f, s_, r = fetch_prices(new, config.START_DATE, end); frames.update(f); source.update(s_)
        rep.sources["full"] = r.by_source; rep.missing += r.missing
    if old:
        # group by required window so a stale ticker does not widen everyone's re-pull
        groups: dict[str, list[str]] = {}
        for t in old:
            groups.setdefault(_since(last[t]), []).append(t)
        agg = {"yfinance_batch": 0, "yfinance_single": 0, "alphavantage_unadj": 0}
        for since, ts in groups.items():
            f, s_, r = fetch_prices(ts, since, end); frames.update(f); source.update(s_)
            for k in agg:
                agg[k] += r.by_source.get(k, 0)
            rep.missing += r.missing
        rep.sources["incremental"] = {**agg, "windows": len(groups)}
    with repo.get_engine().begin() as c:
        for t, df in frames.items():
            rep.rows_prices += repo.upsert_prices(t, df, source[t], conn=c)
    for t in rep.missing:
        repo.set_security_note(t, False, "no_price_history")
    return [t for t in tickers if t in frames]


def run_daily(tickers: list[str] | None = None, dry_run: bool = False,
              incremental: bool = True) -> RunReport:
    rep = RunReport(job="daily" if not dry_run else "reprice")
    T = _Timer(rep)
    t0 = time.perf_counter()
    from src.db.engine import database_url, db_available
    if not (database_url() and db_available()):
        rep.mode = "file"
        return _file_mode_refresh(rep)
    from src.db import repository as repo
    from src.db.features_db import _market_inputs, rebuild_features
    from src.tool import pricing_service as ps
    if tickers is None:
        tickers = repo.list_securities()["ticker"].tolist() or [config.TICKER]

    with T("market"):
        update_market_tables(dry_run, rep)
    with T("fetch+prices"):
        if not dry_run:
            tickers = fetch_and_store_prices(tickers, rep)
    with T("features"):
        market = _market_inputs()
        if incremental:
            from src.db.features_db import rebuild_many
            try:
                errs: dict = {}
                counts = rebuild_many(tickers, market=market, errors=errs)
                rep.rows_features += sum(counts.values()); rep.tickers += [t for t in tickers if t in counts]
                if errs:                       # Week 12: per-ticker isolation; the others are still priced
                    rep.status = "partial"
                    rep.errors += [f"{t}: {m}" for t, m in list(errs.items())[:10]]
                    if len(errs) > 10:
                        rep.errors.append(f"... {len(errs) - 10} more feature errors")
            except Exception as e:
                rep.errors.append(f"features batch: {e}"[:160]); rep.status = "partial"
        else:
            for t in tickers:
                try:
                    rep.rows_features += len(rebuild_features(t, incremental=False, market=market))
                    rep.tickers.append(t)
                except Exception as e:
                    rep.errors.append(f"{t}: {e}"[:160]); rep.status = "partial"
    with T("quotes"):
        uq = ps.quote_universe(rep.tickers)
        if len(uq):
            qrows = [{"ticker": t, "date": pd.Timestamp(r["date"]).date(), "spot": r["S"], "strike": r["K"],
                      "t1": r["t1"], "t2": r["t2"], "sigma_ewma": r["sigma_ewma"],
                      "anchor": r["anchor"], "sigma_anchor": r["sigma_anchor"],
                      "sigma_garch_1d": None if pd.isna(r["sigma_garch_1d"]) else r["sigma_garch_1d"],
                      "sigma_ml": None if pd.isna(r["sigma_ml"]) else r["sigma_ml"],
                      "ml_ratio": None if pd.isna(r["ml_ratio"]) else r["ml_ratio"], "r": r["r"], "q": r["q"], "regime": r["regime"],
                      "primary_price": r["primary_price"], "band_lo": r["band_lo"], "band_hi": r["band_hi"],
                      "ml_reference_price": None if pd.isna(r["ml_reference_price"]) else r["ml_reference_price"], "call_leg": r["call_leg"],
                      "put_leg": r["put_leg"], "model_version": r["model_version"]}
                     for t, r in uq.iterrows()]
            sc = ps.scenario_universe(uq)
            srows = [{"ticker": r["ticker"], "date": pd.Timestamp(r["date"]).date(), "scenario": r["scenario"],
                      "price": float(r["price"]), "change_usd": float(r["change_usd"]),
                      "change_pct": float(r["change_pct"])} for _, r in sc.iterrows()]
            with repo.get_engine().begin() as c:
                rep.rows_quotes += repo.upsert_quotes_batch(qrows, conn=c)
                repo.upsert_scenarios_batch(srows, conn=c)
    with T("cache"):
        for t in rep.tickers:
            cache.invalidate(t)
    rep.seconds = time.perf_counter() - t0
    if rep.errors and rep.status == "ok":
        rep.status = "partial"          # e.g. a market source failed: quotes used the last stored rates
    msg = "; ".join(rep.errors)
    if rep.missing:
        msg = f"no_price_history: {','.join(rep.missing[:30])}{'...' if len(rep.missing) > 30 else ''} | " + msg
    repo.log_run(rep.job, rep.started, rep.status, msg, timings={**rep.timings, "sources": rep.sources},
                 tickers=len(rep.tickers), rows_prices=rep.rows_prices,
                 rows_features=rep.rows_features, rows_quotes=rep.rows_quotes,
                 data_version=repo.data_version())
    return rep


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    r = run_daily(args or None, dry_run="--dry-run" in sys.argv, incremental="--full" not in sys.argv)
    print(r.summary())
    raise SystemExit(0 if r.status == "ok" else 1)
