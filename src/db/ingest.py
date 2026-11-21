"""Bootstrap backfill: load the Week 1-2 file datasets into PostgreSQL and
rebuild the feature table from the DB copies. Bootstrap semantics: existing
rows are never overwritten (the live pipeline is authoritative), so re-running
this after the pipeline has pulled newer data is harmless.

    python -m src.db.ingest            (run from project root; tables must exist)

Steps
  1. securities   JPM
  2. prices_daily     <- data/raw/jpm_daily.csv
  3. index_daily      <- vix / sp500 / treasury_10y_yahoo / tbill_13w_yahoo
  4. rates_daily      <- data/raw/fred_rates_daily.csv
  5. features_daily   <- recomputed from the DB tables with src.features
                         (audit 1 in analysis/week9 checks == features_daily.csv)
  6. model_registry   <- models/final_*.joblib (sha256 pinned)
"""

import datetime as dt

import pandas as pd

from src import config, features as F
from src.db import repository as repo
from src.db.engine import get_engine

INDEX_FILES = {"^VIX": "vix_daily.csv", "^GSPC": "sp500_daily.csv",
               "^TNX": "treasury_10y_yahoo_daily.csv", "^IRX": "tbill_13w_yahoo_daily.csv"}
EQUITY_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                 "Adj Close": "adj_close", "Volume": "volume",
                 "Dividends": "dividend", "Stock Splits": "split"}
INDEX_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
RATES_RENAME = {c: c.lower() for c in ["DGS3MO", "DGS1", "DGS10", "FEDFUNDS", "T10Y2Y"]}


def _csv(name):
    return pd.read_csv(config.DATA_RAW / name, index_col="date", parse_dates=True)


def rebuild_features(ticker: str, conn=None) -> pd.DataFrame:
    """Full recompute of one ticker's features (+ stored EWMA / r_1y_dev state)."""
    from src.db.features_db import rebuild_full
    return rebuild_full(ticker)


def main() -> int:
    t0 = dt.datetime.now(dt.timezone.utc)
    counts = {}
    with get_engine().begin() as c:
        repo.upsert_security(config.TICKER, "JPMorgan Chase & Co.", "custom", conn=c)
        eq = _csv("jpm_daily.csv").rename(columns=EQUITY_RENAME)
        counts["prices"] = repo.upsert_prices(config.TICKER, eq, "yahoo", conn=c, overwrite=False)
        counts["index"] = 0
        for sym, fname in INDEX_FILES.items():
            counts["index"] += repo.upsert_index(sym, _csv(fname).rename(columns=INDEX_RENAME), conn=c, overwrite=False)
        counts["rates"] = repo.upsert_rates(_csv("fred_rates_daily.csv").rename(columns=RATES_RENAME), conn=c, overwrite=False)
    feats = rebuild_features(config.TICKER)
    counts["features"] = len(feats)
    for p in sorted((config.PROJECT_ROOT / "models").glob("final_*.joblib")):
        role = "reference" if p.name == "final_ridge_anch.joblib" else "archived"
        repo.register_model(p.stem, p, role, "Week 6 deployed bundle")
    repo.log_run("ingest", t0, "ok", "file backfill", tickers=1,
                 rows_prices=counts["prices"], rows_features=counts["features"], rows_quotes=0,
                 data_version=repo.data_version())
    for k, v in counts.items():
        print(f"  {k:10s} {v:>6} rows")
    print("table counts:", repo.table_counts(exact=True))   # exact: a fresh DB has no planner stats yet
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
