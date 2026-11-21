"""Read / write helpers over the Week 9 schema.

Writes are idempotent upserts (INSERT ... ON CONFLICT DO UPDATE) so the
daily job can be re-run safely. Reads return pandas frames indexed by date,
in the generic column names of `src.features` (use
`features.to_file_names` when a Week 1-8 consumer needs jpm_* names).
"""

import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db import schema
from src.db.engine import get_engine

CHUNK = 5000


# ------------------------------------------------------------------ writes

def _upsert(table: sa.Table, rows: list[dict], conn, overwrite: bool = True) -> int:
    """overwrite=True: INSERT ... ON CONFLICT DO UPDATE (live pipeline is authoritative).
    overwrite=False: ON CONFLICT DO NOTHING (file backfill only bootstraps, never
    regresses rows the pipeline already wrote)."""
    if not rows:
        return 0
    pk = [c.name for c in table.primary_key.columns]
    # Week 12: on conflict, update only the columns this write actually carries. Updating every
    # column would NULL out anything the caller did not supply (e.g. the GARCH state columns).
    present = set().union(*(r.keys() for r in rows))
    n = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        stmt = pg_insert(table).values(chunk)
        update_cols = {c.name: stmt.excluded[c.name]
                       for c in table.columns if c.name not in pk and c.name in present
                       and c.name not in ("added_at", "registered_at", "computed_at")}
        if update_cols and overwrite:
            stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=pk)
        conn.execute(stmt)
        n += len(chunk)
    return n


def _frame_rows(df: pd.DataFrame, extra: dict | None = None) -> list[dict]:
    """DataFrame (date index) -> list of dicts with NaN -> None."""
    out = df.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.astype(object).where(pd.notna(out), None)
    rows = out.to_dict("records")
    if extra:
        for r in rows:
            r.update(extra)
    return rows


def upsert_security(ticker: str, name: str | None = None,
                    universe: str = "custom", conn=None) -> None:
    def _do(c):
        _upsert(schema.securities, [{"ticker": ticker, "name": name,
                                     "universe": universe}], c)
    if conn is not None:
        _do(conn)
    else:
        with get_engine().begin() as c:
            _do(c)


def upsert_securities(rows: list[dict], conn=None) -> int:
    """Bulk upsert of securities (Week 10 universe load)."""
    if conn is not None:
        return _upsert(schema.securities, rows, conn)
    with get_engine().begin() as c:
        return _upsert(schema.securities, rows, c)


def set_security_note(ticker: str, active: bool, note: str | None) -> None:
    t = schema.securities
    with get_engine().begin() as c:
        c.execute(sa.update(t).where(t.c.ticker == ticker).values(active=active, note=note))


def replace_membership(index: str, mem: pd.DataFrame) -> int:
    t = schema.index_membership
    rows = [{"index": index, "ticker": r["ticker"], "start": pd.Timestamp(r["start"]).date(),
             "end": None if pd.isna(r["end"]) else pd.Timestamp(r["end"]).date(),
             "start_known": bool(r["start_known"])} for _, r in mem.iterrows()]
    with get_engine().begin() as c:
        c.execute(sa.delete(t).where(t.c.index == index))
        return _upsert(t, rows, c)


def read_membership(index: str = "sp500") -> pd.DataFrame:
    t = schema.index_membership
    df = _read(sa.select(t).where(t.c.index == index))
    df["start"] = pd.to_datetime(df["start"]); df["end"] = pd.to_datetime(df["end"])
    return df


def upsert_prices(ticker: str, df: pd.DataFrame, source: str, conn=None,
                  overwrite: bool = True) -> int:
    """df columns: open high low close adj_close volume dividend split.
    Rows without a close price (partial in-progress sessions from the provider)
    are dropped: they would poison every rolling-window feature downstream."""
    cols = ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split"]
    df = df[df["close"].notna() & df["adj_close"].notna()]
    rows = _frame_rows(df[cols], {"ticker": ticker, "source": source})
    for r in rows:
        r["volume"] = int(r["volume"]) if r["volume"] is not None else None
    if conn is not None:
        return _upsert(schema.prices_daily, rows, conn, overwrite)
    with get_engine().begin() as c:
        return _upsert(schema.prices_daily, rows, c, overwrite)


def upsert_index(symbol: str, df: pd.DataFrame, conn=None, overwrite: bool = True) -> int:
    df = df[df["close"].notna()]
    rows = _frame_rows(df[["open", "high", "low", "close"]], {"symbol": symbol})
    if conn is not None:
        return _upsert(schema.index_daily, rows, conn, overwrite)
    with get_engine().begin() as c:
        return _upsert(schema.index_daily, rows, c, overwrite)


def upsert_rates(df: pd.DataFrame, conn=None, overwrite: bool = True) -> int:
    rows = _frame_rows(df[["dgs3mo", "dgs1", "dgs10", "fedfunds", "t10y2y"]])
    if conn is not None:
        return _upsert(schema.rates_daily, rows, conn, overwrite)
    with get_engine().begin() as c:
        return _upsert(schema.rates_daily, rows, c, overwrite)


# every stored feature-state column (Week 10 EWMA/r_1y_dev, Week 12 GARCH anchor + state)
STORED_COLS = ["close", "adj_close", "vix_close"] + schema.FEATURE_COLUMNS + ["sigma_ewma", "r_1y_dev"] + schema.GARCH_COLUMNS


def upsert_features(ticker: str, df: pd.DataFrame, conn=None) -> int:
    cols = STORED_COLS
    cols = [c for c in cols if c in df.columns]
    rows = _frame_rows(df[cols], {"ticker": ticker})
    if conn is not None:
        return _upsert(schema.features_daily, rows, conn)
    with get_engine().begin() as c:
        return _upsert(schema.features_daily, rows, c)


def upsert_quotes_batch(rows: list[dict], conn=None) -> int:
    """Week 10: one statement for the whole universe."""
    if conn is not None:
        return _upsert(schema.quotes_daily, rows, conn)
    with get_engine().begin() as c:
        return _upsert(schema.quotes_daily, rows, c)


def upsert_scenarios_batch(rows: list[dict], conn=None) -> int:
    if conn is not None:
        return _upsert(schema.scenarios_daily, rows, conn)
    with get_engine().begin() as c:
        return _upsert(schema.scenarios_daily, rows, c)


def upsert_quote(ticker: str, q: dict, t1: float, T2: float,
                 model_version: str, conn=None) -> int:
    row = {"ticker": ticker, "date": pd.Timestamp(q["as_of"]).date(),
           "spot": q["S"], "strike": q["K"], "t1": t1, "t2": T2,
           "sigma_ewma": q["sigma_ewma"], "sigma_ml": q["sigma_ml"],
           "anchor": q.get("anchor"), "sigma_anchor": q.get("sigma_anchor"), "sigma_garch_1d": q.get("sigma_garch_1d"),
           "ml_ratio": q["ml_ratio"], "r": q["r"], "q": q["q"],
           "regime": q["regime"], "primary_price": q["primary_price"],
           "band_lo": q["primary_band"][0], "band_hi": q["primary_band"][1],
           "ml_reference_price": q["ml_reference_price"],
           "call_leg": q["call_leg"], "put_leg": q["put_leg"],
           "model_version": model_version}
    if conn is not None:
        return _upsert(schema.quotes_daily, [row], conn)
    with get_engine().begin() as c:
        return _upsert(schema.quotes_daily, [row], c)


def upsert_scenarios(ticker: str, as_of, table: pd.DataFrame, conn=None) -> int:
    d = pd.Timestamp(as_of).date()
    rows = [{"ticker": ticker, "date": d, "scenario": r["情景"],
             "price": float(r["价格"]), "change_usd": float(r["变化($)"]),
             "change_pct": float(r["变化(%)"])} for _, r in table.iterrows()]
    if conn is not None:
        return _upsert(schema.scenarios_daily, rows, conn)
    with get_engine().begin() as c:
        return _upsert(schema.scenarios_daily, rows, c)


def register_model(name: str, path: Path, role: str, notes: str = "", conn=None) -> str:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    row = {"name": name, "path": str(path), "sha256": sha, "role": role, "notes": notes}
    if conn is not None:
        _upsert(schema.model_registry, [row], conn)
    else:
        with get_engine().begin() as c:
            _upsert(schema.model_registry, [row], c)
    return sha


def log_run(job: str, started: dt.datetime, status: str, message: str = "",
            timings: dict | None = None, **counts) -> int:
    import json
    row = {"job": job, "started_at": started,
           "finished_at": dt.datetime.now(dt.timezone.utc), "status": status,
           "message": message[:4000], "timings": json.dumps(timings) if timings else None, **counts}
    with get_engine().begin() as c:
        res = c.execute(sa.insert(schema.pipeline_runs).values(row))
        return int(res.inserted_primary_key[0])


# ------------------------------------------------------------------ reads

def _read(stmt, conn=None) -> pd.DataFrame:
    if conn is not None:
        df = pd.read_sql(stmt, conn)
    else:
        with get_engine().connect() as c:
            df = pd.read_sql(stmt, c)
    df.columns = [str(c) for c in df.columns]   # SQLAlchemy quoted_name -> str
    return df


def _dated(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def list_securities(active_only: bool = True) -> pd.DataFrame:
    t = schema.securities
    stmt = sa.select(t).where(t.c.active.is_(True)) if active_only else sa.select(t)
    return _read(stmt)


def read_prices(ticker: str, start=None, end=None) -> pd.DataFrame:
    t = schema.prices_daily
    stmt = sa.select(t).where(t.c.ticker == ticker)
    if start is not None:
        stmt = stmt.where(t.c.date >= pd.Timestamp(start).date())
    if end is not None:
        stmt = stmt.where(t.c.date <= pd.Timestamp(end).date())
    return _dated(_read(stmt.order_by(t.c.date))).drop(columns=["ticker"])


def read_index(symbol: str) -> pd.DataFrame:
    t = schema.index_daily
    return _dated(_read(sa.select(t).where(t.c.symbol == symbol).order_by(t.c.date))).drop(columns=["symbol"])


def read_rates() -> pd.DataFrame:
    t = schema.rates_daily
    return _dated(_read(sa.select(t).order_by(t.c.date)))


def read_features(ticker: str, end=None) -> pd.DataFrame:
    t = schema.features_daily
    stmt = sa.select(t).where(t.c.ticker == ticker)
    if end is not None:
        stmt = stmt.where(t.c.date <= pd.Timestamp(end).date())
    return _dated(_read(stmt.order_by(t.c.date))).drop(columns=["ticker"])


def last_price_dates() -> dict[str, pd.Timestamp]:
    """{ticker: max(date)} for every ticker with prices (one query)."""
    t = schema.prices_daily
    with get_engine().connect() as c:
        rows = c.execute(sa.select(t.c.ticker, sa.func.max(t.c.date)).group_by(t.c.ticker)).all()
    return {r[0]: pd.Timestamp(r[1]) for r in rows}


def read_prices_tail(ticker: str, n: int) -> pd.DataFrame:
    """Last n rows of prices (for the incremental feature rebuild)."""
    t = schema.prices_daily
    sub = sa.select(t).where(t.c.ticker == ticker).order_by(t.c.date.desc()).limit(n).subquery()
    df = _read(sa.select(sub).order_by(sub.c.date))
    return _dated(df).drop(columns=["ticker"])


def read_features_tail(ticker: str, n: int) -> pd.DataFrame:
    t = schema.features_daily
    sub = sa.select(t).where(t.c.ticker == ticker).order_by(t.c.date.desc()).limit(n).subquery()
    df = _read(sa.select(sub).order_by(sub.c.date))
    return _dated(df).drop(columns=["ticker"])


LATEST_REQUIRED = ["close", "sigma_ewma", "r_1y", "div_yield_ttm", "vix_level", "vix_sentiment_pct",
                   "r_1y_dev", "log_return", "realized_vol_21d", "realized_vol_63d", "realized_vol_252d",
                   "div_growth_yoy", "curve_slope", "rate_momentum_21d", "spx_log_return", "beta_63d",
                   "corr_vix_63d", "vix_momentum_5d", "fear_regime"]


def read_latest_features_scan(tickers: list[str] | None = None, as_of=None) -> pd.DataFrame:
    """Week 9/10 path: latest COMPLETE feature row per ticker via DISTINCT ON over
    features_daily (kept as the as-of path and as the audit reference)."""
    t = schema.features_daily
    cond = sa.and_(*[t.c[c].isnot(None) for c in LATEST_REQUIRED])
    if tickers is not None:
        cond = sa.and_(cond, t.c.ticker.in_(tickers))
    if as_of is not None:
        cond = sa.and_(cond, t.c.date <= pd.Timestamp(as_of).date())
    stmt = sa.select(t).where(cond).distinct(t.c.ticker).order_by(t.c.ticker, t.c.date.desc())
    df = _read(stmt)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("ticker")


def read_latest_features(tickers: list[str] | None = None, as_of=None) -> pd.DataFrame:
    """Week 11 path: O(tickers) read from features_latest (maintained on rebuild).
    An explicit as-of date still goes through the history scan."""
    if as_of is not None:
        return read_latest_features_scan(tickers, as_of)
    t = schema.features_latest
    stmt = sa.select(t)
    if tickers is not None:
        stmt = stmt.where(t.c.ticker.in_(tickers))
    df = _read(stmt.order_by(t.c.ticker))
    if not len(df):                                   # not yet materialised -> fall back
        return read_latest_features_scan(tickers, None)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("ticker")


def upsert_features_many(tails: dict[str, pd.DataFrame], conn) -> int:
    """Week 11 (perf): all tickers' tail rows in ONE multi-row upsert per chunk."""
    cols = STORED_COLS
    rows = []
    for t, df in tails.items():
        rows += _frame_rows(df[[c for c in cols if c in df.columns]], {"ticker": t})
    return _upsert(schema.features_daily, rows, conn)


def upsert_latest_many(tails: dict[str, pd.DataFrame], conn) -> int:
    cols = STORED_COLS
    rows, drop = [], []
    for t, df in tails.items():
        ok = df.dropna(subset=[c for c in LATEST_REQUIRED if c in df.columns])
        if len(ok):
            rows += _frame_rows(ok.iloc[[-1]][[c for c in cols if c in ok.columns]], {"ticker": t})
        else:
            drop.append(t)
    if drop:
        conn.execute(sa.delete(schema.features_latest).where(schema.features_latest.c.ticker.in_(drop)))
    return _upsert(schema.features_latest, rows, conn)


def upsert_latest(ticker: str, feats: pd.DataFrame, conn=None) -> int:
    """Store the ticker's latest COMPLETE row (or delete the entry if none)."""
    cols = STORED_COLS
    ok = feats.dropna(subset=[c for c in LATEST_REQUIRED if c in feats.columns])
    def _do(c):
        if not len(ok):
            c.execute(sa.delete(schema.features_latest).where(schema.features_latest.c.ticker == ticker))
            return 0
        row = ok.iloc[[-1]][[c_ for c_ in cols if c_ in ok.columns]]
        return _upsert(schema.features_latest, _frame_rows(row, {"ticker": ticker}), c)
    if conn is not None:
        return _do(conn)
    with get_engine().begin() as c:
        return _do(c)


def read_log_returns(tickers: list[str]) -> dict[str, pd.Series]:
    """Week 12: full stored log-return history of several tickers in one query
    (a GARCH refit needs every return up to its day)."""
    t = schema.features_daily
    df = _read(sa.select(t.c.ticker, t.c.date, t.c.log_return).where(t.c.ticker.in_(tickers))
               .order_by(t.c.ticker, t.c.date))
    df["date"] = pd.to_datetime(df["date"])
    return {k: g.set_index("date")["log_return"] for k, g in df.groupby("ticker")}


def read_features_tails(tickers: list[str], n: int) -> dict[str, pd.DataFrame]:
    """Week 11: last n feature rows for MANY tickers in one window-function query."""
    t = schema.features_daily
    rn = sa.func.row_number().over(partition_by=t.c.ticker, order_by=t.c.date.desc()).label("rn")
    sub = sa.select(t, rn).where(t.c.ticker.in_(tickers)).subquery()
    df = _read(sa.select(sub).where(sub.c.rn <= n).order_by(sub.c.ticker, sub.c.date))
    df["date"] = pd.to_datetime(df["date"])
    return {k: g.drop(columns=["ticker", "rn"]).set_index("date") for k, g in df.groupby("ticker")}


def read_prices_tails(tickers: list[str], n: int) -> dict[str, pd.DataFrame]:
    t = schema.prices_daily
    rn = sa.func.row_number().over(partition_by=t.c.ticker, order_by=t.c.date.desc()).label("rn")
    sub = sa.select(t, rn).where(t.c.ticker.in_(tickers)).subquery()
    df = _read(sa.select(sub).where(sub.c.rn <= n).order_by(sub.c.ticker, sub.c.date))
    df["date"] = pd.to_datetime(df["date"])
    return {k: g.drop(columns=["ticker", "rn"]).set_index("date") for k, g in df.groupby("ticker")}


def last_price_date(ticker: str):
    t = schema.prices_daily
    with get_engine().connect() as c:
        d = c.execute(sa.select(sa.func.max(t.c.date)).where(t.c.ticker == ticker)).scalar()
    return None if d is None else pd.Timestamp(d)


def read_universe_quotes(as_of=None) -> pd.DataFrame:
    """Latest quote per ticker (DISTINCT ON), joined with securities names/sectors."""
    q, sec = schema.quotes_daily, schema.securities
    stmt = sa.select(q, sec.c.name, sec.c.sector)
    if as_of is not None:
        stmt = stmt.where(q.c.date <= pd.Timestamp(as_of).date())
    stmt = stmt.select_from(q.join(sec, q.c.ticker == sec.c.ticker)).distinct(q.c.ticker).order_by(q.c.ticker, q.c.date.desc())
    df = _read(stmt)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("ticker")


def read_quotes(ticker: str, limit: int | None = None) -> pd.DataFrame:
    t = schema.quotes_daily
    stmt = sa.select(t).where(t.c.ticker == ticker).order_by(t.c.date.desc())
    if limit:
        stmt = stmt.limit(limit)
    df = _read(stmt)
    return _dated(df).drop(columns=["ticker"]) if len(df) else df


def read_runs(limit: int = 20) -> pd.DataFrame:
    t = schema.pipeline_runs
    return _read(sa.select(t).order_by(t.c.id.desc()).limit(limit))


def data_version() -> str:
    """Reproducibility tag for a run: latest price date and row count in prices_daily."""
    t = schema.prices_daily
    with get_engine().connect() as c:
        d, n = c.execute(sa.select(sa.func.max(t.c.date), sa.func.count()).select_from(t)).one()
    return f"prices@{d} n={n}"


def table_counts(exact: bool = False) -> dict[str, int]:
    """Row counts. exact=False (default, Week 11) uses the planner's estimates from
    pg_class (microseconds) instead of nine COUNT(*) scans; audits pass exact=True."""
    out = {}
    with get_engine().connect() as c:
        if exact:
            for name, t in schema.metadata.tables.items():
                out[name] = int(c.execute(sa.select(sa.func.count()).select_from(t)).scalar())
        else:
            rows = c.execute(sa.text("SELECT relname, GREATEST(n_live_tup, 0) FROM pg_stat_user_tables")).all()
            est = {r[0]: int(r[1]) for r in rows}
            out = {name: est.get(name, 0) for name in schema.metadata.tables}
    return out
