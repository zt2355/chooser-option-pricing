"""Feature rebuild against the database (Week 10): full or incremental.

Full rebuild      recompute every row of one ticker from its whole price
                  history (Week 9 behaviour).
Incremental       recompute only the tail. The longest look-back in the
                  Week 2 feature set is 504 rows (div_growth_yoy = 252-day
                  change of a 252-day rolling sum), so a tail of TAIL rows
                  reproduces every feature exactly for the last
                  TAIL - 504 rows. We rewrite the last WRITE_BACK rows,
                  which comfortably covers the 10-day provider overlap.
                  The EWMA anchor is an infinite-memory recursion, so it is
                  seeded from the stored value at the cut-over row:
                  v_t = lambda v_{t-1} + (1-lambda) r_t^2 is exact given the
                  seed, and analysis/week10 audit 2 asserts full ==
                  incremental bit-for-bit.

Derived state stored alongside the 18 features:
  sigma_ewma   EWMA(0.94) annualised volatility (the tool's primary anchor)
  r_1y_dev     r_1y minus its trailing 252-day mean (W6 ML feature)
so that a quote reads one row instead of the whole history.
"""

import numpy as np
import pandas as pd

from src import features as F
from src import garch_state as gs
from src.db import repository as repo

EWMA_LAMBDA = 0.94
TAIL, WRITE_BACK = 760, 40


def derive_state(df: pd.DataFrame, seed_var: float | None = None) -> pd.DataFrame:
    """Add sigma_ewma / r_1y_dev to a generic feature frame.

    seed_var=None  -> pandas ewm(adjust=False) on the whole frame (identical to
                      src.tool.pricing_service.derive, incl. its NaN semantics)
    seed_var=v     -> exact recursion v_t = lambda v_{t-1} + (1-lambda) r_t^2
                      seeded with the EWMA variance of the row before df
                      (used by the incremental rebuild; tail has no NaN returns)
    """
    out = df.copy()
    r2 = out["log_return"] ** 2
    if seed_var is None:
        v = r2.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().to_numpy()
    else:
        x = r2.to_numpy(dtype=float)
        v = np.empty(len(x)); prev = seed_var
        for i, xi in enumerate(x):
            if not np.isnan(xi):
                prev = EWMA_LAMBDA * prev + (1 - EWMA_LAMBDA) * xi
            v[i] = prev
    out["sigma_ewma"] = np.sqrt(v * 252)
    out["r_1y_dev"] = out["r_1y"] - out["r_1y"].rolling(252, min_periods=1).mean()
    return out


def _market_inputs():
    vix = repo.read_index("^VIX")["close"]
    spx = repo.read_index("^GSPC")["close"]
    rates = repo.read_rates()
    return vix, spx, rates


def with_garch(feats: pd.DataFrame) -> pd.DataFrame:
    """Week 12: full GARCH-1d path + state for a whole history (NaN before MIN_OBS)."""
    out = feats.drop(columns=[c for c in gs.OUT_COLS if c in feats.columns])
    if not len(out):
        return out.reindex(columns=list(out.columns) + gs.OUT_COLS)
    return out.join(gs.path(out["log_return"]), how="left")


def garch_tail(tail: pd.DataFrame, stored: pd.DataFrame, fetch_full_lr) -> pd.DataFrame:
    """Week 12: GARCH columns for the rewritten tail, continuing the stored recursion.

    Rows whose date and log return are unchanged keep their stored GARCH values (they
    depend only on returns up to that day). From the first new/changed row on, the
    recursion restarts from the stored state of the previous return; a refit, when the
    schedule calls for one, uses the full return history (`fetch_full_lr()`).
    Bit-identical to with_garch() on the full history (analysis/week12 audit)."""
    tail = tail.drop(columns=[c for c in gs.OUT_COLS if c in tail.columns])
    have = stored.reindex(tail.index)
    if "garch_h_next" not in stored.columns or stored["garch_h_next"].isna().all():
        changed = pd.Series(True, index=tail.index)                    # never computed: full path needed
    else:
        same_r = (have["log_return"] == tail["log_return"]) | (have["log_return"].isna() & tail["log_return"].isna())
        changed = pd.Series(~(tail.index.isin(stored.index) & same_r.to_numpy()), index=tail.index)
    if not changed.any():
        return tail.join(have[gs.OUT_COLS])
    k = tail.index[changed.to_numpy().argmax()]                       # first new/changed row
    seed = None
    if "garch_h_next" in stored.columns:
        prev = stored[(stored.index < k) & stored["log_return"].notna()]
        if len(prev) and pd.notna(prev["garch_h_next"].iloc[-1]):
            seed = prev[gs.STATE_COLS].iloc[-1].to_dict()
    # ordinary day: no refit scheduled among the new returns -> continue from the stored state
    # without reading the return history (Week 12 perf; same arithmetic, audit 3)
    new = gs.continue_path(tail.loc[k:, "log_return"], seed)
    if new is None:                                                   # refit day / no state: full history
        hist = fetch_full_lr()
        full_lr = pd.concat([hist[hist.index < k], tail.loc[k:, "log_return"]])
        r = full_lr.dropna()
        pos = int((r.index < k).sum())                                # position of k in the return series
        if seed is not None and int(seed["garch_i"]) + 1 != pos:     # state/history mismatch: recompute all
            seed = None
        new = gs.path(full_lr, start=pos if seed is not None else 0, seed=seed)
    out = tail.join(have[gs.OUT_COLS])
    sel = new.index[new.index >= k]
    out.loc[k:, gs.OUT_COLS] = np.nan
    out.loc[sel, gs.OUT_COLS] = new.loc[sel, gs.OUT_COLS].to_numpy()
    return out


def rebuild_full(ticker: str, market=None) -> pd.DataFrame:
    vix, spx, rates = market or _market_inputs()
    eq = repo.read_prices(ticker)
    feats = with_garch(derive_state(F.build_features(eq, vix, spx, rates)))
    with repo.get_engine().begin() as c:
        repo.upsert_features(ticker, feats, conn=c)
        repo.upsert_latest(ticker, feats, conn=c)
    return feats


def rebuild_incremental(ticker: str, market=None) -> pd.DataFrame:
    """Recompute the tail and write back the last WRITE_BACK rows. Falls back to
    a full rebuild when the ticker has no stored state yet."""
    vix, spx, rates = market or _market_inputs()
    stored = repo.read_features_tail(ticker, TAIL)
    if len(stored) < TAIL or stored["sigma_ewma"].isna().all():
        return rebuild_full(ticker, (vix, spx, rates))
    eq = repo.read_prices_tail(ticker, TAIL)
    feats = F.build_features(eq, vix, spx, rates)
    cut = feats.index[-WRITE_BACK]                      # first row we rewrite
    prior = stored.loc[:cut].iloc[:-1] if cut in stored.index else stored.loc[:cut]
    seed = None
    if len(prior) and pd.notna(prior["sigma_ewma"].iloc[-1]):
        seed = float(prior["sigma_ewma"].iloc[-1]) ** 2 / 252.0
    tail = derive_state(feats.loc[cut:], seed)
    # r_1y_dev needs 252 rows of r_1y context: compute on the wide frame, then slice
    tail["r_1y_dev"] = (feats["r_1y"] - feats["r_1y"].rolling(252, min_periods=1).mean()).loc[cut:]
    tail = garch_tail(tail, stored, lambda: repo.read_log_returns([ticker]).get(ticker, pd.Series(dtype=float)))
    with repo.get_engine().begin() as c:
        repo.upsert_features(ticker, tail, conn=c)
        repo.upsert_latest(ticker, tail, conn=c)
    return tail


def _incremental_tail(ticker: str, eq: pd.DataFrame, stored: pd.DataFrame, market) -> pd.DataFrame | None:
    """Pure computation of the write-back tail; None when a full rebuild is needed."""
    vix, spx, rates = market
    if len(stored) < TAIL or stored["sigma_ewma"].isna().all() or len(eq) < TAIL:
        return None
    feats = F.build_features(eq, vix, spx, rates)
    cut = feats.index[-WRITE_BACK]
    prior = stored.loc[:cut].iloc[:-1] if cut in stored.index else stored.loc[:cut]
    seed = float(prior["sigma_ewma"].iloc[-1]) ** 2 / 252.0 if len(prior) and pd.notna(prior["sigma_ewma"].iloc[-1]) else None
    tail = derive_state(feats.loc[cut:], seed)
    tail["r_1y_dev"] = (feats["r_1y"] - feats["r_1y"].rolling(252, min_periods=1).mean()).loc[cut:]
    return tail


def rebuild_many(tickers: list[str], market=None, chunk: int = 100, errors: dict | None = None) -> dict[str, int]:
    """Week 11 (perf): incremental rebuild for many tickers with batched I/O —
    the tails of `chunk` tickers are read in two window-function queries and
    written back in one transaction, instead of 3 round trips per ticker.
    Matches rebuild_incremental / rebuild_full to floating-point rounding
    (analysis/week11 audit 2: max diff ~1e-14, tolerance 1e-9).
    Week 12: per-ticker fault isolation - a ticker that fails is recorded in `errors`
    (ticker -> message) and skipped; the rest of the chunk is still written."""
    errors = {} if errors is None else errors
    market = market or _market_inputs()
    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        stored = repo.read_features_tails(batch, TAIL)
        prices = repo.read_prices_tails(batch, TAIL)
        full_needed, tails = [], {}
        for t in batch:
            try:
                tail = _incremental_tail(t, prices.get(t, pd.DataFrame()), stored.get(t, pd.DataFrame()), market)
            except Exception as e:
                errors[t] = f"tail: {e}"[:160]; continue
            if tail is None:
                full_needed.append(t)
            else:
                tails[t] = tail
        # Week 12: GARCH tails; the full return history is read once per chunk, lazily
        lr_cache: dict = {}
        def fetch(t):
            if not lr_cache:
                lr_cache.update(repo.read_log_returns(list(tails)))
            return lr_cache.get(t, pd.Series(dtype=float))
        for t in list(tails):
            try:
                tails[t] = garch_tail(tails[t], stored.get(t, pd.DataFrame()), lambda t=t: fetch(t))
                out[t] = len(tails[t])
            except Exception as e:
                errors[t] = f"garch: {e}"[:160]; tails.pop(t)
        if tails:
            with repo.get_engine().begin() as c:          # one multi-row statement per chunk
                repo.upsert_features_many(tails, c)
                repo.upsert_latest_many(tails, c)
        for t in full_needed:
            try:
                out[t] = len(rebuild_full(t, market))
            except Exception as e:
                errors[t] = f"full: {e}"[:160]
    return out


def rebuild_features(ticker: str, incremental: bool = True, market=None) -> pd.DataFrame:
    return rebuild_incremental(ticker, market) if incremental else rebuild_full(ticker, market)


# ------------------------------------------------------------ Week 12 backfill

def _garch_job(args):
    import warnings
    t, lr = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = gs.path(lr)
    out["ticker"] = t
    return out.reset_index().rename(columns={"index": "date"})


def backfill_garch(tickers: list[str] | None = None, workers: int = 8, chunk: int = 100) -> int:
    """One-off Week 12 migration: GARCH-1d path + state for every stored history.
    Computes in a process pool, writes through COPY into a temp table and one
    UPDATE ... FROM (row-by-row upserts of ~3M rows would take minutes), then
    copies the values of each ticker's latest complete row into features_latest."""
    import io
    from concurrent.futures import ProcessPoolExecutor
    import sqlalchemy as sa
    eng = repo.get_engine()
    if tickers is None:
        with eng.connect() as c:
            tickers = [r[0] for r in c.execute(sa.text("SELECT DISTINCT ticker FROM features_daily ORDER BY 1"))]
    cols = gs.OUT_COLS
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(tickers), chunk):
            lrs = repo.read_log_returns(tickers[i:i + chunk])
            parts = list(ex.map(_garch_job, list(lrs.items())))
            df = pd.concat(parts)[["ticker", "date"] + cols]
            buf = io.StringIO(); df.to_csv(buf, index=False, header=False, na_rep="\\N"); buf.seek(0)
            raw = eng.raw_connection()
            try:
                cur = raw.cursor()
                cur.execute("CREATE TEMP TABLE tmp_garch (ticker varchar(16), date date, "
                            + ", ".join(f"{c} double precision" for c in cols) + ") ON COMMIT DROP")
                cur.copy_expert("COPY tmp_garch FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
                cur.execute("UPDATE features_daily f SET " + ", ".join(f"{c} = t.{c}" for c in cols)
                            + " FROM tmp_garch t WHERE f.ticker = t.ticker AND f.date = t.date")
                n += cur.rowcount
                raw.commit()
            finally:
                raw.close()
    with eng.begin() as c:
        c.execute(sa.text("UPDATE features_latest l SET " + ", ".join(f"{x} = f.{x}" for x in cols)
                          + " FROM features_daily f WHERE f.ticker = l.ticker AND f.date = l.date"))
    return n


if __name__ == "__main__":
    import sys
    if "--backfill-garch" in sys.argv:
        print("GARCH rows written:", backfill_garch())
