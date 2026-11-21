"""Multi-source, rate-limit-aware price collection (Week 10).

Order of sources for a batch of tickers:
  1. yfinance batch download (one HTTP round trip per ~100 tickers, threaded)
  2. yfinance single-ticker history for whatever the batch left empty
     (a different endpoint; retried with exponential back-off)
  3. Alpha Vantage TIME_SERIES_DAILY for the residue — free tier, 25 calls a
     day, UNADJUSTED prices only (no dividends/splits) so rows are tagged
     source='alphavantage_unadj' and adj_close is set equal to close.
Tickers still empty after all three are returned in `report["missing"]`; the
pipeline marks them inactive with note 'no_price_history' instead of failing.

Every DataFrame returned has generic columns
  open high low close adj_close volume dividend split
indexed by naive trading dates.

Stooq was evaluated and rejected: its CSV endpoint now returns an HTML
bot-check page to scripted clients.
"""

import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from src import config

CHUNK = 100
RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
          "Adj Close": "adj_close", "Volume": "volume",
          "Dividends": "dividend", "Stock Splits": "split"}
COLS = list(RENAME.values())


@dataclass
class FetchReport:
    by_source: dict = field(default_factory=lambda: {"yfinance_batch": 0, "yfinance_single": 0,
                                                     "alphavantage_unadj": 0})
    missing: list = field(default_factory=list)
    retries: int = 0
    seconds: float = 0.0


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in ("dividend", "split") else None
    df = df[COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[df["close"].notna()]
    return df[~df.index.duplicated(keep="last")].sort_index()


def _with_backoff(fn, report: FetchReport, tries: int = 4, base: float = 1.5):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (rate limits surface as generic errors)
            report.retries += 1
            if k == tries - 1:
                raise
            time.sleep(base * (2 ** k))


def _yf_batch(tickers: list[str], start: str, end: str, report: FetchReport) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        raw = _with_backoff(lambda: yf.download(chunk, start=start, end=end, auto_adjust=False, actions=True,
                                                group_by="ticker", threads=True, progress=False), report)
        if raw is None or raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):     # single-ticker chunk
            raw = pd.concat({chunk[0]: raw}, axis=1)
        for t in chunk:
            if t in raw.columns.get_level_values(0):
                df = _clean(raw[t].dropna(how="all"))
                if len(df):
                    out[t] = df
        report.by_source["yfinance_batch"] += len([t for t in chunk if t in out])
    return out


def _yf_single(ticker: str, start: str, end: str, report: FetchReport) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = _with_backoff(lambda: yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False), report, tries=3)
    except Exception:
        return None
    df = _clean(df) if df is not None and len(df) else None
    if df is not None and len(df):
        report.by_source["yfinance_single"] += 1
    return df


def _alphavantage(ticker: str, start: str, report: FetchReport) -> pd.DataFrame | None:
    if not config.ALPHA_VANTAGE_API_KEY:
        return None
    try:
        r = requests.get("https://www.alphavantage.co/query", timeout=30,
                         params={"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "full",
                                 "apikey": config.ALPHA_VANTAGE_API_KEY})
        ts = r.json().get("Time Series (Daily)")
        if not ts:
            return None
    except Exception:
        return None
    df = pd.DataFrame(ts).T.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low",
                                             "4. close": "close", "5. volume": "volume"}).astype(float)
    df.index = pd.to_datetime(df.index)
    df["adj_close"] = df["close"]; df["dividend"] = 0.0; df["split"] = 0.0
    df = _clean(df[COLS]); df = df[df.index >= pd.Timestamp(start)]
    if len(df):
        report.by_source["alphavantage_unadj"] += 1
    return df if len(df) else None


def fetch_prices(tickers: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], dict[str, str], FetchReport]:
    """Returns ({ticker: frame}, {ticker: source}, report)."""
    t0 = time.perf_counter()
    rep = FetchReport()
    frames = _yf_batch(tickers, start, end, rep)
    source = {t: "yahoo" for t in frames}
    for t in [t for t in tickers if t not in frames]:
        df = _yf_single(t, start, end, rep)
        if df is not None:
            frames[t], source[t] = df, "yahoo"
    residue = [t for t in tickers if t not in frames]
    for t in residue[:20]:                      # respect the free-tier daily budget
        df = _alphavantage(t, start, rep)
        if df is not None:
            frames[t], source[t] = df, "alphavantage_unadj"
    rep.missing = [t for t in tickers if t not in frames]
    rep.seconds = time.perf_counter() - t0
    return frames, source, rep
