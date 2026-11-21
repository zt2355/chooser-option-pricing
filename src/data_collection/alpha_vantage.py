"""Alpha Vantage collector (backup/cross-validation equity source).

Free tier allows 25 requests/day and restricts TIME_SERIES_DAILY to
outputsize=compact (latest ~100 sessions), so this source is used only
to cross-check recent Yahoo Finance prices, not as the primary feed.
"""

import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    key = "Time Series (Daily)"
    if key not in payload:
        raise RuntimeError(f"Alpha Vantage error for {symbol}: {payload}")
    df = pd.DataFrame.from_dict(payload[key], orient="index", dtype=float)
    df.columns = [c.split(". ")[1] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()
