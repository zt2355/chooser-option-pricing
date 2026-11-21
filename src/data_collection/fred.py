"""FRED collector (macro/rates source, via the official REST API).

Series collected:
  - DGS3MO   3-Month Treasury constant maturity (risk-free proxy for short-dated options)
  - DGS1     1-Year Treasury constant maturity (matches T2 = 1y in the paper)
  - DGS10    10-Year Treasury constant maturity
  - FEDFUNDS Effective federal funds rate (monthly)
  - T10Y2Y   10y-2y spread (yield-curve regime feature)
"""

import pandas as pd
import requests

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES = ["DGS3MO", "DGS1", "DGS10", "FEDFUNDS", "T10Y2Y"]


def fetch_series(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    s = pd.Series(
        {o["date"]: o["value"] for o in obs}, name=series_id, dtype="object"
    )
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    # FRED encodes missing values as "."
    return pd.to_numeric(s, errors="coerce")


def fetch_all(api_key: str, start: str, end: str) -> pd.DataFrame:
    return pd.concat(
        [fetch_series(sid, api_key, start, end) for sid in SERIES], axis=1
    )
