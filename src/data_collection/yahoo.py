"""Yahoo Finance collector (primary equity/volatility source, via yfinance).

Series collected:
  - JPM daily OHLCV + adjusted close, dividends, stock splits
  - ^VIX  CBOE Volatility Index (market-implied 30d volatility)
  - ^TNX  10-Year Treasury yield index
  - ^IRX  13-Week T-bill yield index
  - ^GSPC S&P 500 (market context for feature engineering)
"""

import pandas as pd
import yfinance as yf

INDEX_TICKERS = {
    "^VIX": "vix",
    "^TNX": "treasury_10y_yahoo",
    "^IRX": "tbill_13w_yahoo",
    "^GSPC": "sp500",
}


def fetch_equity(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV with unadjusted prices plus dividends/splits columns."""
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


def fetch_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    # Index series carry no volume/dividend information worth keeping
    return df[["Open", "High", "Low", "Close"]]
