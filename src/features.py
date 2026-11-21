"""Ticker-agnostic feature engineering (Week 9 refactor of the Week 2 pipeline).

The Week 2 code computed the 18 features on JPM-named columns (jpm_close,
corr_jpm_vix_63d, ...). To serve a multi-ticker database the same arithmetic
now lives here on generic column names; `src.preprocess` calls this module
and renames back to the historical jpm_* names for the file outputs, so every
Week 3-8 script keeps its inputs unchanged.  analysis/week9 audit 1 asserts
the DB and file representations agree bit-for-bit.

Generic input columns (one ticker, NYSE calendar index named `date`):
  close adj_close high low volume dividend      equity
  vix_close spx_adj_close                       market context
  dgs3mo dgs1 dgs10 t10y2y fedfunds             rates (percent)
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FEATURE_COLS = [
    "log_return", "realized_vol_21d", "realized_vol_63d", "realized_vol_252d",
    "parkinson_vol_21d", "div_yield_ttm", "div_growth_yoy",
    "r_3m", "r_1y", "curve_slope", "rate_momentum_21d",
    "spx_log_return", "beta_63d", "corr_vix_63d",
    "vix_level", "vix_sentiment_pct", "vix_momentum_5d", "fear_regime",
]

# file (Week 2, JPM-specific) name  <->  generic / DB name
FILE_TO_DB = {"jpm_close": "close", "jpm_adj_close": "adj_close",
              "jpm_high": "high", "jpm_low": "low", "jpm_volume": "volume",
              "jpm_dividend": "dividend", "corr_jpm_vix_63d": "corr_vix_63d"}
DB_TO_FILE = {v: k for k, v in FILE_TO_DB.items()}


def align_market(equity: pd.DataFrame, vix: pd.Series, spx: pd.Series,
                 rates: pd.DataFrame) -> pd.DataFrame:
    """Merge one ticker's OHLCV with market context on the ticker's own
    trading calendar (same cleaning rules as Week 2)."""
    cal = equity.index
    df = pd.DataFrame(index=cal)
    df.index.name = "date"
    for c in ("close", "adj_close", "high", "low", "volume"):
        df[c] = equity[c]
    df["dividend"] = equity["dividend"].fillna(0.0)
    df["vix_close"] = vix.reindex(cal)
    df["spx_adj_close"] = spx.reindex(cal)
    daily_cols = ["dgs3mo", "dgs1", "dgs10", "t10y2y"]
    rates_daily = rates[daily_cols].interpolate(method="time", limit_direction="both")
    df[daily_cols] = rates_daily.reindex(cal).ffill()
    df["fedfunds"] = rates["fedfunds"].ffill().reindex(cal).ffill()
    df[["vix_close", "spx_adj_close"]] = df[["vix_close", "spx_adj_close"]].ffill()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """18 features on generic columns. Arithmetic identical to Week 2."""
    out = df.copy()
    r = np.log(out["adj_close"]).diff()

    out["log_return"] = r
    for w in (21, 63, 252):
        out[f"realized_vol_{w}d"] = r.rolling(w).std() * np.sqrt(TRADING_DAYS)
    hl = np.log(out["high"] / out["low"]) ** 2
    hl = hl.where((out["low"] > 0) & (out["high"] > 0))          # multi-ticker guard: bad prints -> NaN, not inf
    out["parkinson_vol_21d"] = np.sqrt(hl.rolling(21).mean() / (4 * np.log(2))) * np.sqrt(TRADING_DAYS)

    ttm_div = out["dividend"].rolling(TRADING_DAYS, min_periods=200).sum()
    out["div_yield_ttm"] = ttm_div / out["close"]
    # YoY growth of the trailing-12m dividend. Multi-ticker guard (Week 10):
    # a zero prior-year dividend makes pct_change +inf (first payer) or NaN
    # (non-payer); both are defined as 0 growth. JPM never hits this branch,
    # so the Week 2 file feature is unchanged (Week 9 audit 1).
    lag = ttm_div.shift(TRADING_DAYS)
    out["div_growth_yoy"] = np.where(lag > 0, (ttm_div - lag) / lag, np.where(lag == 0, 0.0, np.nan))
    out["div_growth_yoy"] = out["div_growth_yoy"].where(ttm_div.notna() & lag.notna())

    out["r_3m"] = out["dgs3mo"] / 100.0
    out["r_1y"] = out["dgs1"] / 100.0
    out["curve_slope"] = out["t10y2y"]
    out["rate_momentum_21d"] = out["dgs10"].diff(21)

    spx_r = np.log(out["spx_adj_close"]).diff()
    out["spx_log_return"] = spx_r
    out["beta_63d"] = r.rolling(63).cov(spx_r) / spx_r.rolling(63).var()
    out["corr_vix_63d"] = r.rolling(63).corr(out["vix_close"].diff())

    out["vix_level"] = out["vix_close"]
    out["vix_sentiment_pct"] = out["vix_close"].rolling(TRADING_DAYS).rank(pct=True)
    out["vix_momentum_5d"] = out["vix_close"].diff(5)
    out["fear_regime"] = ((out["vix_sentiment_pct"] > 0.75) | (out["vix_close"] > 30)).astype(int)
    return out


def build_features(equity: pd.DataFrame, vix: pd.Series, spx: pd.Series,
                   rates: pd.DataFrame) -> pd.DataFrame:
    """Full per-ticker feature frame: close, adj_close, vix_close + 18 features."""
    full = engineer_features(align_market(equity, vix, spx, rates))
    return full[["close", "adj_close", "vix_close"] + FEATURE_COLS]


def to_file_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=DB_TO_FILE)


def to_db_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FILE_TO_DB)
