"""Week 2 preprocessing & feature engineering pipeline.

Steps:
  1. Load raw series from data/raw/
  2. Align everything to the NYSE trading calendar (JPM's index)
  3. Clean: interpolate FRED holiday gaps, forward-fill monthly FEDFUNDS,
     flag (but retain) IQR outliers with an audit report
  4. Engineer features (18; project requirement >= 10)
  5. Save full and model-ready datasets to data/processed/ (CSV + Parquet)

Usage:  python -m src.preprocess          (run from project root)
"""

import json

import numpy as np
import pandas as pd

from src import config

TRADING_DAYS = 252


# ---------------------------------------------------------------- loading

def load_raw() -> dict[str, pd.DataFrame]:
    read = lambda name: pd.read_csv(
        config.DATA_RAW / name, index_col="date", parse_dates=True
    )
    return {
        "jpm": read("jpm_daily.csv"),
        "vix": read("vix_daily.csv"),
        "sp500": read("sp500_daily.csv"),
        "rates": read("fred_rates_daily.csv"),
    }


# ---------------------------------------------------------------- cleaning

def align_and_clean(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all sources onto the NYSE calendar defined by JPM trading days."""
    calendar = raw["jpm"].index

    df = pd.DataFrame(index=calendar)
    df.index.name = "date"

    df["jpm_close"] = raw["jpm"]["Close"]
    df["jpm_adj_close"] = raw["jpm"]["Adj Close"]
    df["jpm_high"] = raw["jpm"]["High"]
    df["jpm_low"] = raw["jpm"]["Low"]
    df["jpm_volume"] = raw["jpm"]["Volume"]
    df["jpm_dividend"] = raw["jpm"]["Dividends"].fillna(0.0)
    df["vix_close"] = raw["vix"]["Close"].reindex(calendar)
    df["spx_adj_close"] = raw["sp500"]["Close"].reindex(calendar)

    # FRED series live on a business-day calendar with NaN on holidays:
    # interpolate daily series in time, step-fill the monthly FEDFUNDS,
    # then sample onto the NYSE calendar.
    rates = raw["rates"]
    daily_cols = ["DGS3MO", "DGS1", "DGS10", "T10Y2Y"]
    rates_daily = rates[daily_cols].interpolate(method="time", limit_direction="both")
    fedfunds = rates["FEDFUNDS"].ffill()
    df[[c.lower() for c in daily_cols]] = rates_daily.reindex(calendar).ffill()
    df["fedfunds"] = fedfunds.reindex(calendar).ffill()

    # VIX/SPX occasionally trade on days JPM doesn't (and vice versa)
    df[["vix_close", "spx_adj_close"]] = df[["vix_close", "spx_adj_close"]].ffill()
    return df


def iqr_outlier_report(df: pd.DataFrame, cols: list[str], k: float = 3.0) -> dict:
    """Flag extreme values by the IQR rule. Flagged points are RETAINED:
    in this window they correspond to genuine market events (COVID crash,
    2022 hikes), which are exactly the regimes the project studies. The
    report is saved for audit so the decision is explicit, not silent."""
    report = {}
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        flagged = s[(s < lo) | (s > hi)]
        report[col] = {
            "n_flagged": int(len(flagged)),
            "bounds": [round(float(lo), 6), round(float(hi), 6)],
            "worst_dates": [d.strftime("%Y-%m-%d") for d in flagged.abs().nlargest(3).index],
        }
    return report


# ---------------------------------------------------------------- features

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Week 2 features on the JPM-named frame.

    Week 9: the arithmetic moved to the ticker-agnostic `src.features`
    (shared with the database pipeline); this wrapper maps jpm_* names to
    the generic names and back so every downstream file consumer is
    unchanged.  analysis/week9 audit 1 checks the round trip is exact.
    """
    from src import features as F
    return F.to_file_names(F.engineer_features(F.to_db_names(df)))


FEATURE_COLS = [
    "log_return", "realized_vol_21d", "realized_vol_63d", "realized_vol_252d",
    "parkinson_vol_21d", "div_yield_ttm", "div_growth_yoy",
    "r_3m", "r_1y", "curve_slope", "rate_momentum_21d",
    "spx_log_return", "beta_63d", "corr_jpm_vix_63d",
    "vix_level", "vix_sentiment_pct", "vix_momentum_5d", "fear_regime",
]


# ---------------------------------------------------------------- pipeline

def main() -> int:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df = align_and_clean(raw)

    report = iqr_outlier_report(
        df.assign(log_return=np.log(df["jpm_adj_close"]).diff()),
        ["log_return", "vix_close", "dgs3mo"],
    )
    (config.DATA_PROCESSED / "outlier_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    full = engineer_features(df)
    keep = ["jpm_close", "jpm_adj_close", "vix_close"] + FEATURE_COLS

    full[keep].to_csv(config.DATA_PROCESSED / "features_daily.csv")
    full[keep].to_parquet(config.DATA_PROCESSED / "features_daily.parquet")

    # Model-ready view: drop the rolling-window warm-up rows
    model_ready = full[keep].dropna()
    model_ready.to_csv(config.DATA_PROCESSED / "features_daily_model_ready.csv")
    model_ready.to_parquet(config.DATA_PROCESSED / "features_daily_model_ready.parquet")

    print(f"features           : {len(FEATURE_COLS)}")
    print(f"full dataset       : {full.shape[0]} rows  ({full.index.min().date()} .. {full.index.max().date()})")
    print(f"model-ready dataset: {model_ready.shape[0]} rows  ({model_ready.index.min().date()} .. {model_ready.index.max().date()})")
    print(f"outliers flagged   : " + ", ".join(f"{k}={v['n_flagged']}" for k, v in report.items()))
    print(f"NaN in model-ready : {int(model_ready.isna().sum().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
