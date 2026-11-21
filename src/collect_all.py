"""Week 1 initial data pull: collect all raw series and write CSVs to data/raw/.

Usage:  python -m src.collect_all          (run from project root)
"""

import sys

from src import config
from src.data_collection import alpha_vantage, fred, yahoo


def main() -> int:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    start, end = config.START_DATE, config.END_DATE

    print(f"Collecting raw data {start} -> {end} into {config.DATA_RAW}")

    # --- Yahoo Finance: JPM equity ---
    jpm = yahoo.fetch_equity(config.TICKER, start, end)
    jpm.to_csv(config.DATA_RAW / "jpm_daily.csv")
    print(f"  jpm_daily.csv            {len(jpm):>5} rows  ({jpm.index.min().date()} .. {jpm.index.max().date()})")

    # --- Yahoo Finance: indices ---
    for ticker, name in yahoo.INDEX_TICKERS.items():
        df = yahoo.fetch_index(ticker, start, end)
        df.to_csv(config.DATA_RAW / f"{name}_daily.csv")
        print(f"  {name + '_daily.csv':<24} {len(df):>5} rows  ({df.index.min().date()} .. {df.index.max().date()})")

    # --- FRED: rates ---
    if not config.FRED_API_KEY:
        print("  !! FRED_API_KEY missing - skipping FRED collection", file=sys.stderr)
    else:
        rates = fred.fetch_all(config.FRED_API_KEY, start, end)
        rates.to_csv(config.DATA_RAW / "fred_rates_daily.csv")
        print(f"  fred_rates_daily.csv     {len(rates):>5} rows  ({rates.index.min().date()} .. {rates.index.max().date()})")

    # --- Alpha Vantage: JPM cross-check, last ~100 sessions only (free tier) ---
    if not config.ALPHA_VANTAGE_API_KEY:
        print("  !! ALPHA_VANTAGE_API_KEY missing - skipping cross-check", file=sys.stderr)
    else:
        # Free tier only returns the latest ~100 sessions; keep them unfiltered.
        # Serves as connectivity check now and real-time backup source in Week 7.
        av = alpha_vantage.fetch_daily(config.TICKER, config.ALPHA_VANTAGE_API_KEY)
        av.to_csv(config.DATA_RAW / "jpm_daily_alphavantage.csv")
        print(f"  jpm_daily_alphavantage.csv {len(av):>3} rows  ({av.index.min().date()} .. {av.index.max().date()})")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
