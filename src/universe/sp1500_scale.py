"""Scale-test universe (Week 11): current S&P 400 and S&P 600 constituents.

Purpose: stress the batched pipeline at ~3x the S&P 500 ticker count. These
names are CURRENT constituents only (no point-in-time reconstruction), so they
are tagged universe='sp400' / 'sp600' in `securities`, get NO index_membership
rows, and are therefore excluded from the research panel (which reads
membership for 'sp500' only). They are priced by the daily job like any other
active security.

Usage: python -m src.universe.sp1500_scale --load
"""

import io
import re
import sys

import pandas as pd
import requests

from src.universe.sp500 import UA, _norm

PAGES = {"sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
         "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"}


def fetch(universe: str) -> pd.DataFrame:
    html = requests.get(PAGES[universe], headers=UA, timeout=60).text
    tb = re.findall(r'<table class="wikitable[^"]*"(.*?)</table>', html, flags=re.S)[0]
    t = pd.read_html(io.StringIO("<table>" + tb + "</table>"))[0]
    t.columns = [str(c) for c in t.columns]
    return pd.DataFrame({"ticker": t["Symbol"].map(_norm), "name": t["Security"],
                         "sector": t["GICS Sector"], "universe": universe}).drop_duplicates("ticker")


def main() -> int:
    frames = [fetch(u) for u in PAGES]
    df = pd.concat(frames, ignore_index=True)
    print({u: int((df["universe"] == u).sum()) for u in PAGES}, "total", len(df))
    if "--load" in sys.argv:
        from src.db import repository as repo
        existing = set(repo.list_securities(active_only=False)["ticker"])
        new = df[~df["ticker"].isin(existing)]          # never re-tag an S&P 500 member
        n = repo.upsert_securities([{"ticker": r.ticker, "name": r.name, "sector": r.sector,
                                     "universe": r.universe, "active": True} for r in new.itertuples()])
        print(f"loaded {n} new securities ({len(df) - len(new)} already present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
