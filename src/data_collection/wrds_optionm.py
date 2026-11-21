"""OptionMetrics extract for the S&P 500 research panel (Week 12) — LOCAL ONLY.

Licence: WRDS data is for the author's own academic research. Everything this
module writes goes to data/wrds_local/ (git-ignored, excluded from the delivery
zip by scripts/package_week.sh and from the Docker image by .dockerignore).
Nothing from it is used by the pricing tool.

What it extracts
  sigma_iv(ticker, date) = mean of the 365-day, |delta| = 50 call and put
  implied vols from optionm.vsurfdYYYY (the Week 11 JPM definition).

Security mapping (pre-registered, docs/week12/preregistration.json)
  Tickers are reused (FB was a building-products firm, then Facebook, later an
  ETF) and renamed (FB -> META), so a ticker match alone is unsafe. Every secid
  that ever carried the ticker in optionm.secnmd is a candidate; the candidate
  whose daily returns (optionm.secprd.return) correlate best with ours is kept,
  and only if corr >= 0.98 over >= 250 common days.

Connection: direct SQLAlchemy to wrds-pgdata (the `wrds` package's catalog
initialisation is cut off by the server). The user name comes from WRDS_USER
(.env) or from the wrds-pgdata line of ~/.pgpass; the password stays in ~/.pgpass.

Usage: python -m src.data_collection.wrds_optionm
Outputs: data/wrds_local/optionm_sp500_iv365.parquet (ticker, date, sigma_iv)
         data/wrds_local/optionm_secid_map.csv      (ticker -> secid, match stats)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src import config

OUT_DIR = config.PROJECT_ROOT / "data" / "wrds_local"
HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"
YEARS = range(2019, 2025)
MIN_CORR, MIN_DAYS = 0.98, 250


def _wrds_user() -> str:
    if os.environ.get("WRDS_USER"):
        return os.environ["WRDS_USER"]
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        for line in pgpass.read_text().splitlines():
            parts = line.split(":")
            if len(parts) >= 5 and parts[0] == HOST:
                return parts[3]
    raise RuntimeError("set WRDS_USER in .env or add a wrds-pgdata line to ~/.pgpass")


def engine():
    return sa.create_engine(f"postgresql+psycopg2://{_wrds_user()}@{HOST}:{PORT}/{DB}?sslmode=require")


def panel_tickers() -> list[str]:
    from src.db import repository as repo
    mem = repo.read_membership("sp500")
    return sorted(set(mem["ticker"]) & set(repo.list_securities()["ticker"]))


def candidates(c, tickers: list[str]) -> pd.DataFrame:
    q = sa.text("SELECT DISTINCT secid::bigint AS secid, ticker FROM optionm.secnmd WHERE ticker = ANY(:t)")
    return pd.read_sql(q, c, params={"t": tickers})


def om_returns(c, secids: list[int]) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        q = sa.text(f"SELECT secid::bigint AS secid, date, return FROM optionm.secprd{y} WHERE secid = ANY(:s)")
        parts.append(pd.read_sql(q, c, params={"s": secids}))
    out = pd.concat(parts)
    out["date"] = pd.to_datetime(out["date"])
    return out.dropna()


def our_returns(tickers: list[str]) -> pd.DataFrame:
    from src.db import repository as repo
    rows = []
    for t in tickers:
        p = repo.read_prices(t)
        if not len(p):
            continue
        r = p["adj_close"].pct_change().dropna()
        r = r[(r.index >= f"{YEARS[0]}-01-01") & (r.index <= f"{YEARS[-1]}-12-31")]
        rows.append(pd.DataFrame({"ticker": t, "date": r.index, "ret": r.to_numpy()}))
    return pd.concat(rows)


def match(cand: pd.DataFrame, omr: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in cand.groupby("ticker"):
        mine = ours[ours["ticker"] == t].set_index("date")["ret"]
        best = None
        for sid in g["secid"]:
            theirs = omr[omr["secid"] == sid].set_index("date")["return"]
            common = mine.index.intersection(theirs.index)
            if len(common) < MIN_DAYS:
                continue
            corr = float(np.corrcoef(mine.loc[common], theirs.loc[common])[0, 1])
            if best is None or corr > best[1]:
                best = (int(sid), corr, len(common))
        ok = best is not None and best[1] >= MIN_CORR
        rows.append({"ticker": t, "secid": best[0] if best else None, "corr": best[1] if best else None,
                     "common_days": best[2] if best else 0, "candidates": int(len(g)), "matched": bool(ok)})
    return pd.DataFrame(rows)


def iv365(c, secids: list[int]) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        q = sa.text(f"SELECT secid::bigint AS secid, date, cp_flag, impl_volatility FROM optionm.vsurfd{y} "
                    "WHERE secid = ANY(:s) AND days = 365 AND delta IN (50, -50)")
        parts.append(pd.read_sql(q, c, params={"s": secids}))
    df = pd.concat(parts)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index=["secid", "date"], columns="cp_flag", values="impl_volatility").reset_index()
    wide["sigma_iv"] = wide[["C", "P"]].mean(axis=1)
    return wide[["secid", "date", "sigma_iv"]].dropna()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = panel_tickers()
    ours = our_returns(tickers)
    with engine().connect() as c:
        cand = candidates(c, tickers)
        omr = om_returns(c, sorted(cand["secid"].unique().tolist()))
        m = match(cand, omr, ours)
        ok = m[m["matched"]]
        iv = iv365(c, sorted(ok["secid"].astype(int).unique().tolist()))
    iv = iv.merge(ok[["ticker", "secid"]].astype({"secid": int}), on="secid")[["ticker", "date", "sigma_iv"]]
    m.to_csv(OUT_DIR / "optionm_secid_map.csv", index=False)
    iv.to_parquet(OUT_DIR / "optionm_sp500_iv365.parquet", index=False)
    print(f"tickers {len(tickers)} | with candidates {m['ticker'].nunique()} | matched {int(m['matched'].sum())} "
          f"| IV rows {len(iv)} | {iv['date'].min().date()}..{iv['date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
