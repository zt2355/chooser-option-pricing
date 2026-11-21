"""Point-in-time S&P 500 membership from public sources (Week 10).

Sources (all free, no licence restriction on redistribution of the list):
  * current constituents table on Wikipedia "List of S&P 500 companies"
  * the "Selected changes" table (effective date, added, removed). Wikipedia
    dropped that table from the live page in 2026; we read it from the last
    revision that still carries it (revid 1355685534, 2026-05-23) and treat
    later changes as: anything in the current table with "Date added" after
    the revision date is an addition; anything in the revision's constituent
    table that is absent from the current table is a removal dated at the
    current fetch date (exact removal date unknown — flagged in the CSV).

Reconstruction: walk the changes backwards from today's set. A ticker's
membership is a list of [start, end) intervals; start=None means "member
before the study window" (data start 2018-01-01), end=None means "still a
member". Ticker symbols are normalised to Yahoo style (BRK.B -> BRK-B).

Known limitations (documented, not hidden): symbol changes (FB -> META) are
recorded as remove+add in the source and therefore look like two tickers;
tickers delisted before Yahoo's history are flagged `active=False` with
reason 'no_price_history' by the pipeline, so the training panel carries a
partial survivorship bias for pre-2020 removals. Coverage is reported by
analysis/week10/01_universe_audits.py.

Usage:
    python -m src.universe.sp500            build and print coverage
    python -m src.universe.sp500 --load     also write securities + index_membership
Outputs: data/universe/sp500_current.csv, sp500_changes.csv, sp500_membership.csv
"""

import io
import re
import sys

import pandas as pd
import requests

from src import config

UA = {"User-Agent": "Mozilla/5.0 (academic research script; contact: zt2355@columbia.edu)"}
PAGE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OLD_REVISION = 1355685534          # 2026-05-23, last revision with the changes table
OLD_REVISION_DATE = "2026-05-23"
OUT = config.PROJECT_ROOT / "data" / "universe"
WINDOW_START = pd.Timestamp(config.START_DATE)


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def _wikitables(html: str) -> list[pd.DataFrame]:
    tabs = re.findall(r'<table class="wikitable[^"]*"(.*?)</table>', html, flags=re.S)
    out = []
    for tb in tabs:
        try:
            out.append(pd.read_html(io.StringIO("<table>" + tb + "</table>"))[0])
        except Exception:
            pass
    return out


def fetch_current() -> pd.DataFrame:
    html = requests.get(PAGE, headers=UA, timeout=60).text
    t = _wikitables(html)[0]
    t.columns = [str(c) for c in t.columns]
    df = pd.DataFrame({"ticker": t["Symbol"].map(_norm), "name": t["Security"],
                       "sector": t["GICS Sector"], "date_added": pd.to_datetime(t["Date added"], errors="coerce")})
    return df.drop_duplicates("ticker").reset_index(drop=True)


def fetch_changes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(changes, constituents at the old revision)."""
    html = requests.get("https://en.wikipedia.org/w/index.php",
                        params={"title": "List_of_S%26P_500_companies", "oldid": OLD_REVISION},
                        headers=UA, timeout=60).text
    tabs = _wikitables(html)
    old = tabs[0]; old.columns = [str(c) for c in old.columns]
    old_set = pd.DataFrame({"ticker": old["Symbol"].map(_norm)})
    ch = tabs[1]
    ch.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"][:len(ch.columns)]
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch["added"] = ch["added"].map(lambda x: _norm(x) if isinstance(x, str) else None)
    ch["removed"] = ch["removed"].map(lambda x: _norm(x) if isinstance(x, str) else None)
    ch = ch.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return ch, old_set


def build_membership(current: pd.DataFrame, changes: pd.DataFrame, old_set: pd.DataFrame,
                     today: pd.Timestamp | None = None) -> pd.DataFrame:
    today = today or pd.Timestamp.today().normalize()
    rev_date = pd.Timestamp(OLD_REVISION_DATE)
    # events after the old revision, inferred from the two constituent snapshots
    extra = []
    cur_set = set(current["ticker"])
    for _, r in current.iterrows():
        if pd.notna(r["date_added"]) and r["date_added"] > rev_date:
            extra.append({"date": r["date_added"], "added": r["ticker"], "removed": None, "reason": "inferred: in current table, added after revision"})
    for t in set(old_set["ticker"]) - cur_set:
        extra.append({"date": today, "added": None, "removed": t, "reason": "inferred: dropped after revision (exact date unknown)"})
    ev = pd.concat([changes[["date", "added", "removed", "reason"]], pd.DataFrame(extra)], ignore_index=True)
    ev = ev[ev["date"] >= WINDOW_START - pd.Timedelta(days=1)].sort_values("date")

    # walk backwards from today's set to produce intervals
    intervals: dict[str, list[list]] = {t: [[None, None]] for t in cur_set}   # open interval for current members
    for _, e in ev.sort_values("date", ascending=False).iterrows():
        d = e["date"]
        if e["added"]:
            t = e["added"]
            if t in intervals and intervals[t] and intervals[t][-1][0] is None:
                intervals[t][-1][0] = d          # the open interval started here
            elif t not in intervals:
                intervals[t] = [[d, d]]          # added then removed the same day? degenerate, keep
        if e["removed"]:
            t = e["removed"]
            intervals.setdefault(t, []).append([None, d])   # a membership that ended on d (start unknown yet)
    rows = []
    for t, ivs in intervals.items():
        for s, e in ivs:
            s2 = WINDOW_START if s is None else max(s, WINDOW_START)
            if e is not None and e <= s2:
                continue
            rows.append({"ticker": t, "start": s2, "end": e, "start_known": s is not None})
    mem = pd.DataFrame(rows).sort_values(["ticker", "start"]).reset_index(drop=True)
    return mem


def universe_on(mem: pd.DataFrame, date) -> set[str]:
    d = pd.Timestamp(date)
    m = mem[(mem["start"] <= d) & (mem["end"].isna() | (mem["end"] > d))]
    return set(m["ticker"])


def build(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cur = fetch_current()
    ch, old_set = fetch_changes()
    mem = build_membership(cur, ch, old_set)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        cur.to_csv(OUT / "sp500_current.csv", index=False)
        ch.to_csv(OUT / "sp500_changes.csv", index=False)
        mem.to_csv(OUT / "sp500_membership.csv", index=False)
    return cur, ch, mem


def load_to_db(cur: pd.DataFrame, mem: pd.DataFrame) -> dict:
    from src.db import repository as repo
    names = dict(zip(cur["ticker"], cur["name"]))
    sectors = dict(zip(cur["ticker"], cur["sector"]))
    tickers = sorted(set(mem["ticker"]))
    n = repo.upsert_securities([{"ticker": t, "name": names.get(t), "sector": sectors.get(t),
                                 "universe": "sp500", "active": True} for t in tickers])
    m = repo.replace_membership("sp500", mem)
    return {"securities": n, "membership_rows": m, "deactivated_after_exit": deactivate_post_exit_tickers()}


def post_exit_tickers() -> list[str]:
    """Week 12: former members whose stored price history starts only AFTER they left
    the index. Either the ticker was reused by another issuer (APC, CSRA, INFO, NFX,
    SBNY, STI) or only a post-exit fragment was collected (LEG): in both cases the
    rows are not the member's history and must not be quoted as an S&P 500 name."""
    import sqlalchemy as sa
    from src.db import repository as repo
    from src.db.engine import get_engine
    mem = repo.read_membership("sp500")
    closed = mem.groupby("ticker")["end"].apply(lambda s: None if s.isna().any() else s.max()).dropna()
    with get_engine().connect() as c:
        first = dict(c.execute(sa.text("SELECT ticker, min(date) FROM prices_daily GROUP BY ticker")).all())
    return sorted(t for t, end in closed.items() if t in first and pd.Timestamp(first[t]) > pd.Timestamp(end))


def deactivate_post_exit_tickers() -> list[str]:
    from src.db import repository as repo
    out = post_exit_tickers()
    for t in out:
        repo.set_security_note(t, active=False, note="prices start after index exit (ticker reused / post-exit fragment)")
    return out


def main() -> int:
    cur, ch, mem = build()
    today = pd.Timestamp.today().normalize()
    u_now = universe_on(mem, today)
    print(f"current constituents {len(cur)} | changes rows {len(ch)} ({ch['date'].min().date()}..{ch['date'].max().date()})")
    print(f"membership intervals {len(mem)} over {mem['ticker'].nunique()} tickers; "
          f"point-in-time today {len(u_now)} (== current table: {u_now == set(cur['ticker'])})")
    for d in ["2018-01-02", "2020-06-30", "2022-12-30", "2024-12-31"]:
        print(f"  universe on {d}: {len(universe_on(mem, d))}")
    if "--load" in sys.argv:
        print("db:", load_to_db(cur, mem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
