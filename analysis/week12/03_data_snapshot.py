"""Snapshot of database facts cited by the final report (Week 12) — 佟铸城 (Zhucheng Tong)

Some numbers the report cites live only in the database (pipeline timings, coverage of
the point-in-time universe). This script freezes them into docs/week12/db_snapshot.json
together with the data version, so every number in the report traces to a file.

  coverage   members during the study window (2019-10-17..2024-12-31); members with no
             stored prices (delisted before collection: survivorship); ACTIVE members whose
             history is truncated (< 600 rows although listed long before: collection
             failures); former members kept inactive because their prices start after
             index exit (ticker reuse / post-exit fragments); point-in-time membership size
  runs       the pipeline runs cited in Weeks 10-12 (ids fixed below) with stage timings
  tool       anchor / ML-scope counts of the latest stored quotes

Run from project root with services up:  python analysis/week12/03_data_snapshot.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[2]
CITED_RUNS = {"w10_590_before": 15, "w11_1517_backfill": 51, "w11_1517_steady": 52, "w11_590_after": 58,
              "w12_1510_catchup_6days": 91}
STUDY = ("2019-10-17", "2024-12-31")


def main() -> int:
    from src.db import repository as repo
    from src.db.engine import get_engine
    from src.universe.sp500 import post_exit_tickers, universe_on
    mem = repo.read_membership("sp500")
    sec = repo.list_securities(active_only=False).set_index("ticker")
    with get_engine().connect() as c:
        st = pd.read_sql(sa.text("SELECT ticker, count(*) AS n, min(date) AS d0 FROM prices_daily GROUP BY ticker"), c).set_index("ticker")
        runs = pd.read_sql(sa.text("SELECT id, job, started_at, finished_at, status, tickers, timings FROM pipeline_runs WHERE id = ANY(:i)"),
                           c, params={"i": list(CITED_RUNS.values())}).set_index("id")
        dv = repo.data_version()
    in_study = mem[(mem["start"] <= STUDY[1]) & (mem["end"].isna() | (mem["end"] >= STUDY[0]))]["ticker"].unique()
    no_prices = sorted(t for t in in_study if t not in st.index)
    truncated = sorted(t for t in in_study if t in st.index and st.loc[t, "n"] < 600 and bool(sec.loc[t, "active"])
                       and pd.Timestamp(mem[mem["ticker"] == t]["start"].min()) < pd.Timestamp("2021-01-01"))
    current = universe_on(mem, pd.Timestamp.today().normalize())
    coverage = {"membership_intervals": int(len(mem)), "membership_tickers": int(mem["ticker"].nunique()),
                "members_in_study_window": int(len(in_study)), "members_without_prices": len(no_prices),
                "active_members_truncated_history": truncated,
                "current_members_truncated": [t for t in truncated if t in current],
                "post_exit_inactive": post_exit_tickers(),
                "active_securities": int(sec["active"].sum()),
                "by_universe_active": {k: int(v) for k, v in sec[sec["active"]].groupby("universe").size().items()}}
    run_out = {}
    for name, rid in CITED_RUNS.items():
        if rid in runs.index:
            r = runs.loc[rid]
            tm = json.loads(r["timings"]) if r["timings"] else {}
            run_out[name] = {"id": rid, "job": r["job"], "tickers": int(r["tickers"]), "status": r["status"],
                             "total_s": round((r["finished_at"] - r["started_at"]).total_seconds(), 1),
                             "stages": {k: v for k, v in tm.items() if k != "sources"}}
    uq = repo.read_universe_quotes(); d = uq[uq["date"] == uq["date"].max()]
    tool = {"as_of": str(d["date"].max().date()), "quoted": int(len(d)),
            "anchor_counts": {k: int(v) for k, v in d["anchor"].value_counts().items()},
            "ml_reference_count": int(d["ml_reference_price"].notna().sum())}
    import subprocess
    jpm_file = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", "data/processed/features_daily.csv"],
                              cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"data_version": dv, "jpm_file_version": jpm_file or "uncommitted", "coverage": coverage, "cited_runs": run_out, "tool": tool}
    (ROOT / "docs" / "week12" / "db_snapshot.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
