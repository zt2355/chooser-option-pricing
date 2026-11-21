"""Week 12 audits (anchor swap, cross-sectional ML reference, pre-registered tests) —
佟铸城 (Zhucheng Tong)

  audit 1   GARCH state engine: src.garch_state.path == analysis_w9.garch_anchors
            bit-for-bit on a 5-ticker sample (full histories)
  audit 2   stored GARCH columns == path() over the stored log returns (same sample).
            The stored values may have been written on the other platform (Linux container
            vs macOS host), where the optimiser stops at a slightly different point of a flat
            likelihood: measured across platforms, sigma outputs differ by <= 2.1e-5 relative
            and the parameters / state by <= 1.6e-4. Tolerances: sigma columns 1e-4 relative
            (what prices), parameters and state 1e-2. Bit-for-bit identity is asserted only
            within one platform (audits 1 and 3).
  audit 3   incremental GARCH == full: rebuild_incremental and rebuild_many reproduce
            rebuild_full's GARCH columns bit-for-bit; the "changed return" branch of
            garch_tail (a revised return inside the tail) equals a full recompute; and
            for one new ORDINARY day (no refit due) garch_tail must not read the return
            history at all yet match the full path bit-for-bit (refit day: same match)
  audit 4   pricing: primary == closed form at the quote's own anchor; anchor = GARCH-1d
            iff a GARCH state exists (else EWMA); band == regime MAE of that anchor;
            quote_universe == scalar quote on all fields for the sample
  audit 5   bands: constants in pricing_service == docs/week12/bands.json (2 dp), and the
            same code reproduces the Week 4 EWMA bands 9.01 / 10.83
  audit 6   ML reference: serving z-score == training transform (analysis_w11) bit-for-bit
            on a panel date; model sha256 == model_registry; trained on the Week 11 panel
            (479,160 rows / 518 tickers); in scope <=> point-in-time S&P 500 member with a
            GARCH anchor; sigma_ml == sigma_anchor * exp(ratio)
  audit 7   pre-registration: sha256(preregistration.json) == preregistration.sha256;
            decisions in significance.json recomputed from its own numbers with the
            pre-registered rules (procedure choice from size checks, Holm, year rule,
            IV class, ML_SCOPE consequence)
  audit 8   licence: nothing under data/wrds_local is tracked by git; iv_panel.json holds
            aggregates only (no per-date series)
  audit 9   data hygiene: no ACTIVE former member whose stored prices start after its
            index exit (ticker reuse / post-exit fragment)
  audit 10  scheduler catch-up rule on eight calendar cases (weekend, due time, never run)
  audit 11  PRIMARY_ANCHOR=ewma (subprocess) still prices JPM at the EWMA closed form with
            the Week 4 bands
  audit 12  regressions: Week 11 audits (which run Week 10 -> Week 9) and Week 7 audits pass

Run from project root with services up:  python analysis/week12/01_w12_audits.py
"""

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W12 = ROOT / "docs" / "week12"
FIELDS = ["S", "anchor", "sigma_anchor", "sigma_garch_1d", "sigma_ewma", "sigma_ml", "ml_ratio", "r", "q",
          "primary_price", "ml_reference_price", "call_leg", "put_leg", "straddle"]


def _eq(a, b) -> bool:
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def main() -> int:
    warnings.simplefilter("ignore")
    from src import garch_state as gs
    from src.analysis_w9 import garch_anchors
    from src.db import repository as repo
    from src.db.engine import db_available
    from src.db.features_db import garch_tail, rebuild_full, rebuild_incremental, rebuild_many, with_garch
    from src.tool import pricing_service as ps
    assert db_available()
    rng = np.random.default_rng(12)
    sec = repo.list_securities()
    sp = sorted(sec[sec["universe"] == "sp500"]["ticker"])
    latest = repo.read_latest_features()
    with_g = [t for t in sp if t in latest.index and pd.notna(latest.loc[t, "sigma_garch_1d"]) and t != "JPM"]
    sample = ["JPM"] + list(rng.choice(with_g, 4, replace=False))

    # audit 1
    for t in sample:
        lr = repo.read_features(t)["log_return"]
        a = garch_anchors(lr)[["sigma_garch_1d", "sigma_garch_1y"]]
        b = gs.path(lr)[["sigma_garch_1d", "sigma_garch_1y"]]
        assert a.index.equals(b.index) and _eq(a, b), t
    print(f"audit 1 PASS: garch_state.path == W9 garch_anchors bit-for-bit on {sample}")

    # audit 2
    worst_sig = worst_state = 0.0
    sig_cols = ["sigma_garch_1d", "sigma_garch_1y"]
    for t in sample:
        f = repo.read_features(t); g = with_garch(f)[gs.OUT_COLS].reindex(f.index)
        assert (f[gs.OUT_COLS].isna().to_numpy() == g.isna().to_numpy()).all(), t
        for cols, tol in ((sig_cols, 1e-4), ([c for c in gs.OUT_COLS if c not in sig_cols], 1e-2)):
            a = f[cols].to_numpy(float); b = g[cols].to_numpy(float); m = ~np.isnan(a)
            rel = float((np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-12)).max()) if m.any() else 0.0
            assert rel < tol, (t, cols[0], rel)
            if cols == sig_cols:
                worst_sig = max(worst_sig, rel)
            else:
                worst_state = max(worst_state, rel)
    print(f"audit 2 PASS: stored GARCH == recomputation from stored returns (max rel diff sigma {worst_sig:.1e}, "
          f"parameters/state {worst_state:.1e}; cross-platform optimiser digits)")

    # audit 3
    for t in sample:
        rebuild_full(t); full = repo.read_features(t)[gs.OUT_COLS]
        rebuild_incremental(t); inc = repo.read_features(t)[gs.OUT_COLS]
        assert _eq(full, inc), ("incremental", t)
    rebuild_many(sample)
    for t in sample:
        many = repo.read_features(t)[gs.OUT_COLS]
        rebuild_full(t); full = repo.read_features(t)[gs.OUT_COLS]
        assert _eq(full, many), ("many", t)
    # changed-return branch, in memory: revise one return 10 rows before the end
    t = sample[0]
    stored = repo.read_features(t)
    tail = stored.iloc[-40:].drop(columns=gs.OUT_COLS).copy()
    k = tail.index[-10]
    tail.loc[k, "log_return"] += 1e-4
    revised_full = stored["log_return"].copy(); revised_full.loc[k] += 1e-4
    got = garch_tail(tail, stored.iloc[-760:], lambda: stored["log_return"])
    want = gs.path(revised_full).reindex(tail.index)
    assert _eq(got[gs.OUT_COLS], want[gs.OUT_COLS]), "changed-return branch"
    from src.analysis_w9 import MIN_OBS, REFIT
    def _no_history():
        raise AssertionError("history read on an ordinary day")
    for t in sample:
        f = repo.read_features(t); full = gs.path(f["log_return"]); idx = full.dropna().index
        for kind in ("ordinary", "refit"):
            d = next(x for x in idx[::-1] if ((int(full.loc[x, "garch_i"]) + 1 - MIN_OBS) % REFIT == 0) == (kind == "refit"))
            st = f.loc[:d]
            prior = st.iloc[:-1].copy(); prior[gs.OUT_COLS] = full.reindex(prior.index)[gs.OUT_COLS]
            got = garch_tail(st.iloc[-40:].drop(columns=gs.OUT_COLS), prior,
                             _no_history if kind == "ordinary" else (lambda f=f: f["log_return"]))
            assert _eq(got[gs.OUT_COLS], full.reindex(got.index)[gs.OUT_COLS]), (t, kind)
    print("audit 3 PASS: rebuild_incremental / rebuild_many == rebuild_full (GARCH bit-for-bit); revised-return branch == full "
          "recompute; ordinary day continues without the history, refit day refits - both == full path")

    # audit 4
    uq = ps.quote_universe()
    for t in sample:
        st = ps.load_market_state(ticker=t, source="db")
        q = ps.quote(st, cached=False)
        direct = ps.price_chooser(st.S, st.S, st.r_1y, st.q, st.sigma_anchor, 0.5, 1.0)
        assert abs(direct - q["primary_price"]) < 1e-12, t
        assert q["anchor"] == ("garch1d" if np.isfinite(st.sigma_garch_1d) else "ewma")
        lo, hi = q["primary_band"]
        assert abs((hi - lo) / 2 - ps.BAND_MAE_BY_ANCHOR[q["anchor"]][q["regime"]]) < 1e-9
        for k_ in FIELDS:
            a, b = uq.loc[t, k_], q[k_]
            if k_ == "anchor":
                assert a == b, (t, k_)
            else:
                assert (np.isnan(a) and np.isnan(b)) or abs(float(a) - float(b)) < 1e-9, (t, k_, a, b)
    fb = uq[uq["anchor"] == "ewma"]
    assert fb["sigma_garch_1d"].isna().all() and uq.loc[uq["anchor"] == "garch1d", "sigma_garch_1d"].notna().all()
    print(f"audit 4 PASS: primary = closed form at own anchor; universe == scalar on {len(FIELDS)} fields; "
          f"{(uq['anchor'] == 'garch1d').sum()} GARCH-anchored, {len(fb)} EWMA fallback (no GARCH state)")

    # audit 5
    bands = json.load(open(W12 / "bands.json"))
    for reg in ("calm", "fear"):
        assert round(bands["garch1d"][reg], 2) == ps.BAND_MAE_BY_ANCHOR["garch1d"][reg]
        assert round(bands["ewma_full_w4_panel"][reg], 2) == ps.BAND_MAE_BY_ANCHOR["ewma"][reg]
    print(f"audit 5 PASS: band constants == bands.json; EWMA bands reproduce Week 4 "
          f"({ps.BAND_MAE_BY_ANCHOR['ewma']}) with the same code")

    # audit 6
    import src.analysis_w11 as w
    from src.ml import cross_sectional as xs
    from src.universe.sp500 import universe_on
    p = w.panel_round2().dropna(subset=["sigma_garch_1d"]).reset_index(drop=True)
    assert (len(p), p["ticker"].nunique()) == (479160, 518), (len(p), p["ticker"].nunique())
    X_tr, _, _ = w.cross_sectional(p, "sigma_garch_1d", sector_dummies=False)
    d0 = p["date"].iloc[len(p) // 2]
    m = (p["date"] == d0).to_numpy()
    b = xs.bundle()
    assert _eq(X_tr[m][b["features"]], xs.zscore_within_date(p[m], b["features"]))
    sha = hashlib.sha256(xs.MODEL_PATH.read_bytes()).hexdigest()
    import sqlalchemy as sa
    from src.db.engine import get_engine
    with get_engine().connect() as c:
        reg = c.execute(sa.text("SELECT sha256 FROM model_registry WHERE name = :n"), {"n": xs.MODEL_VERSION}).scalar()
    assert reg == sha, "model file differs from its registry entry"
    assert (b["train_rows"], b["train_tickers"]) == (479160, 518)
    cs = xs.cross_section()
    members = universe_on(repo.read_membership("sp500"), cs.attrs["date"])
    in_scope = set(uq.index[uq["ml_in_scope"]])
    expect = {t for t in uq.index if t in members and uq.loc[t, "anchor"] == "garch1d"
              and pd.Timestamp(uq.loc[t, "date"]) == cs.attrs["date"]}
    assert in_scope == expect, (len(in_scope), len(expect))
    sc = uq[uq["ml_in_scope"]]
    assert np.allclose(sc["sigma_ml"], np.maximum(sc["sigma_anchor"] * np.exp(sc["ml_ratio"]), 0.01), atol=1e-12, rtol=0)
    out = uq[~uq["ml_in_scope"]]
    assert out["ml_reference_price"].isna().all()
    print(f"audit 6 PASS: serving transform == training transform; registry sha matches; scope = {len(in_scope)} "
          f"point-in-time members with GARCH; {len(out)} out of scope have no ML price")

    # audit 7
    rec = (W12 / "preregistration.sha256").read_text().split()[0]
    assert hashlib.sha256((W12 / "preregistration.json").read_bytes()).hexdigest() == rec
    sig = json.load(open(W12 / "significance.json"))
    T = sig["tests"]
    lo_b, hi_b = 0.025, 0.075
    for k_, t_ in T.items():
        ok_t = lo_b <= t_["size_check"]["rejection_rate_at_5pct"] <= hi_b
        if ok_t:
            want_proc = "ticker bootstrap"
        else:
            ok_s = lo_b <= t_["fallback_sector_bootstrap"]["size_check"]["rejection_rate_at_5pct"] <= hi_b
            want_proc = "sector bootstrap" if ok_s else "descriptive only (both procedures mis-sized)"
        assert t_["procedure"] == want_proc, k_
    ps_ = sorted(((v["p_used"], k_) for k_, v in T.items())); mm = len(ps_); run = 0.0
    for j, (pv, k_) in enumerate(ps_):
        run = max(run, min(1.0, (mm - j) * pv)); assert abs(T[k_]["p_holm"] - run) < 1e-12
        assert T[k_]["significant_holm_5pct"] == (run < 0.05 and T[k_]["procedure"].startswith(("ticker", "sector")))
    h2 = T["H2"]; yrs = {y: v for y, v in h2["by_year"].items() if v["dates"] >= 120}
    est = h2["significant_holm_5pct"] and h2["delta_mae"] > 0 and sum(v["delta_mae"] > 0 for v in yrs.values()) >= 2
    D = sig["decisions"]
    assert D["ml_increment"] == ("ESTABLISHED" if est else "NOT ESTABLISHED")
    ivc = "research_candidate" if all(T[h]["significant_holm_5pct"] and T[h]["delta_mae"] > 0 for h in ("H3", "H4")) else "research_benchmark"
    assert D["iv_class"] == ivc and D["ml_scope_expansion"] == ("allowed" if est else "not allowed (keep ML_SCOPE = {JPM})")
    print(f"audit 7 PASS: pre-registration intact (sha {rec[:12]}); decisions follow the rules: "
          f"ML increment {D['ml_increment']}, IV {D['iv_class']}")

    # audit 8
    import shutil
    if (ROOT / ".git").exists() and shutil.which("git"):
        tracked = subprocess.run(["git", "ls-files", "data/wrds_local"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        assert tracked == "", tracked
        where = "no WRDS file tracked by git"
    else:                                   # e.g. inside the Docker image: the extract must not be there at all
        assert not (ROOT / "data" / "wrds_local").exists(), "WRDS extract present in a non-git copy (image/zip)"
        where = "no WRDS extract in this copy"
    ivp = json.load(open(W12 / "iv_panel.json"))
    def _walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from _walk(v)
        elif isinstance(x, list):
            yield len(x)
    assert all(n < 50 for n in _walk(ivp)), "iv_panel.json must hold aggregates only"
    print(f"audit 8 PASS: {where}; iv_panel.json holds aggregates only")

    # audit 9
    from src.universe.sp500 import post_exit_tickers
    bad = [t for t in post_exit_tickers() if t in set(sec["ticker"])]
    assert not bad, bad
    print(f"audit 9 PASS: no active former member with post-exit-only prices ({len(post_exit_tickers())} kept inactive)")

    # audit 10
    from zoneinfo import ZoneInfo
    from src.service import scheduler as s
    ny = ZoneInfo("America/New_York"); U = dt.timezone.utc
    T_ = lambda *a: dt.datetime(*a, tzinfo=ny)
    cases = [(T_(2026, 10, 2, 17, 29), T_(2026, 10, 1, 17, 31), False), (T_(2026, 10, 2, 17, 31), T_(2026, 10, 1, 17, 31), True),
             (T_(2026, 10, 2, 17, 40), T_(2026, 10, 2, 17, 35), False), (T_(2026, 10, 3, 12, 0), T_(2026, 10, 2, 17, 35), False),
             (T_(2026, 10, 4, 12, 0), T_(2026, 10, 1, 17, 35), True), (T_(2026, 10, 5, 9, 0), T_(2026, 10, 2, 17, 35), False),
             (T_(2026, 10, 5, 23, 0), None, True), (T_(2026, 10, 5, 17, 30), T_(2026, 10, 2, 17, 35), True)]
    for now, last, exp in cases:
        assert s.needs_run(now.astimezone(U), last.astimezone(U) if last else None) == exp, (now, last)
    print("audit 10 PASS: scheduler catch-up rule correct on 8 calendar cases")

    # audit 11
    code = ("import sys; sys.path.insert(0, '.'); from src.tool import pricing_service as ps;"
            "st = ps.load_market_state(ticker='JPM', source='db'); q = ps.quote(st, cached=False);"
            "d = ps.price_chooser(st.S, st.S, st.r_1y, st.q, st.sigma_ewma, 0.5, 1.0);"
            "assert q['anchor'] == 'ewma' and abs(q['primary_price'] - d) < 1e-12;"
            "lo, hi = q['primary_band']; assert abs((hi - lo) / 2 - {'calm': 9.01, 'fear': 10.83}[q['regime']]) < 1e-9; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PRIMARY_ANCHOR": "ewma"})
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-500:]
    print("audit 11 PASS: PRIMARY_ANCHOR=ewma prices JPM at the EWMA closed form with the Week 4 bands")

    # audit 12
    for script, tag in (("analysis/week11/02_perf_audits.py", "All audits passed"), ("analysis/week7/01_tool_audits.py", "All audits passed")):
        r = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0 and tag in r.stdout, script + "\n" + r.stdout[-800:] + r.stderr[-800:]
    print("audit 12 PASS: Week 11 (-> Week 10 -> Week 9) and Week 7 audits still pass")

    print("\nAll audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
