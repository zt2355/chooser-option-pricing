"""Week 12 theory line — answers to the mentor's two questions (2026-09-30), run
exactly as pre-registered in docs/week12/preregistration.json (SHA-256 in HANDOFF).

Part A  Significance bound of the ML increment.
        Walk-forward out-of-sample (OOS) predictions of the Week 11 model
        (GARCH-1d anchor + cross-sectional GBDT) for EVERY quarterly block from
        the first one whose purged training set has >= 252 dates, through
        2024-12-31. Primary model: WITHOUT sector dummies (they are look-ahead,
        W11 Part D); the W11 reported model (with dummies) is secondary.
          H1  Delta_MAE(EWMA, GARCH-1d)          anchor swap
          H2  Delta_MAE(GARCH-1d, GARCH-1d + ML) ML increment (primary)
Part B  Implied volatility as an anchor on the S&P 500 panel (OptionMetrics,
        local only; src/data_collection/wrds_optionm.py), common rows where
        EWMA, GARCH-1d, GARCH-1y, 252d realised and IV all exist.
          H3  Delta_MAE(GARCH-1d, IV)
          H4  Delta_MAE(S, IV), S = the free smooth anchor (252d realised or
              GARCH-1y) with the LOWER MAE on those rows (conservative for IV)
Inference  ticker-cluster bootstrap (B=2000, seed 12), conditional on the realised
        time path; each hypothesis' procedure is first SIZE-CHECKED on 300
        simulated null panels (sector-day AR(1) factor + ticker AR(1) noise
        fitted to that hypothesis' own loss differences, mean forced to 0);
        Holm over {H1..H4}. Time dimension: calendar years, descriptive only.

Part C  Regime error bands for the tool's new primary anchor (GARCH-1d), Week 4
        definition, with the old EWMA bands reproduced by the same code.

Outputs (docs/week12/): significance.json, iv_panel.json, bands.json, yearly.csv,
  figures/increment_by_year.png, figures/iv_panel_mae.png
  (figures show derived quantities only - never the IV series).

Usage: python -m src.analysis_w12
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.analysis_w10 import price
from src.metrics import composite_score, metric_suite

OUT = config.PROJECT_ROOT / "docs" / "week12"
FIG = OUT / "figures"
B, SEED, N_SIM = 2000, 12, 300
SIZE_BAND = (0.025, 0.075)
BLUE, ORANGE, GREEN, GRAY, INK, GRID = "#2f6db3", "#d1603d", "#3a7d44", "#8a8a8a", "#333333", "#d9d9d9"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK); ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ inference

def _ticker_sums(d: pd.Series, ticker: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    g = pd.DataFrame({"d": d.to_numpy(float), "t": ticker.to_numpy()}).groupby("t")["d"]
    return g.sum().to_numpy(), g.count().to_numpy(float)


def _weights(n_tickers: int, rng) -> np.ndarray:
    return rng.multinomial(n_tickers, np.full(n_tickers, 1.0 / n_tickers), size=B).astype(float)


def bootstrap(d: pd.Series, ticker: pd.Series, seed: int = SEED) -> dict:
    """Ticker-cluster bootstrap of the pooled mean of d (= Delta_MAE)."""
    S, n = _ticker_sums(d, ticker)
    W = _weights(len(S), np.random.default_rng(seed))
    stats = (W @ S) / (W @ n)
    est = float(S.sum() / n.sum())
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {"delta_mae": est, "ci95": [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))],
            "p_boot": float(max(p, 1.0 / B)), "tickers": int(len(S)), "rows": int(n.sum())}


def size_check(d: pd.Series, ticker: pd.Series, sector: pd.Series, date: pd.Series, seed: int,
               procedure: str = "ticker") -> dict:
    """Rejection rate at nominal 5% of `procedure` ("ticker" or "sector" bootstrap) on null
    panels with this hypothesis' own dependence."""
    df = pd.DataFrame({"d": d.to_numpy(float), "t": ticker.to_numpy(), "g": sector.to_numpy(), "date": date.to_numpy()})
    df["d"] -= df["d"].mean()
    sg = df.groupby(["g", "date"])["d"].transform("mean")
    df["u"] = df["d"] - sg
    s_tab = df.groupby(["g", "date"])["d"].mean().unstack("date")              # sector x date
    n_tab = df.groupby(["g", "date"])["d"].count().unstack("date")
    u_tab = df.pivot_table(index="t", columns="date", values="u")
    phi_u = float(np.nanmedian([r.autocorr(1) for _, r in u_tab.iterrows() if r.notna().sum() > 30]))
    sd_u = float(np.nanstd(u_tab.to_numpy()))
    # a sector-date mean contains the ticker noise averaged over n tickers (variance sd_u^2/n,
    # lag-1 autocovariance phi_u sd_u^2/n): remove it, otherwise the null panels carry a sector
    # correlation that is not in the data (meta-test: 12.5% size on pure iid noise before this fix)
    noise_var = float(np.nanmean((sd_u ** 2) / n_tab.to_numpy()))
    sv = s_tab.to_numpy()
    var_s = max(float(np.nanvar(sv)) - noise_var, 0.0)
    acov1 = float(np.nanmean((sv[:, 1:] - np.nanmean(sv)) * (sv[:, :-1] - np.nanmean(sv))))
    phi_s = float(np.clip((acov1 - phi_u * noise_var) / var_s, -0.99, 0.99)) if var_s > 0 else 0.0
    sd_s = float(np.sqrt(var_s))
    dates = np.array(sorted(df["date"].unique())); T = len(dates); dpos = {x: i for i, x in enumerate(dates)}
    gcodes, gidx = np.unique(df["g"].to_numpy(), return_inverse=True)
    tcodes, tidx = np.unique(df["t"].to_numpy(), return_inverse=True)
    di = df["date"].map(dpos).to_numpy()
    rng = np.random.default_rng(seed)
    W = _weights(len(tcodes), rng)
    t_sector = pd.Series(gidx).groupby(tidx).first().to_numpy()              # sector of each ticker code
    rej = 0
    for _ in range(N_SIM):
        s = np.empty((len(gcodes), T)); u = np.empty((len(tcodes), T))
        s[:, 0] = rng.normal(0, sd_s, len(gcodes)); u[:, 0] = rng.normal(0, sd_u, len(tcodes))
        es, eu = np.sqrt(max(1 - phi_s ** 2, 1e-6)) * sd_s, np.sqrt(max(1 - phi_u ** 2, 1e-6)) * sd_u
        for k in range(1, T):
            s[:, k] = phi_s * s[:, k - 1] + rng.normal(0, es, len(gcodes))
            u[:, k] = phi_u * u[:, k - 1] + rng.normal(0, eu, len(tcodes))
        dsim = s[gidx, di] + u[tidx, di]
        S = np.bincount(tidx, weights=dsim, minlength=len(tcodes)); n = np.bincount(tidx, minlength=len(tcodes)).astype(float)
        stats = (W @ S) / (W @ n) if procedure == "ticker" else _sector_stats(S, n, t_sector, rng)
        lo, hi = np.quantile(stats, .025), np.quantile(stats, .975)
        rej += int(lo > 0 or hi < 0)
    rate = rej / N_SIM
    return {"procedure": procedure, "rejection_rate_at_5pct": rate, "accepted": bool(SIZE_BAND[0] <= rate <= SIZE_BAND[1]),
            "fitted": {"phi_sector": phi_s, "sd_sector": sd_s, "phi_ticker": phi_u, "sd_ticker": sd_u, "noise_var_removed": noise_var}, "n_sim": N_SIM}


def _sector_stats(S: np.ndarray, n: np.ndarray, t_sector: np.ndarray, rng) -> np.ndarray:
    """B draws of the two-stage (sectors, then tickers within) bootstrap mean."""
    G = int(t_sector.max()) + 1
    draws = rng.integers(0, G, size=(B, G))                       # drawn sector for each of G slots
    num = np.zeros(B); den = np.zeros(B)
    for g in range(G):
        members = np.where(t_sector == g)[0]
        rows, cols = np.where(draws == g)
        if not len(rows) or not len(members):
            continue
        w = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)), size=len(rows)).astype(float)
        np.add.at(num, rows, w @ S[members]); np.add.at(den, rows, w @ n[members])
    return num / den


def sector_bootstrap(d: pd.Series, ticker: pd.Series, sector: pd.Series, seed: int = SEED) -> dict:
    """Pre-registered fallback: resample sectors, then tickers within the drawn sectors."""
    df = pd.DataFrame({"d": d.to_numpy(float), "t": ticker.to_numpy(), "g": sector.to_numpy()})
    per_t = df.groupby(["t"]).agg(S=("d", "sum"), n=("d", "count"), g=("g", "first"))
    _, t_sector = np.unique(per_t["g"].to_numpy(), return_inverse=True)
    stats = _sector_stats(per_t["S"].to_numpy(), per_t["n"].to_numpy(float), t_sector, np.random.default_rng(seed))
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {"ci95": [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))], "p_boot": float(max(p, 1.0 / B)),
            "clusters": int(t_sector.max() + 1)}


def size_check_self_test(seed: int = 7) -> dict:
    """The size check is itself a self-written procedure: verify it on synthetic panels with a
    KNOWN answer before using it (300 tickers x 250 days, 12 sectors, true mean 0).
      iid noise            -> the ticker bootstrap must be about right (~5%)
      strong sector shocks -> the ticker bootstrap must be flagged (over-rejects) and the
                              sector bootstrap must come back near nominal
    The first version of size_check failed this test (12.5% on pure noise: sector-date means
    carry ticker noise of variance sd_u^2/n); the noise-variance correction fixed it."""
    rng = np.random.default_rng(seed)
    T, N, G = 250, 300, 12
    dates = pd.date_range("2023-01-02", periods=T, freq="B")
    t_lab = np.repeat([f"T{i}" for i in range(N)], T); g_lab = np.repeat([f"S{i % G}" for i in range(N)], T)
    d_lab = np.tile(dates, N)
    gi = np.repeat(np.arange(N) % G, T); ti = np.repeat(np.arange(N), T); di = np.tile(np.arange(T), N)
    out = {}
    for name, sd_s in (("iid_noise", 0.0), ("strong_sector_shocks", 3.0)):
        s_ = rng.normal(0, sd_s, (G, T)); u = rng.normal(0, 1, (N, T))
        d = pd.Series(s_[gi, di] + u[ti, di])
        out[name] = {"ticker_bootstrap_size": size_check(d, pd.Series(t_lab), pd.Series(g_lab), pd.Series(d_lab), seed, "ticker")["rejection_rate_at_5pct"],
                     "sector_bootstrap_size": size_check(d, pd.Series(t_lab), pd.Series(g_lab), pd.Series(d_lab), seed + 1, "sector")["rejection_rate_at_5pct"]}
    return out


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, run = {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p)); out[k] = run
    return out


def test_hypothesis(name: str, d: pd.Series, rows: pd.DataFrame, seed: int) -> dict:
    res = {"hypothesis": name, **bootstrap(d, rows["ticker"])}
    res["size_check"] = size_check(d, rows["ticker"], rows["sector"], rows["date"], seed)
    if res["size_check"]["accepted"]:
        res["procedure"] = "ticker bootstrap"
    else:
        fb = sector_bootstrap(d, rows["ticker"], rows["sector"])
        fb["size_check"] = size_check(d, rows["ticker"], rows["sector"], rows["date"], seed + 1, procedure="sector")
        res["fallback_sector_bootstrap"] = fb
        res["procedure"] = "sector bootstrap" if fb["size_check"]["accepted"] else "descriptive only (both procedures mis-sized)"
        res["ci95_used"], res["p_used"] = fb["ci95"], fb["p_boot"]
    res.setdefault("ci95_used", res["ci95"]); res.setdefault("p_used", res["p_boot"])
    by_year = rows.assign(d=d.to_numpy()).groupby(rows["date"].dt.year)
    res["by_year"] = {int(y): {"delta_mae": float(g["d"].mean()), "dates": int(g["date"].nunique()), "rows": int(len(g))}
                      for y, g in by_year}
    return res


# ------------------------------------------------------------------ Part A

def part_a() -> tuple[dict, pd.DataFrame]:
    import src.analysis_w11 as w
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = w.panel_round2().dropna(subset=["sigma_garch_1d"]).reset_index(drop=True)
    dates = np.array(sorted(p["date"].unique()))
    s0 = 252 + w.HORIZON + 1                       # first block whose purged training set has 252 dates
    period = dates[s0:]
    mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=200, random_state=0)
    preds = {}
    for tag, dummies in (("ml_nodummies", False), ("ml_w11", True)):
        X, y_cs, _ = w.cross_sectional(p, "sigma_garch_1d", sector_dummies=dummies)
        preds[tag], _, _ = w.walk_forward(p, X, y_cs, dates, period, mk, False)
    oos = p[p["date"].isin(period) & preds["ml_nodummies"].notna()].copy()
    po = price(oos, "sigma_oracle")
    oos["err_ewma"] = np.abs(price(oos, "sigma_ewma") - po)
    oos["err_garch1d"] = np.abs(price(oos, "sigma_garch_1d") - po)
    for tag in preds:
        oos[f"sigma_{tag}"] = np.maximum(oos["sigma_garch_1d"] * np.exp(preds[tag].loc[oos.index].to_numpy()), 0.01)
        oos[f"err_{tag}"] = np.abs(price(oos, f"sigma_{tag}") - po)
    rg = oos["fear_regime"].reset_index(drop=True); pos = pd.Series(po)
    ref = metric_suite(pd.Series(price(oos, "sigma_ewma")), pos, rg)
    comps = {c: composite_score(metric_suite(pd.Series(price(oos, c)), pos, rg), ref)
             for c in ("sigma_garch_1d", "sigma_ml_nodummies", "sigma_ml_w11")}
    info = {"oos_start": str(pd.Timestamp(period[0]).date()), "oos_end": str(pd.Timestamp(period[-1]).date()),
            "oos_dates": int(oos["date"].nunique()), "oos_rows": int(len(oos)), "tickers": int(oos["ticker"].nunique()),
            "first_block_training_dates": int(s0 - w.HORIZON - 1),
            "mae": {k: float(oos[f"err_{k}"].mean()) for k in ("ewma", "garch1d", "ml_nodummies", "ml_w11")},
            "composite_vs_ewma_descriptive": comps}
    return info, oos


# ------------------------------------------------------------------ Part B

def part_b() -> tuple[dict, pd.DataFrame]:
    from src.analysis_w10 import build_panel
    from src.db import repository as repo
    ivp = config.PROJECT_ROOT / "data" / "wrds_local" / "optionm_sp500_iv365.parquet"
    if not ivp.exists():
        return {"skipped": "local OptionMetrics extract not present (licence: research use on the author's machine)"}, pd.DataFrame()
    p = build_panel()
    g = pd.read_parquet(config.PROJECT_ROOT / "data" / "processed" / "garch_anchors_sp500.parquet"); g["date"] = pd.to_datetime(g["date"])
    iv = pd.read_parquet(ivp); iv["date"] = pd.to_datetime(iv["date"])
    p = p.merge(g[["ticker", "date", "sigma_garch_1d", "sigma_garch_1y"]], on=["ticker", "date"], how="inner")
    p = p.merge(iv, on=["ticker", "date"], how="inner")
    p = p.dropna(subset=["sigma_ewma", "sigma_garch_1d", "sigma_garch_1y", "realized_vol_252d", "sigma_iv"]).reset_index(drop=True)
    sec = repo.list_securities(active_only=False).set_index("ticker")["sector"]
    p["sector"] = p["ticker"].map(sec).fillna("Unknown")
    po = price(p, "sigma_oracle")
    anchors = {"ewma": "sigma_ewma", "garch1d": "sigma_garch_1d", "garch1y": "sigma_garch_1y",
               "rv252": "realized_vol_252d", "iv": "sigma_iv"}
    for k, c in anchors.items():
        p[f"err_{k}"] = np.abs(price(p, c) - po)
    mae = {k: float(p[f"err_{k}"].mean()) for k in anchors}
    S = "rv252" if mae["rv252"] < mae["garch1y"] else "garch1y"
    rg = p["fear_regime"].reset_index(drop=True); pos = pd.Series(po)
    ref = metric_suite(pd.Series(price(p, "sigma_ewma")), pos, rg)
    comps = {k: composite_score(metric_suite(pd.Series(price(p, c)), pos, rg), ref) for k, c in anchors.items()}
    # per-ticker dispersion of each anchor (the W11 "smoothness" reading, now on the panel)
    disp = {k: float(p.groupby("ticker")[c].std().median()) for k, c in {**anchors, "oracle": "sigma_oracle"}.items()}
    win = {}
    for other in ("garch1d", S):
        t = p.groupby("ticker").apply(lambda x: x[f"err_iv"].mean() < x[f"err_{other}"].mean())
        win[other] = float(t.mean())
    # mechanism (post-hoc, DESCRIPTIVE, look-ahead diagnostics - not part of the pre-registered tests):
    # is IV's edge its per-ticker LEVEL or its TIMING? 2x2 = {IV, GARCH-1y} x {actual path,
    # constant at the ticker's own sample mean}; plus within-ticker time-series correlation with the oracle.
    mech = {}
    for k, c in (("iv", "sigma_iv"), ("garch1y", "sigma_garch_1y"), ("garch1d", "sigma_garch_1d")):
        p[f"{c}_tmean"] = p.groupby("ticker")[c].transform("mean")
        mech[f"mae_{k}_constant_at_ticker_mean"] = float(np.abs(price(p, f"{c}_tmean") - po).mean())
        mech[f"median_ts_corr_{k}_vs_oracle"] = float(p.groupby("ticker").apply(lambda x: x[c].corr(x["sigma_oracle"])).median())
    mech["xs_corr_ticker_mean_iv_vs_oracle"] = float(p.groupby("ticker")[["sigma_iv", "sigma_oracle"]].mean().corr().iloc[0, 1])
    mech["xs_corr_ticker_mean_garch1y_vs_oracle"] = float(p.groupby("ticker")[["sigma_garch_1y", "sigma_oracle"]].mean().corr().iloc[0, 1])
    mech["note"] = "constant-at-ticker-mean anchors use the whole sample (look-ahead): diagnostic only"
    info = {"rows": int(len(p)), "tickers": int(p["ticker"].nunique()), "start": str(p["date"].min().date()),
            "end": str(p["date"].max().date()), "mae": mae, "free_smooth_comparator": S,
            "composite_vs_ewma_descriptive": comps, "median_within_ticker_std": disp,
            "iv_ticker_win_rate_vs": win, "mechanism_descriptive": mech}
    return info, p


# ------------------------------------------------------------------ Part C

def part_c() -> dict:
    """Error bands for the tool after the anchor swap (Week 4 definition: MAE of the ATM
    chooser price by VIX regime on the JPM oracle panel). The EWMA bands of the tool
    (9.01 / 10.83) are first reproduced with the same code to show the definition is
    unchanged; GARCH-1d bands are computed on the rows where GARCH exists."""
    from src import garch_state as gs
    from src.analysis_w4 import T1, T2, build_panel as jpm_panel
    from src.models.chooser import chooser_closed_form
    jp = jpm_panel()
    full = pd.read_csv(config.DATA_PROCESSED / "features_daily.csv", index_col="date", parse_dates=True)
    jp = jp.join(gs.path(full["log_return"])[["sigma_garch_1d"]], how="left")
    def bands(df, col):
        pr = np.asarray(chooser_closed_form(S0=df["jpm_close"].to_numpy(), K=df["jpm_close"].to_numpy(), r=df["r_1y"].to_numpy(),
                                            q=df["div_yield_ttm"].to_numpy(), sigma=df[col].to_numpy(), t1=T1, T2=T2), float)
        po = np.asarray(chooser_closed_form(S0=df["jpm_close"].to_numpy(), K=df["jpm_close"].to_numpy(), r=df["r_1y"].to_numpy(),
                                            q=df["div_yield_ttm"].to_numpy(), sigma=df["sigma_oracle"].to_numpy(), t1=T1, T2=T2), float)
        e = pd.Series(np.abs(pr - po), index=df.index)
        return {r: float(e[df["fear_regime"] == v].mean()) for r, v in (("calm", 0), ("fear", 1))}
    g = jp.dropna(subset=["sigma_garch_1d"])
    return {"ewma_full_w4_panel": {**bands(jp, "sigma_ewma"), "days": int(len(jp))},
            "ewma_same_window_as_garch": {**bands(g, "sigma_ewma"), "days": int(len(g))},
            "garch1d": {**bands(g, "sigma_garch_1d"), "days": int(len(g)), "start": str(g.index.min().date()), "end": str(g.index.max().date())}}


# ------------------------------------------------------------------ main

def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    a_info, oos = part_a()
    print(f"[A] OOS {a_info['oos_start']}..{a_info['oos_end']} ({a_info['oos_dates']} dates, {a_info['oos_rows']} rows, "
          f"{a_info['tickers']} tickers); MAE " + ", ".join(f"{k} {v:.3f}" for k, v in a_info["mae"].items()))
    b_info, ivp = part_b()
    tests = {
        "H1": test_hypothesis("anchor swap: EWMA -> GARCH-1d", oos["err_ewma"] - oos["err_garch1d"], oos, 101),
        "H2": test_hypothesis("ML increment (no sector dummies)", oos["err_garch1d"] - oos["err_ml_nodummies"], oos, 102),
    }
    secondary = {"H2b": {**bootstrap(oos["err_garch1d"] - oos["err_ml_w11"], oos["ticker"]),
                         "note": "W11 reported model (with look-ahead sector dummies); not in the family"}}
    if len(ivp):
        S = b_info["free_smooth_comparator"]
        tests["H3"] = test_hypothesis("IV vs GARCH-1d", ivp["err_garch1d"] - ivp["err_iv"], ivp, 103)
        tests["H4"] = test_hypothesis(f"IV vs free smooth ({S})", ivp[f"err_{S}"] - ivp["err_iv"], ivp, 104)
    adj = holm({k: v["p_used"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
        tests[k]["significant_holm_5pct"] = bool(adj[k] < 0.05 and tests[k]["procedure"] != "descriptive only (both procedures mis-sized)")
    # ---- decisions, exactly as pre-registered
    h2 = tests["H2"]; years = {y: v for y, v in h2["by_year"].items() if v["dates"] >= 120}
    pos_years = sum(v["delta_mae"] > 0 for v in years.values())
    ml_established = bool(h2["significant_holm_5pct"] and h2["delta_mae"] > 0 and pos_years >= 2)
    if len(ivp):
        h3, h4 = tests["H3"], tests["H4"]
        sig = h3["significant_holm_5pct"] and h4["significant_holm_5pct"] and h3["delta_mae"] > 0 and h4["delta_mae"] > 0
        iv_class = "research_candidate" if sig else "research_benchmark"     # deployability not verifiable (pre-declared)
    else:
        iv_class = "not evaluated (no local OptionMetrics extract)"
    decisions = {"ml_increment": "ESTABLISHED" if ml_established else "NOT ESTABLISHED",
                 "ml_increment_years_used": sorted(years), "ml_increment_positive_years": int(pos_years),
                 "iv_class": iv_class, "deployability": "NOT VERIFIED (pre-declared: no same-day overlap between OptionMetrics, ending 2025-08-29, and live Yahoo chains)",
                 "ml_scope_expansion": "allowed" if ml_established else "not allowed (keep ML_SCOPE = {JPM})"}
    self_test = size_check_self_test()
    sig_out = {"preregistration": "docs/week12/preregistration.json", "B": B, "seed": SEED, "part_a": a_info,
               "size_check_self_test": self_test,
               "tests": tests, "secondary": secondary, "decisions": decisions}
    (OUT / "significance.json").write_text(json.dumps(sig_out, indent=2, ensure_ascii=False, default=str))
    (OUT / "iv_panel.json").write_text(json.dumps(b_info, indent=2, ensure_ascii=False, default=str))
    c_info = part_c()
    (OUT / "bands.json").write_text(json.dumps(c_info, indent=2, ensure_ascii=False, default=str))
    print("[C] bands:", json.dumps(c_info))
    yr = [{"hypothesis": k, "year": y, **v} for k, t in tests.items() for y, v in t["by_year"].items()]
    pd.DataFrame(yr).to_csv(OUT / "yearly.csv", index=False)
    # ---- figures (derived quantities only)
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=200)
    # only years with >= 120 dates, the pre-registered rule (yearly.csv keeps all years with their counts)
    labs = [f"{k} {y}" for k, t in tests.items() for y, v in t["by_year"].items() if v["dates"] >= 120]
    vals = [v["delta_mae"] for t in tests.values() for v in t["by_year"].values() if v["dates"] >= 120]
    ax.barh(labs[::-1], vals[::-1], color=[GREEN if v > 0 else ORANGE for v in vals[::-1]]); ax.axvline(0, color=INK, lw=0.8)
    style(ax, "Delta MAE by calendar year, years with >= 120 dates (positive = second model better)", "Delta MAE ($)", "")
    fig.tight_layout(); fig.savefig(FIG / "increment_by_year.png"); plt.close(fig)
    if len(ivp):
        fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=200)
        ks = list(b_info["mae"]); ax.bar(ks, [b_info["mae"][k] for k in ks], color=[BLUE if k != "iv" else GREEN for k in ks])
        for i, k in enumerate(ks):
            ax.text(i, b_info["mae"][k] + 0.05, f"{b_info['mae'][k]:.2f}", ha="center", fontsize=8, color=INK)
        style(ax, f"S&P 500 panel, common rows ({b_info['tickers']} tickers): MAE by anchor", "", "MAE ($)")
        fig.tight_layout(); fig.savefig(FIG / "iv_panel_mae.png"); plt.close(fig)
    for k, t in tests.items():
        print(f"{k} {t['hypothesis']}: Delta_MAE {t['delta_mae']:+.3f} CI {t['ci95_used'][0]:+.3f}..{t['ci95_used'][1]:+.3f} "
              f"| size {t['size_check']['rejection_rate_at_5pct']:.3f} -> {t['procedure']} | p_holm {t['p_holm']:.4f} | years "
              + ", ".join(f"{y}:{v['delta_mae']:+.3f}" for y, v in t["by_year"].items()))
    print("decisions:", decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
