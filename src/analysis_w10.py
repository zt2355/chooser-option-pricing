"""Week 10 theory line on the S&P 500 panel.

Part A  GARCH(1,1) term-structure anchor vs EWMA, re-checked across the
        point-in-time S&P 500 universe (Week 9 found it wins on JPM alone).
        Same protocol as Week 9 per ticker (21-day refits, >= 750 obs), same
        Week 4 metric suite; the reference for every ticker is ITS OWN EWMA.
Part B  Anchored residual learning, round 1, on the panel (ticker x date):
        target y = ln(sigma_oracle / sigma_EWMA), Week 6 feature set,
        purged chronological split by DATE (70/15/15, 252-row purge),
        quarterly walk-forward refits inside the VALIDATION period only.
        The test period is sealed: no row with date > val_end is read by any
        model in this script (audit 10 checks the JSON).

Panel: every (ticker, date) with the ticker a member of the index on that
date, date in the Week 4 study window [2019-10-01, 2024-12-31], oracle =
realised vol over the following 252 trading days.

Outputs (docs/week10/): panel_summary.json, garch_panel_metrics.csv,
  ml_panel_metrics.csv, figures/garch_panel_composite_hist.png,
  figures/garch_panel_by_regime.png, figures/ml_panel_composite.png

Usage: python -m src.analysis_w10  [--workers N]
"""

import json
import sys
import warnings
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.analysis_w4 import STUDY_END, STUDY_START, T1, T2
from src.analysis_w9 import garch_anchors
from src.metrics import composite_score, metric_suite
from src.models.chooser import chooser_closed_form

OUT = config.PROJECT_ROOT / "docs" / "week10"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN, GRAY = "#2f6db3", "#d1603d", "#3a7d44", "#8a8a8a"
INK, GRID = "#333333", "#d9d9d9"
HORIZON, BLOCK, RATIO_CLIP = 252, 63, 1.5
ML_FEATURES = ["log_return", "realized_vol_21d", "realized_vol_63d", "realized_vol_252d",
               "div_yield_ttm", "div_growth_yoy", "curve_slope", "rate_momentum_21d",
               "spx_log_return", "beta_63d", "corr_vix_63d", "vix_level",
               "vix_sentiment_pct", "vix_momentum_5d", "fear_regime", "r_1y_dev"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK); ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


MIN_ROWS = 252          # per-ticker minimum for a meaningful composite


def safe_composite(m: dict, ref: dict) -> float:
    """composite_score, but NaN when the reference has a zero metric (a ticker
    with too few rows can have Hit10 == 1 or a single regime); such tickers
    are excluded from win rates and their count is reported."""
    from src.metrics import _lower_better
    if any(v == 0 for v in _lower_better(ref).values()):
        return float("nan")
    return composite_score(m, ref)


def price(df: pd.DataFrame, sigma_col: str) -> np.ndarray:
    return np.asarray(chooser_closed_form(S0=df["close"].to_numpy(float), K=df["close"].to_numpy(float),
                                          r=df["r_1y"].to_numpy(float), q=df["div_yield_ttm"].to_numpy(float),
                                          sigma=df[sigma_col].to_numpy(float), t1=T1, T2=T2), float)


# ------------------------------------------------------------------ panel

def build_panel() -> pd.DataFrame:
    from src.db import repository as repo
    from src.db import schema
    from src.universe.sp500 import universe_on
    mem = repo.read_membership("sp500")
    tickers = sorted(set(mem["ticker"]) & set(repo.list_securities()["ticker"]))
    frames = []
    for t in tickers:
        # Week 12: features_daily now also stores the tool's GARCH state; the research
        # panel keeps its Week 10 definition (anchors come from the saved parquet)
        f = repo.read_features(t).drop(columns=schema.GARCH_COLUMNS, errors="ignore")
        if len(f) < 600:
            continue
        f = f.copy(); f["ticker"] = t
        f["sigma_oracle"] = f["realized_vol_252d"].shift(-HORIZON)
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel[(panel["date"] >= STUDY_START) & (panel["date"] <= STUDY_END)]
    # point-in-time membership
    keep = np.zeros(len(panel), dtype=bool)
    for t, g in panel.groupby("ticker"):
        ivs = mem[mem["ticker"] == t]
        m = np.zeros(len(g), dtype=bool)
        for _, iv in ivs.iterrows():
            m |= (g["date"] >= iv["start"]).to_numpy() & (iv["end"] is pd.NaT or pd.isna(iv["end"]) or (g["date"] < iv["end"]).to_numpy())
        keep[panel.index.get_indexer(g.index)] = m
    panel = panel[keep]
    need = ["close", "sigma_ewma", "sigma_oracle", "r_1y", "div_yield_ttm", "fear_regime"] + ML_FEATURES
    panel = panel.dropna(subset=need).sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


# ------------------------------------------------------------------ part A

def _garch_one(args):
    t, lr = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = garch_anchors(pd.Series(lr["log_return"].to_numpy(), index=pd.to_datetime(lr["date"])))
    g["ticker"] = t
    return g.reset_index().rename(columns={"index": "date"})


def part_a(panel: pd.DataFrame, workers: int) -> dict:
    from src.db import repository as repo
    tickers = sorted(panel["ticker"].unique())
    jobs = []
    for t in tickers:
        lr = repo.read_features(t)[["log_return"]].dropna().reset_index()
        jobs.append((t, lr))
    with Pool(workers) as pool:
        parts = pool.map(_garch_one, jobs, chunksize=8)
    g = pd.concat(parts)
    g["date"] = pd.to_datetime(g["date"])
    # persisted for Week 11 round 2 (GARCH 1-day anchor); float32 to keep the file small
    keep = g[["ticker", "date", "sigma_garch_1d", "sigma_garch_1y", "garch_persist"]].dropna(subset=["sigma_garch_1d"])
    keep.astype({"sigma_garch_1d": "float32", "sigma_garch_1y": "float32", "garch_persist": "float32"}).to_parquet(
        config.PROJECT_ROOT / "data" / "processed" / "garch_anchors_sp500.parquet", index=False)
    p = panel.merge(g[["ticker", "date", "sigma_garch_1d", "sigma_garch_1y", "garch_persist"]],
                    on=["ticker", "date"], how="inner").dropna(subset=["sigma_garch_1y"])
    p["p_oracle"] = price(p, "sigma_oracle"); p["p_ewma"] = price(p, "sigma_ewma")
    p["p_g1y"] = price(p, "sigma_garch_1y"); p["p_g1d"] = price(p, "sigma_garch_1d")
    rows = []
    for t, grp in p.groupby("ticker"):
        if len(grp) < MIN_ROWS:
            continue
        ref = metric_suite(grp["p_ewma"], grp["p_oracle"], grp["fear_regime"])
        row = {"ticker": t, "n": len(grp), "ewma_mae": ref["MAE"], "persist": float(grp["garch_persist"].mean())}
        for c in ("p_g1y", "p_g1d"):
            m = metric_suite(grp[c], grp["p_oracle"], grp["fear_regime"])
            row[f"{c}_mae"] = m["MAE"]; row[f"{c}_bias"] = m["Bias"]; row[f"{c}_composite"] = safe_composite(m, ref)
        rows.append(row)
    per = pd.DataFrame(rows).set_index("ticker")
    n_undefined = int(per["p_g1y_composite"].isna().sum())
    per = per.dropna(subset=["p_g1y_composite"])
    per.to_csv(OUT / "garch_panel_metrics.csv")
    pooled_ref = metric_suite(p["p_ewma"], p["p_oracle"], p["fear_regime"])
    pooled = {"ewma": pooled_ref}
    for c in ("p_g1y", "p_g1d"):
        m = metric_suite(p[c], p["p_oracle"], p["fear_regime"]); m["composite"] = composite_score(m, pooled_ref)
        pooled[c] = m
    by_regime = {}
    for name, grp in p.groupby(p["fear_regime"].map({0: "calm", 1: "fear"})):
        by_regime[name] = {c: float((grp[c] - grp["p_oracle"]).abs().mean()) for c in ("p_ewma", "p_g1y", "p_g1d")}
    by_year = {int(y): {c: float((grp[c] - grp["p_oracle"]).abs().mean()) for c in ("p_ewma", "p_g1y", "p_g1d")}
               for y, grp in p.groupby(p["date"].dt.year)}
    summary = {
        "tickers": int(per.shape[0]), "rows": int(len(p)), "tickers_undefined_composite": n_undefined,
        "g1y_win_rate": float((per["p_g1y_composite"] < 1).mean()),
        "g1d_win_rate": float((per["p_g1d_composite"] < 1).mean()),
        "g1y_composite_median": float(per["p_g1y_composite"].median()),
        "g1y_composite_q25_q75": [float(per["p_g1y_composite"].quantile(.25)), float(per["p_g1y_composite"].quantile(.75))],
        "g1y_bias_median": float(per["p_g1y_bias"].median()),
        "persistence_median": float(per["persist"].median()),
        "pooled": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pooled.items()},
        "mae_by_regime": by_regime, "mae_by_year": by_year,
        "jpm": per.loc["JPM"].to_dict() if "JPM" in per.index else None,
    }
    # figures
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    ax.hist(per["p_g1y_composite"].clip(upper=2.5), bins=40, color=BLUE, alpha=.85)
    ax.axvline(1.0, color=ORANGE, lw=1.2, label="EWMA = 1")
    ax.axvline(per["p_g1y_composite"].median(), color=GREEN, lw=1.2, ls="--", label=f"median {per['p_g1y_composite'].median():.3f}")
    style(ax, f"GARCH 1y term-structure anchor: per-ticker composite vs own EWMA (n={len(per)}, win rate {summary['g1y_win_rate']:.0%})",
          "composite (lower = better; <1 beats EWMA)", "tickers")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(FIG / "garch_panel_composite_hist.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=200)
    labels = ["EWMA", "GARCH 1y", "GARCH 1d"]; cols = ["p_ewma", "p_g1y", "p_g1d"]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, [by_regime["calm"][c] for c in cols], w, color=BLUE, label="calm")
    ax.bar(x + w / 2, [by_regime["fear"][c] for c in cols], w, color=ORANGE, label="fear")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    style(ax, "Pooled pricing MAE vs oracle by regime (S&P 500 panel)", "", "MAE (USD)")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(FIG / "garch_panel_by_regime.png"); plt.close(fig)
    return summary, p


# ------------------------------------------------------------------ part B

def part_b(panel: pd.DataFrame, pa: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    df = panel.copy()
    df["y"] = np.log(df["sigma_oracle"] / df["sigma_ewma"])
    dates = np.array(sorted(df["date"].unique()))
    n = len(dates); i_tr, i_va = int(n * .70), int(n * .85)
    train_end, val_start, val_end, test_start = dates[i_tr - 1], dates[i_tr], dates[i_va - 1], dates[i_va]
    # SEAL: rows at/after test_start are never read below
    df = df[df["date"] <= val_end].copy()
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    val_dates = np.array(sorted(val["date"].unique()))
    blocks = [val_dates[i:i + BLOCK] for i in range(0, len(val_dates), BLOCK)]
    date_pos = {d: i for i, d in enumerate(dates)}
    models = {
        "ridge_anch": lambda: ("ridge", Ridge(alpha=10.0)),
        # diagnostic variant: features clipped at the training 1st/99th percentiles
        # (cross-sectional outliers such as special-dividend yields blow a linear model up)
        "ridge_anch_winsor": lambda: ("ridge_w", Ridge(alpha=10.0)),
        "gbdt_anch": lambda: ("gbdt", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                                                   min_samples_leaf=200, random_state=0)),
    }
    preds = {k: pd.Series(np.nan, index=val.index) for k in models}
    max_train_date = None
    for blk in blocks:
        cut_pos = date_pos[blk[0]] - HORIZON - 1                  # purge: label windows must not overlap the block
        if cut_pos < 0:
            continue
        cut_date = dates[cut_pos]
        tr = df[df["date"] <= cut_date]
        max_train_date = max(max_train_date or cut_date, cut_date)
        Xtr, ytr = tr[ML_FEATURES].to_numpy(float), tr["y"].to_numpy(float)
        mask = val["date"].isin(blk)
        Xb = val.loc[mask, ML_FEATURES].to_numpy(float)
        lo, hi = np.nanpercentile(Xtr, 1, axis=0), np.nanpercentile(Xtr, 99, axis=0)
        for name, mk in models.items():
            kind, model = mk()
            if kind == "ridge":
                sc = StandardScaler().fit(Xtr); model.fit(sc.transform(Xtr), ytr); yp = model.predict(sc.transform(Xb))
            elif kind == "ridge_w":
                Xw, Xbw = np.clip(Xtr, lo, hi), np.clip(Xb, lo, hi)
                sc = StandardScaler().fit(Xw); model.fit(sc.transform(Xw), ytr); yp = model.predict(sc.transform(Xbw))
            else:
                model.fit(Xtr, ytr); yp = model.predict(Xb)
            preds[name].loc[mask] = np.clip(yp, -RATIO_CLIP, RATIO_CLIP)
    val = val.dropna(subset=[]).copy()
    ok = preds["ridge_anch"].notna()
    val = val[ok]
    val["p_oracle"] = price(val, "sigma_oracle"); val["p_ewma"] = price(val, "sigma_ewma")
    ref = metric_suite(val["p_ewma"], val["p_oracle"], val["fear_regime"])
    table = {"ewma": {**ref, "composite": 1.0}}
    per_ticker = {}
    for name in models:
        val[f"sigma_{name}"] = np.maximum(val["sigma_ewma"] * np.exp(preds[name][ok]), 0.01)
        val[f"p_{name}"] = price(val, f"sigma_{name}")
        m = metric_suite(val[f"p_{name}"], val["p_oracle"], val["fear_regime"]); m["composite"] = composite_score(m, ref)
        ss_res = float(((val["y"] - preds[name][ok]) ** 2).sum()); ss_tot = float(((val["y"] - val["y"].mean()) ** 2).sum())
        m["ratio_R2"] = 1 - ss_res / ss_tot
        table[name] = m
        wins = []
        for t, g in val.groupby("ticker"):
            if len(g) < 63:
                continue
            r_ = metric_suite(g["p_ewma"], g["p_oracle"], g["fear_regime"])
            c_ = safe_composite(metric_suite(g[f"p_{name}"], g["p_oracle"], g["fear_regime"]), r_)
            if not np.isnan(c_):
                wins.append(c_ < 1)
        per_ticker[name] = {"win_rate": float(np.mean(wins)), "n_tickers": len(wins)}
    # GARCH 1y on the same validation rows, for the comparison table
    pv = pa.merge(val[["ticker", "date"]], on=["ticker", "date"], how="inner")
    if len(pv):
        rref = metric_suite(pv["p_ewma"], pv["p_oracle"], pv["fear_regime"])
        m = metric_suite(pv["p_g1y"], pv["p_oracle"], pv["fear_regime"]); m["composite"] = composite_score(m, rref)
        table["garch_1y_anchor"] = m
    tbl = pd.DataFrame(table).T
    tbl.to_csv(OUT / "ml_panel_metrics.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=200)
    names = list(tbl.index); vals = tbl["composite"].to_numpy(float)
    ax.barh(names, vals, color=[GREEN if v < 1 else BLUE for v in vals]); ax.axvline(1, color=ORANGE, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    style(ax, "Validation composite vs EWMA — S&P 500 panel, walk-forward, test sealed", "composite", "")
    fig.tight_layout(); fig.savefig(FIG / "ml_panel_composite.png"); plt.close(fig)
    return {
        "split": {"train_end": str(pd.Timestamp(train_end).date()), "val_start": str(pd.Timestamp(val_start).date()),
                  "val_end": str(pd.Timestamp(val_end).date()), "test_start": str(pd.Timestamp(test_start).date()),
                  "test_end": str(pd.Timestamp(dates[-1]).date()), "purge_rows": HORIZON, "block": BLOCK},
        "sealed": True, "max_train_date_used": str(pd.Timestamp(max_train_date).date()),
        "val_rows": int(len(val)), "val_tickers": int(val["ticker"].nunique()),
        "metrics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in table.items()},
        "per_ticker_win_rate": per_ticker,
    }


def main() -> int:
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    FIG.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    print(f"panel rows {len(panel)}, tickers {panel['ticker'].nunique()}, dates {panel['date'].min().date()}..{panel['date'].max().date()}")
    a, pa = part_a(panel, workers)
    print(f"GARCH 1y: win rate {a['g1y_win_rate']:.1%}, median composite {a['g1y_composite_median']:.3f}, "
          f"pooled MAE {a['pooled']['p_g1y']['MAE']:.2f} vs EWMA {a['pooled']['ewma']['MAE']:.2f}, bias median {a['g1y_bias_median']:+.2f}")
    b = part_b(panel, pa)
    for k, v in b["metrics"].items():
        print(f"  {k:16s} MAE {v['MAE']:.3f} composite {v['composite']:.3f}" + (f" ratio_R2 {v['ratio_R2']:+.3f}" if 'ratio_R2' in v else ""))
    print("per-ticker win rate:", b["per_ticker_win_rate"], "| test sealed from", b["split"]["test_start"])
    summary = {"panel": {"rows": int(len(panel)), "tickers": int(panel["ticker"].nunique()),
                         "start": str(panel["date"].min().date()), "end": str(panel["date"].max().date())},
               "garch": a, "ml": b}
    (OUT / "panel_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
