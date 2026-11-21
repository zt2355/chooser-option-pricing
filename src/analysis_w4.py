"""Week 4 analysis: baseline BSM performance evaluation.

Three evaluation layers (CME prices replaced by mentor-approved MC benchmark):
  1. Implementation correctness: MC vs closed form, MAE/RMSE over independent
     seeds at increasing path counts.
  2. Assumption failure (core): daily rolling-calibrated ATM chooser prices
     using backward-looking sigma estimators vs the ex-post "oracle" price
     using the volatility actually realized over the option's 1y life.
     Errors grouped by fear regime / year quantify BSM's constant-vol cost.
  3. Choice-rule cost on real dates: W3's moneyness-rule gap re-checked in the
     2021 low-rate vs 2023 high-rate environments.

Outputs (docs/week4/): figures/*.png, results_summary.json,
                       baseline_metrics.csv (the Week 5-6 benchmark table)

Usage: python -m src.analysis_w4
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.metrics import metric_suite, composite_score, weight_sensitivity, WEIGHTS
from src.models.chooser import chooser_closed_form, chooser_mc

OUT = config.PROJECT_ROOT / "docs" / "week4"
FIG = OUT / "figures"

STUDY_START, STUDY_END = "2019-10-17", "2024-12-31"
T1, T2 = 0.5, 1.0
EWMA_LAMBDA = 0.94  # RiskMetrics

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2f6db3", "#d1603d", "#3a7d44", "#7b5ca6", "#8a8a8a"
INK, GRID = "#333333", "#d9d9d9"

SIGMA_ESTIMATORS = ["sigma_21d", "sigma_63d", "sigma_252d", "sigma_ewma"]
LABELS = {"sigma_21d": "21d realized", "sigma_63d": "63d realized",
          "sigma_252d": "252d realized", "sigma_ewma": f"EWMA(λ={EWMA_LAMBDA})"}
COLORS = {"sigma_21d": ORANGE, "sigma_63d": GREEN,
          "sigma_252d": PURPLE, "sigma_ewma": BLUE}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------ layer 1: MC

def mc_validation_metrics() -> dict:
    """MAE/RMSE of independent MC estimates vs the closed form (paper params)."""
    p = json.load(open(config.PROJECT_ROOT / "configs" / "paper_params.json"))
    p.pop("description")
    cf = chooser_closed_form(**p)
    rows = []
    for n in [1_000, 10_000, 100_000]:
        est = np.array([chooser_mc(**p, n_paths=n, seed=s).price for s in range(30)])
        err = est - cf
        rows.append({"n_paths": n, "runs": 30,
                     "MAE": float(np.abs(err).mean()),
                     "RMSE": float(np.sqrt((err**2).mean())),
                     "MAE_pct_of_price": float(np.abs(err).mean() / cf * 100)})
    return {"closed_form": cf, "table": rows}


# ------------------------------------------------ layer 2: rolling pricing

def build_panel() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "features_daily.csv",
                     index_col="date", parse_dates=True)

    # EWMA variance from daily log returns, annualized
    r2 = df["log_return"] ** 2
    ewma_var = r2.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    df["sigma_ewma"] = np.sqrt(ewma_var * 252)

    df["sigma_21d"] = df["realized_vol_21d"]
    df["sigma_63d"] = df["realized_vol_63d"]
    df["sigma_252d"] = df["realized_vol_252d"]
    # oracle: volatility actually realized over the option's 1y life (t, t+252]
    df["sigma_oracle"] = df["realized_vol_252d"].shift(-252)

    panel = df.loc[STUDY_START:STUDY_END].dropna(
        subset=SIGMA_ESTIMATORS + ["sigma_oracle", "r_1y", "div_yield_ttm",
                                   "jpm_close", "fear_regime"]).copy()
    return panel


def price_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Vectorized daily ATM chooser pricing for every sigma column."""
    S = panel["jpm_close"].to_numpy()
    K = S  # ATM chooser, fresh 1y contract each day (comparability over time)
    r = panel["r_1y"].to_numpy()
    q = panel["div_yield_ttm"].to_numpy()
    out = panel.copy()
    for col in SIGMA_ESTIMATORS + ["sigma_oracle"]:
        out[f"price_{col}"] = chooser_closed_form(
            S0=S, K=K, r=r, sigma=panel[col].to_numpy(), q=q, t1=T1, T2=T2)
    for col in SIGMA_ESTIMATORS:
        out[f"err_{col}"] = out[f"price_{col}"] - out["price_sigma_oracle"]
        out[f"relerr_{col}"] = out[f"err_{col}"] / out["price_sigma_oracle"]
    return out


def metric_tables(priced: pd.DataFrame) -> dict:
    def mae_rmse(g, col):
        e = g[f"err_{col}"]
        re = g[f"relerr_{col}"]
        return {"MAE_usd": float(e.abs().mean()),
                "RMSE_usd": float(np.sqrt((e**2).mean())),
                "MAE_pct": float(re.abs().mean() * 100)}

    overall = {c: mae_rmse(priced, c) for c in SIGMA_ESTIMATORS}
    by_regime = {
        regime: {c: mae_rmse(g, c) for c in SIGMA_ESTIMATORS}
        for regime, g in priced.groupby(priced["fear_regime"].map(
            {0: "calm", 1: "fear"}))
    }
    by_year = {
        int(y): {c: mae_rmse(g, c) for c in SIGMA_ESTIMATORS}
        for y, g in priced.groupby(priced.index.year)
    }
    return {"overall": overall, "by_regime": by_regime, "by_year": by_year}


# ------------------------------------------------ layer 3: rule gap, real dates

def rule_gap_real_dates(priced: pd.DataFrame) -> list[dict]:
    """W3's moneyness-rule cost re-checked on representative real dates."""
    dates = {"low-rate (2021-06-30)": "2021-06-30",
             "high-rate (2023-10-02)": "2023-10-02"}
    rows = []
    for label, d in dates.items():
        row = priced.loc[d] if d in priced.index else priced.iloc[
            priced.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]]
        p = dict(S0=float(row["jpm_close"]), K=float(row["jpm_close"]),
                 r=float(row["r_1y"]), sigma=float(row["sigma_ewma"]),
                 q=float(row["div_yield_ttm"]), t1=T1, T2=T2)
        o = chooser_mc(**p, n_paths=400_000, rule="optimal", seed=11)
        m = chooser_mc(**p, n_paths=400_000, rule="moneyness", seed=11)
        cf = chooser_closed_form(**p)
        rows.append({"scenario": label, "r_1y": p["r"], "sigma": p["sigma"],
                     "closed_form": cf,
                     "rule_gap_usd": o.price - m.price,
                     "rule_gap_bp": (o.price - m.price) / cf * 1e4})
    return rows


# ---------------------------------------------------------------- figures

def figures(priced: pd.DataFrame, metrics: dict):
    # 1. price level + regime shading
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=200)
    ax.plot(priced.index, priced["price_sigma_oracle"], color=GRAY,
            linewidth=1.1, label="oracle (fwd realized vol)")
    ax.plot(priced.index, priced["price_sigma_ewma"], color=BLUE,
            linewidth=1.3, label="EWMA-calibrated")
    fear = priced["fear_regime"].to_numpy(dtype=bool)
    ax.fill_between(priced.index, 0, 1, where=fear, alpha=0.12,
                    color=ORANGE, transform=ax.get_xaxis_transform(),
                    label="fear regime")
    style(ax, "Daily 1y ATM chooser price: backward calibration vs ex-post oracle",
          "", "chooser price (USD)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rolling_price.png", bbox_inches="tight")
    plt.close(fig)

    # 2. pricing error time series (EWMA, best backward estimator)
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
    ax.plot(priced.index, priced["err_sigma_ewma"], color=BLUE, linewidth=1.0)
    ax.axhline(0, color=GRAY, linewidth=1.0)
    ax.fill_between(priced.index, 0, 1, where=fear, alpha=0.12, color=ORANGE,
                    transform=ax.get_xaxis_transform())
    style(ax, "BSM constant-vol pricing error (EWMA calibration minus oracle); shaded = fear regime",
          "", "error (USD)")
    fig.tight_layout()
    fig.savefig(FIG / "pricing_error_ts.png", bbox_inches="tight")
    plt.close(fig)

    # 3. grouped MAE by regime
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    regimes = ["calm", "fear"]
    x = np.arange(len(SIGMA_ESTIMATORS))
    width = 0.36
    for i, regime in enumerate(regimes):
        vals = [metrics["by_regime"][regime][c]["MAE_usd"] for c in SIGMA_ESTIMATORS]
        ax.bar(x + (i - 0.5) * width, vals, width * 0.94,
               color=[BLUE, ORANGE][i], label=regime)
    ax.set_xticks(x, [LABELS[c] for c in SIGMA_ESTIMATORS], fontsize=8)
    style(ax, "Pricing MAE vs oracle, by volatility estimator and market regime",
          "backward-looking σ estimator", "MAE (USD)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "mae_by_regime.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)

    print("== layer 1: MC implementation validation (paper params) ==")
    mc_val = mc_validation_metrics()
    print(f"closed form {mc_val['closed_form']:.4f}")
    for row in mc_val["table"]:
        print(f"  n={row['n_paths']:>7,d}  MAE={row['MAE']:.4f}  "
              f"RMSE={row['RMSE']:.4f}  ({row['MAE_pct_of_price']:.3f}% of price)")

    print("\n== layer 2: rolling real-data pricing vs oracle ==")
    panel = build_panel()
    priced = price_panel(panel)
    print(f"panel: {len(priced)} trading days "
          f"({priced.index.min().date()} .. {priced.index.max().date()})")
    metrics = metric_tables(priced)
    for c in SIGMA_ESTIMATORS:
        m = metrics["overall"][c]
        print(f"  {LABELS[c]:>14s}: MAE ${m['MAE_usd']:.3f}  RMSE ${m['RMSE_usd']:.3f}  "
              f"({m['MAE_pct']:.1f}%)")
    for regime in ("calm", "fear"):
        m = metrics["by_regime"][regime]["sigma_ewma"]
        print(f"  EWMA in {regime:>4s}: MAE ${m['MAE_usd']:.3f} ({m['MAE_pct']:.1f}%)")

    print("\n== layer 3: moneyness-rule cost on real dates ==")
    gaps = rule_gap_real_dates(priced)
    for g in gaps:
        print(f"  {g['scenario']:<24s} r={g['r_1y']*100:.2f}%  "
              f"gap = ${g['rule_gap_usd']:.4f} ({g['rule_gap_bp']:.1f} bp)")

    figures(priced, metrics)

    # -- extended metric suite + weighted composite (mentor request) ------
    print("\n== extended metrics + weighted composite (reference = EWMA) ==")
    suites = {
        LABELS[c]: metric_suite(priced[f"price_{c}"], priced["price_sigma_oracle"],
                                priced["fear_regime"])
        for c in SIGMA_ESTIMATORS
    }
    ref_name = LABELS["sigma_ewma"]
    composites = {n: composite_score(m, suites[ref_name]) for n, m in suites.items()}
    sens = weight_sensitivity(suites, ref_name)
    for n, m in suites.items():
        print(f"  {n:>14s}: MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  "
              f"MedAE {m['MedAE']:.2f}  Bias {m['Bias']:+.2f}  P95 {m['P95AE']:.2f}  "
              f"WorstRegime {m['RegimeWorstMAE']:.2f}  R2 {m['R2']:+.3f}  "
              f"Hit10 {m['Hit10']:.1%}  ->  composite {composites[n]:.3f}")
    rankings = {s: v["ranking"] for s, v in sens.items()}
    stable = len({tuple(r) for r in rankings.values()}) == 1
    print(f"  ranking stable across 4 weighting schemes: {stable}")

    # benchmark table for Weeks 5-6 (what ML must beat)
    bench = pd.DataFrame(suites).T
    bench["composite"] = pd.Series(composites)
    bench.to_csv(OUT / "baseline_metrics.csv")

    summary = {"mc_validation": mc_val,
               "panel_days": len(priced),
               "panel_window": [str(priced.index.min().date()),
                                str(priced.index.max().date())],
               "metrics": metrics,
               "metric_suites": suites,
               "composite_weights": WEIGHTS,
               "composites": composites,
               "weight_sensitivity": sens,
               "rule_gap_real_dates": gaps}
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfigures -> {FIG}\nbenchmark -> {OUT/'baseline_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
