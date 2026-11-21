"""Week 9 theory line: GARCH(1,1) as an alternative volatility anchor.

Question: the tool anchors on EWMA (RiskMetrics, lambda = 0.94). EWMA is the
degenerate GARCH(1,1) with omega = 0, alpha + beta = 1, i.e. an integrated
process with no mean reversion. A 1-year chooser needs the volatility
expected over the option's LIFE, so the theoretically matching anchor is the
GARCH term-structure forecast

    sigma_1y^2 = (1/H) * sum_{k=1..H} E_t[h_{t+k}],   H = 252,
    E_t[h_{t+k}] = v + (alpha+beta)^(k-1) (h_{t+1} - v),   v = omega/(1-alpha-beta)

which mean-reverts the spot variance toward the long-run level over the
horizon. This script asks whether that anchor prices the daily ATM chooser
closer to the ex-post oracle than EWMA, using the Week 4 panel and the
Week 4 metric suite (composite < 1 beats EWMA).

Protocol (no look-ahead)
  * parameters re-estimated every 21 trading days on all returns up to the
    refit date (expanding window, >= 750 obs); between refits the fitted
    (omega, alpha, beta) filter the realised returns day by day
  * candidates: sigma_garch_1y (term-structure, the theory-matched anchor),
    sigma_garch_1d (spot, next-day forecast annualised)
  * reference: sigma_ewma; oracle: realised vol over the following 252 days

Outputs (docs/week9/): garch_metrics.csv, results_summary.json,
  figures/garch_vs_ewma.png, figures/garch_composite.png

Usage: python -m src.analysis_w9
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.analysis_w4 import STUDY_END, STUDY_START, T1, T2, build_panel
from src.metrics import composite_score, metric_suite, weight_sensitivity
from src.models.chooser import chooser_closed_form

OUT = config.PROJECT_ROOT / "docs" / "week9"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN, GRAY, PURPLE = "#2f6db3", "#d1603d", "#3a7d44", "#8a8a8a", "#7b4fa3"
INK, GRID = "#333333", "#d9d9d9"
REFIT, MIN_OBS, H = 21, 750, 252


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fit_garch(r_pct: np.ndarray) -> tuple[float, float, float, float, float]:
    """(mu, omega, alpha, beta, conditional variance of the last obs) on percent returns."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal",
                         rescale=False).fit(disp="off", show_warning=False)
    p = res.params
    return (float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"]),
            float(res.conditional_volatility[-1] ** 2))


def garch_anchors(log_ret: pd.Series) -> pd.DataFrame:
    """Daily sigma_garch_1d / sigma_garch_1y with 21-day refits, no look-ahead.

    State h_next = E_t[h_{t+1}] after observing day t's return:
      refit day : (mu, omega, alpha, beta, h_t) from arch on returns <= t
      otherwise : h_t = previous h_next
      h_next    = omega + alpha (r_t - mu)^2 + beta h_t
    """
    r = log_ret.dropna() * 100.0
    out = pd.DataFrame(index=r.index, columns=["sigma_garch_1d", "sigma_garch_1y",
                                               "garch_alpha", "garch_beta", "garch_persist"], dtype=float)
    mu = omega = alpha = beta = h_next = None
    refits = 0
    for i, (t, rt) in enumerate(r.items()):
        if i + 1 < MIN_OBS:
            continue
        if h_next is None or (i + 1 - MIN_OBS) % REFIT == 0:
            mu, omega, alpha, beta, h_t = fit_garch(r.iloc[:i + 1].to_numpy())
            refits += 1
        else:
            h_t = h_next
        h_next = omega + alpha * (rt - mu) ** 2 + beta * h_t
        persist = alpha + beta
        if persist < 1:
            v = omega / (1 - persist)
            avg = v + (h_next - v) * (1 - persist**H) / (H * (1 - persist))
        else:
            avg = h_next
        out.loc[t, "sigma_garch_1d"] = np.sqrt(h_next * 252) / 100.0
        out.loc[t, "sigma_garch_1y"] = np.sqrt(avg * 252) / 100.0
        out.loc[t, ["garch_alpha", "garch_beta", "garch_persist"]] = [alpha, beta, persist]
    out.attrs["refits"] = refits
    return out


def price(panel: pd.DataFrame, col: str) -> np.ndarray:
    return np.asarray(chooser_closed_form(S0=panel["jpm_close"].to_numpy(), K=panel["jpm_close"].to_numpy(),
                                          r=panel["r_1y"].to_numpy(), q=panel["div_yield_ttm"].to_numpy(),
                                          sigma=panel[col].to_numpy(), t1=T1, T2=T2))


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(config.DATA_PROCESSED / "features_daily.csv", index_col="date", parse_dates=True)
    g = garch_anchors(full["log_return"])
    panel = build_panel().join(g, how="left")
    panel = panel.dropna(subset=["sigma_garch_1y", "sigma_garch_1d"])
    print(f"panel {len(panel)} days {panel.index.min().date()}..{panel.index.max().date()}, "
          f"GARCH refits {g.attrs['refits']}, mean persistence {panel['garch_persist'].mean():.4f}")

    cands = ["sigma_ewma", "sigma_garch_1d", "sigma_garch_1y", "sigma_252d"]
    oracle = pd.Series(price(panel, "sigma_oracle"), index=panel.index)
    prices = {c: pd.Series(price(panel, c), index=panel.index) for c in cands}
    regime = panel["fear_regime"]
    metrics = {c: metric_suite(prices[c], oracle, regime) for c in cands}
    ref = metrics["sigma_ewma"]
    for c in cands:
        metrics[c]["composite"] = composite_score(metrics[c], ref)
    sens = weight_sensitivity({c: {k: v for k, v in m.items() if k != "composite"} for c, m in metrics.items()},
                              "sigma_ewma")
    tbl = pd.DataFrame(metrics).T
    tbl.to_csv(OUT / "garch_metrics.csv")
    print(tbl[["MAE", "RMSE", "Bias", "P95AE", "RegimeWorstMAE", "R2", "Hit10", "composite"]].round(4).to_string())
    print("ranking (main weights):", sens["main"]["ranking"])

    by_regime = {}
    for name, grp in panel.groupby(regime.map({0: "calm", 1: "fear"})):
        by_regime[name] = {c: float((prices[c].loc[grp.index] - oracle.loc[grp.index]).abs().mean()) for c in cands}
    by_year = {int(y): {c: float((prices[c].loc[grp.index] - oracle.loc[grp.index]).abs().mean()) for c in cands}
               for y, grp in panel.groupby(panel.index.year)}

    # ---- figure 1: anchors vs oracle ----
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    ax.plot(panel.index, panel["sigma_oracle"], color=GRAY, lw=1.0, label="oracle (fwd 252d realised)")
    ax.plot(panel.index, panel["sigma_ewma"], color=BLUE, lw=1.0, label="EWMA (λ=0.94)")
    ax.plot(panel.index, panel["sigma_garch_1y"], color=ORANGE, lw=1.0, label="GARCH(1,1) 1y term-structure")
    ax.plot(panel.index, panel["sigma_garch_1d"], color=PURPLE, lw=0.7, alpha=0.7, label="GARCH(1,1) 1-day")
    style(ax, "Volatility anchors vs ex-post oracle (JPM, daily, refit every 21d)", "", "annualised sigma")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "garch_vs_ewma.png"); plt.close(fig)

    # ---- figure 2: composite bars ----
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200)
    names = {"sigma_ewma": "EWMA (reference)", "sigma_garch_1d": "GARCH 1-day",
             "sigma_garch_1y": "GARCH 1y term", "sigma_252d": "252d realised"}
    vals = [metrics[c]["composite"] for c in cands]
    cols = [GREEN if v < 1 else BLUE for v in vals]
    ax.barh([names[c] for c in cands], vals, color=cols)
    ax.axvline(1.0, color=ORANGE, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    style(ax, "Weighted composite vs EWMA (lower = better; <1 beats EWMA)", "composite", "")
    fig.tight_layout(); fig.savefig(FIG / "garch_composite.png"); plt.close(fig)

    summary = {
        "panel_days": int(len(panel)), "study": [str(panel.index.min().date()), str(panel.index.max().date())],
        "garch": {"refit_every": REFIT, "min_obs": MIN_OBS, "horizon": H, "refits": int(g.attrs["refits"]),
                  "mean_persistence": float(panel["garch_persist"].mean()),
                  "min_persistence": float(panel["garch_persist"].min()),
                  "max_persistence": float(panel["garch_persist"].max())},
        "metrics": {c: {k: float(v) for k, v in m.items()} for c, m in metrics.items()},
        "weight_sensitivity": sens, "mae_by_regime": by_regime, "mae_by_year": by_year,
        "verdict": ("GARCH 1y term-structure beats EWMA" if metrics["sigma_garch_1y"]["composite"] < 1
                    else "EWMA remains the better anchor"),
    }
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("verdict:", summary["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
