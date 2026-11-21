"""Week 7 analysis: extended sensitivity + extreme scenarios.

Outputs (docs/week7/):
  figures/scenario_tornado.png   extreme-scenario price impacts (USD)
  figures/stress_heatmap.png     price over a sigma x rate stress grid
  figures/shap_price_impact.png  W6 SHAP feature effects translated to USD
                                 (per-feature mean |log-sigma impact| -> price)
  results_summary.json, scenario_table.csv

Usage: python -m src.analysis_w7
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.tool.pricing_service import (load_market_state, price_chooser,
                                      quote, scenario_table)

OUT = config.PROJECT_ROOT / "docs" / "week7"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN, GRAY = "#2f6db3", "#d1603d", "#3a7d44", "#8a8a8a"
INK, GRID = "#333333", "#d9d9d9"
T1, T2 = 0.5, 1.0


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    state = load_market_state()
    q = quote(state)
    tbl = scenario_table(state)
    tbl.to_csv(OUT / "scenario_table.csv", index=False)

    # ---- tornado of scenario impacts ------------------------------------
    imp = tbl[tbl["情景"] != "基准(当前市场)"].copy()
    imp = imp.sort_values("变化($)")
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    labels_en = {"波动率 +50%": "vol +50%", "波动率 -30%": "vol -30%",
                 "利率 +200bp": "rate +200bp", "利率 -100bp": "rate -100bp",
                 "股价 -15%(K 不变)": "spot -15% (K fixed)",
                 "股价 +15%(K 不变)": "spot +15% (K fixed)",
                 "组合压力: 波动率+50% 且 利率+200bp": "stress: vol+50% & rate+200bp"}
    ax.barh([labels_en[s] for s in imp["情景"]], imp["变化($)"],
            color=[ORANGE if v < 0 else BLUE for v in imp["变化($)"]], height=0.6)
    ax.axvline(0, color=GRAY, linewidth=1.0)
    for i, v in enumerate(imp["变化($)"]):
        ax.text(v, i, f" {v:+.1f}", va="center", fontsize=8, color=INK)
    style(ax, f"Extreme-scenario price impacts (base {q['primary_price']:.2f} USD, "
              f"as of {q['as_of']})", "price change (USD)", "")
    fig.tight_layout(); fig.savefig(FIG / "scenario_tornado.png", bbox_inches="tight")
    plt.close(fig)

    # ---- sigma x rate stress heatmap ------------------------------------
    sig_mult = np.linspace(0.5, 2.0, 16)
    rate_shift = np.linspace(-0.02, 0.04, 13)
    Z = np.array([[price_chooser(state.S, state.S, state.r_1y + dr, state.q,
                                 state.sigma_ewma * m, T1, T2)
                   for dr in rate_shift] for m in sig_mult])
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=200)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="Blues",
                   extent=[rate_shift[0] * 1e4, rate_shift[-1] * 1e4,
                           sig_mult[0], sig_mult[-1]])
    cs = ax.contour(np.linspace(rate_shift[0] * 1e4, rate_shift[-1] * 1e4, 13),
                    sig_mult, Z, colors=INK, linewidths=0.6)
    ax.clabel(cs, fontsize=7, fmt="%.0f")
    ax.plot(0, 1.0, marker="o", color=ORANGE, markersize=7)
    ax.annotate("current market", xy=(0, 1.0), fontsize=8, color=ORANGE,
                xytext=(8, 6), textcoords="offset points")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("chooser price (USD)", fontsize=9, color=INK)
    style(ax, "Stress surface: volatility multiplier x rate shift",
          "rate shift (bp)", "volatility multiplier")
    fig.tight_layout(); fig.savefig(FIG / "stress_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ---- SHAP -> USD price impact (Week 7 spec: quantify new-feature impact)
    shap_rank = json.load(open(config.PROJECT_ROOT / "docs" / "week6"
                               / "results_summary.json"))["shap_rank"]
    base_p = q["primary_price"]
    rows = {}
    for feat, m in shap_rank.items():
        up = price_chooser(state.S, state.S, state.r_1y, state.q,
                           state.sigma_ewma * float(np.exp(m)), T1, T2)
        rows[feat] = up - base_p
    s = pd.Series(rows).sort_values()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    ax.barh(s.index, s.values, color=GREEN, height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=7.5, color=INK)
    style(ax, "Feature -> price impact: mean |SHAP| on ln(sigma) translated to USD\n"
              "(at current market, per one typical feature contribution)",
          "price impact (USD)", "")
    fig.tight_layout(); fig.savefig(FIG / "shap_price_impact.png", bbox_inches="tight")
    plt.close(fig)

    summary = {"as_of": q["as_of"], "quote": {k: v for k, v in q.items()
                                              if not isinstance(v, tuple)},
               "primary_band": list(q["primary_band"]),
               "scenarios": tbl.to_dict(orient="records"),
               "shap_price_impact_usd": s.round(4).to_dict(),
               "stress_grid": {"sigma_mult": sig_mult.tolist(),
                               "rate_shift": rate_shift.tolist(),
                               "price_min": float(Z.min()),
                               "price_max": float(Z.max())}}
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2,
                                                         ensure_ascii=False))
    print(f"as of {q['as_of']}: base {base_p:.2f}, band {q['primary_band']}")
    print(tbl.to_string(index=False))
    print("top SHAP price impacts:",
          ", ".join(f"{k}({v:+.2f})" for k, v in s.sort_values(ascending=False)[:4].items()))
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
