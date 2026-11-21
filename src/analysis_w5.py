"""Week 5 analysis: initial ML frameworks, evaluated on the VALIDATION set.

Discipline: the TEST set stays sealed this week — it will be opened exactly
once, in Week 6, after hyperparameter tuning. Everything reported here is
train/validation only.

Evaluation is same-ruler as Week 4: predicted sigma (route A) or predicted
price/spot (route B) -> chooser price -> 8-metric suite + weighted composite
against the EWMA baseline computed on the same validation days.

Outputs: docs/week5/figures/*.png, docs/week5/results_summary.json,
         models/*.joblib (initial fitted sklearn models)

Usage: python -m src.analysis_w5
"""

import json
import warnings

import joblib
import matplotlib

warnings.filterwarnings("ignore")  # sklearn extrapolation warnings pollute logs

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.metrics import composite_score, metric_suite
from src.ml.dataset import build
from src.ml.frameworks import train_and_predict
from src.models.chooser import chooser_closed_form

OUT = config.PROJECT_ROOT / "docs" / "week5"
FIG = OUT / "figures"
MODELS = config.PROJECT_ROOT / "models"

T1, T2 = 0.5, 1.0
SIGMA_FLOOR = 0.01   # guard: linear/NN sigma predictions clipped to >= 1%

BLUE, ORANGE, GREEN, PURPLE, GRAY, TEAL, BROWN = (
    "#2f6db3", "#d1603d", "#3a7d44", "#7b5ca6", "#8a8a8a", "#2a8a84", "#8a6d3b")
INK, GRID = "#333333", "#d9d9d9"

MODEL_LABELS = {
    "ridge_a": "A: Ridge", "rf_a": "A: RandomForest", "gbdt_a": "A: GBDT",
    "lstm_a": "A: LSTM",
    "ridge_anch": "A-anchored: Ridge", "rf_anch": "A-anchored: RF",
    "gbdt_anch": "A-anchored: GBDT",
    "ridge_b": "B: Ridge", "gbdt_b": "B: GBDT", "mlp_b": "B: MLP",
}


def price_from_sigma(panel: pd.DataFrame, idx, sigma: pd.Series) -> pd.Series:
    S = panel.loc[idx, "jpm_close"].to_numpy()
    return pd.Series(
        chooser_closed_form(
            S0=S, K=S, r=panel.loc[idx, "r_1y"].to_numpy(),
            q=panel.loc[idx, "div_yield_ttm"].to_numpy(),
            sigma=np.clip(sigma.loc[idx].to_numpy(), SIGMA_FLOOR, None),
            t1=T1, T2=T2),
        index=idx)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    data = build()
    panel = data.panel
    print(f"panel {len(panel)} days | train(purged) {len(data.idx_train)} "
          f"({data.idx_train.min().date()}..{data.idx_train.max().date()}) | "
          f"val {len(data.idx_val)} ({data.idx_val.min().date()}..{data.idx_val.max().date()}) | "
          f"test {len(data.idx_test)} (SEALED until W6)")

    results = train_and_predict(data)

    # ---- route A: volatility-forecast quality on val ---------------------
    print("\n== route A: sigma forecast on validation (vol points) ==")
    oracle_sigma = data.y_sigma
    vol_stats = {}
    for name in ("ridge_a", "rf_a", "gbdt_a", "lstm_a",
                 "ridge_anch", "rf_anch", "gbdt_anch"):
        p = results[name].preds["val"]
        idx = p.index
        if results[name].route == "A_anch":
            p = panel.loc[idx, "sigma_ewma"] * np.exp(p)
        e = p - oracle_sigma.loc[idx]
        ss_tot = ((oracle_sigma.loc[idx] - oracle_sigma.loc[idx].mean()) ** 2).sum()
        r2 = 1 - (e**2).sum() / ss_tot
        vol_stats[name] = {"MAE_volpts": float(e.abs().mean()),
                           "RMSE_volpts": float(np.sqrt((e**2).mean())),
                           "R2": float(r2)}
        print(f"  {MODEL_LABELS[name]:>16s}: MAE {e.abs().mean():.4f}  "
              f"RMSE {np.sqrt((e**2).mean()):.4f}  R2 {r2:+.3f}")

    # ---- pricing metrics + composite on val (same ruler as W4) ----------
    print("\n== pricing on validation: 8-metric suite + composite (ref = EWMA on val) ==")
    val = data.idx_val
    oracle_price = panel["price_sigma_oracle"]
    regime = panel["fear_regime"]

    ref_suite = metric_suite(panel.loc[val, "price_sigma_ewma"],
                             oracle_price.loc[val], regime.loc[val])

    suites, composites, val_prices = {}, {}, {}
    for name, res in results.items():
        p = res.preds["val"]
        idx = p.index
        if res.route == "A":
            price = price_from_sigma(panel, idx, p)
        elif res.route == "A_anch":
            sigma = panel.loc[idx, "sigma_ewma"] * np.exp(p)
            price = price_from_sigma(panel, idx, sigma)
        else:
            price = p * panel.loc[idx, "jpm_close"]
        val_prices[name] = price
        s = metric_suite(price, oracle_price.loc[idx], regime.loc[idx])
        suites[name] = s
        composites[name] = composite_score(s, ref_suite)
        print(f"  {MODEL_LABELS[name]:>16s}: MAE {s['MAE']:.2f}  RMSE {s['RMSE']:.2f}  "
              f"Bias {s['Bias']:+.2f}  P95 {s['P95AE']:.2f}  R2 {s['R2']:+.3f}  "
              f"Hit10 {s['Hit10']:.1%}  ->  composite {composites[name]:.3f}")
    ewma_suite_print = ref_suite
    print(f"  {'EWMA baseline':>16s}: MAE {ewma_suite_print['MAE']:.2f}  "
          f"R2 {ewma_suite_print['R2']:+.3f}  ->  composite 1.000 (reference)")

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    ax.plot(val, oracle_sigma.loc[val], color=GRAY, linewidth=1.6, label="oracle (realized)")
    colors = {"ridge_a": ORANGE, "rf_a": GREEN, "gbdt_a": PURPLE, "lstm_a": BLUE}
    for name, c in colors.items():
        p = results[name].preds["val"]
        ax.plot(p.index, p.values, color=c, linewidth=1.1, label=MODEL_LABELS[name])
    ax.plot(val, panel.loc[val, "sigma_ewma"], color=INK, linewidth=1.0,
            linestyle="--", label="EWMA baseline")
    ax.set_title("Route A: 1y-forward volatility forecasts on validation set",
                 fontsize=10, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "val_vol_forecasts.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=200)
    names = list(composites)
    vals = [composites[n] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colors_bar = [GREEN if v < 1 else BLUE for v in vals]
    ax.barh([MODEL_LABELS[n] for n in names], vals, color=colors_bar, height=0.6)
    ax.set_xscale("log")   # MLP instability (~90) would crush a linear axis
    ax.axvline(1.0, color=ORANGE, linewidth=1.4)
    ax.annotate("EWMA baseline = 1.0", xy=(1.0, 0.02), xycoords=("data", "axes fraction"),
                fontsize=8, color=ORANGE, xytext=(4, 2), textcoords="offset points")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_title("Weighted composite on validation (lower = better; <1 beats baseline)",
                 fontsize=10, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "val_composite.png", bbox_inches="tight")
    plt.close(fig)

    # ---- persist models + summary ---------------------------------------
    for name, res in results.items():
        if hasattr(res, "fitted"):
            joblib.dump(res.fitted, MODELS / f"{name}.joblib", compress=3)

    summary = {
        "splits": {
            "train_purged": [str(data.idx_train.min().date()), str(data.idx_train.max().date()),
                             len(data.idx_train)],
            "val": [str(data.idx_val.min().date()), str(data.idx_val.max().date()),
                    len(data.idx_val)],
            "test_sealed": [str(data.idx_test.min().date()), str(data.idx_test.max().date()),
                            len(data.idx_test)],
            "purge_rows": 252,
        },
        "route_a_vol_stats_val": vol_stats,
        "pricing_suites_val": suites,
        "ewma_reference_val": ref_suite,
        "composites_val": composites,
    }
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfigures -> {FIG}\nmodels -> {MODELS}\nsummary -> {OUT/'results_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
