"""Week 6 analysis: walk-forward tuning, final TEST evaluation, SHAP.

Protocol:
  1. TUNE on validation: hyperparameters selected by walk-forward pricing MAE
     on the W5 validation window (test still sealed).
  2. OPEN THE TEST SET ONCE: tuned models walk-forward over 2024-03..2024-12;
     full 8-metric suite + weighted composite (reference = EWMA on test) +
     weight-sensitivity ranking.
  3. SHAP on the final deployment GBDT (TreeExplainer, beeswarm + bar).
  4. Save final deployment models (trained on all purgeable history) to
     models/final_*.joblib.

All targets are ANCHORED (W5 design): route A predicts ln(sigma/sigma_EWMA),
route B predicts ln(P/P_EWMA). Rate-level features replaced by deviations
(W6 feature fix). LSTM uses a single fixed config (W5 showed capacity is not
the bottleneck); sklearn models are grid-tuned.

Usage: python -m src.analysis_w6
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics import composite_score, metric_suite, weight_sensitivity
from src.ml.dataset import HORIZON, build
from src.ml.walkforward import ml_features, walk_forward, walk_forward_lstm
from src.models.chooser import chooser_closed_form

warnings.filterwarnings("ignore")

OUT = config.PROJECT_ROOT / "docs" / "week6"
FIG = OUT / "figures"
MODELS = config.PROJECT_ROOT / "models"

T1, T2 = 0.5, 1.0
SIGMA_FLOOR, RATIO_CLIP = 0.01, 1.5   # decode guards
SEED = 42

BLUE, ORANGE, GREEN, PURPLE, GRAY, TEAL = (
    "#2f6db3", "#d1603d", "#3a7d44", "#7b5ca6", "#8a8a8a", "#2a8a84")
INK, GRID = "#333333", "#d9d9d9"

GRIDS = {
    "ridge_anch": [{"alpha": a} for a in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)],
    "rf_anch": [{"max_depth": d} for d in (4, 8, 12)],
    "gbdt_anch": [{"learning_rate": lr, "max_iter": it}
                  for lr in (0.03, 0.1) for it in (200, 400)],
    "ridge_b_anch": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "gbdt_b_anch": [{"learning_rate": 0.05, "max_iter": 300}],
    "mlp_b_anch": [{"hidden_layer_sizes": (64, 32)}],
}

LABELS = {"ridge_anch": "A: Ridge", "rf_anch": "A: RandomForest",
          "gbdt_anch": "A: GBDT", "lstm_anch": "A: LSTM",
          "ridge_b_anch": "B: Ridge", "gbdt_b_anch": "B: GBDT",
          "mlp_b_anch": "B: MLP(robust)"}


def factory(name, params):
    if name.startswith("ridge"):
        return lambda: Ridge(random_state=SEED, **params)
    if name.startswith("rf"):
        return lambda: RandomForestRegressor(n_estimators=300, random_state=SEED,
                                             n_jobs=1, **params)
    if name.startswith("gbdt"):
        return lambda: HistGradientBoostingRegressor(random_state=SEED, **params)
    if name.startswith("mlp"):
        return lambda: MLPRegressor(max_iter=2000, early_stopping=True,
                                    random_state=SEED, **params)
    raise ValueError(name)


def decode_price(panel, preds: pd.Series, route: str) -> pd.Series:
    idx = preds.dropna().index
    p = preds.loc[idx].clip(-RATIO_CLIP, RATIO_CLIP)
    if route == "A":
        sigma = (panel.loc[idx, "sigma_ewma"] * np.exp(p)).clip(lower=SIGMA_FLOOR)
        S = panel.loc[idx, "jpm_close"].to_numpy()
        return pd.Series(chooser_closed_form(
            S0=S, K=S, r=panel.loc[idx, "r_1y"].to_numpy(),
            q=panel.loc[idx, "div_yield_ttm"].to_numpy(),
            sigma=sigma.to_numpy(), t1=T1, T2=T2), index=idx)
    return panel.loc[idx, "price_sigma_ewma"] * np.exp(p)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    data = build()
    panel = data.panel
    X = ml_features(data)
    y_a = np.log(panel["sigma_oracle"] / panel["sigma_ewma"]).rename("y_a")
    y_b = np.log(panel["price_sigma_oracle"] / panel["price_sigma_ewma"]).rename("y_b")

    # ---------------- stage 1: tuning on validation ----------------------
    print("== stage 1: walk-forward tuning on validation ==")
    oracle_price, regime = panel["price_sigma_oracle"], panel["fear_regime"]
    best, tuning_log = {}, {}
    for name, grid in GRIDS.items():
        route = "A" if not name.startswith(("ridge_b", "gbdt_b", "mlp_b")) else "B"
        target = y_a if route == "A" else y_b
        standardize = name.startswith(("ridge", "mlp"))
        clip = 5.0 if name.startswith("mlp") else None
        scores = []
        for params in grid:
            wf = walk_forward(data, X, target, data.idx_val,
                              factory(name, params), standardize, clip)
            price = decode_price(panel, wf.preds, route)
            mae = float((price - oracle_price.loc[price.index]).abs().mean())
            scores.append((mae, params))
        scores.sort(key=lambda t: t[0])
        best[name] = scores[0][1]
        tuning_log[name] = [{"params": p, "val_MAE": m} for m, p in scores]
        print(f"  {LABELS[name]:>16s}: best {best[name]}  val MAE {scores[0][0]:.3f}")

    wf_lstm_val = walk_forward_lstm(data, X, y_a, data.idx_val,
                                    seed=SEED, epochs=20, batch_size=128)
    mae_lstm = float((decode_price(panel, wf_lstm_val.preds, "A")
                      - oracle_price.loc[wf_lstm_val.preds.dropna().index]).abs().mean())
    print(f"  {LABELS['lstm_anch']:>16s}: fixed config  val MAE {mae_lstm:.3f}")

    # ---------------- stage 2: OPEN TEST SET (once) ----------------------
    print("\n== stage 2: final walk-forward evaluation on TEST (opened once) ==")
    test = data.idx_test
    ref_suite = metric_suite(panel.loc[test, "price_sigma_ewma"],
                             oracle_price.loc[test], regime.loc[test])
    suites, composites, test_prices, test_preds = {}, {}, {}, {}
    for name in GRIDS:
        route = "A" if not name.startswith(("ridge_b", "gbdt_b", "mlp_b")) else "B"
        target = y_a if route == "A" else y_b
        standardize = name.startswith(("ridge", "mlp"))
        clip = 5.0 if name.startswith("mlp") else None
        wf = walk_forward(data, X, target, test, factory(name, best[name]),
                          standardize, clip)
        price = decode_price(panel, wf.preds, route)
        test_prices[name], test_preds[name] = price, wf.preds
        s = metric_suite(price, oracle_price.loc[price.index], regime.loc[price.index])
        suites[name] = s
        composites[name] = composite_score(s, ref_suite)

    wf_lstm = walk_forward_lstm(data, X, y_a, test, seed=SEED, epochs=20,
                                batch_size=128)
    price = decode_price(panel, wf_lstm.preds, "A")
    test_prices["lstm_anch"], test_preds["lstm_anch"] = price, wf_lstm.preds
    s = metric_suite(price, oracle_price.loc[price.index], regime.loc[price.index])
    suites["lstm_anch"] = s
    composites["lstm_anch"] = composite_score(s, ref_suite)

    for name, s in sorted(suites.items(), key=lambda kv: composites[kv[0]]):
        print(f"  {LABELS[name]:>16s}: MAE {s['MAE']:.2f}  RMSE {s['RMSE']:.2f}  "
              f"Bias {s['Bias']:+.2f}  P95 {s['P95AE']:.2f}  R2 {s['R2']:+.3f}  "
              f"Hit10 {s['Hit10']:.1%}  ->  composite {composites[name]:.3f}")
    print(f"  {'EWMA baseline':>16s}: MAE {ref_suite['MAE']:.2f}  "
          f"R2 {ref_suite['R2']:+.3f}  Hit10 {ref_suite['Hit10']:.1%}  ->  1.000")

    sens = weight_sensitivity({**suites, "EWMA": ref_suite}, "EWMA")
    stable_winner = {s_: v["ranking"][0] for s_, v in sens.items()}
    print(f"  winner by weighting scheme: {stable_winner}")

    # ---------------- stage 3: SHAP on deployment GBDT -------------------
    print("\n== stage 3: SHAP (deployment GBDT, route A anchored) ==")
    import shap
    deploy_train = panel.index[: len(panel) - HORIZON]
    deploy_train = deploy_train[deploy_train.isin(y_a.dropna().index)]
    gbdt_final = HistGradientBoostingRegressor(random_state=SEED,
                                               **best["gbdt_anch"])
    gbdt_final.fit(X.loc[deploy_train].to_numpy(), y_a.loc[deploy_train].to_numpy())
    explainer = shap.TreeExplainer(gbdt_final)
    sv = explainer(X.loc[test].to_numpy())
    sv.feature_names = list(X.columns)

    plt.figure(figsize=(8.4, 5.6), dpi=200)
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP beeswarm — GBDT on ln(sigma/sigma_EWMA), test period",
              fontsize=10, color=INK)
    plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.6, 4.8), dpi=200)
    shap.plots.bar(sv, max_display=12, show=False)
    plt.title("SHAP mean |value| — feature importance", fontsize=10, color=INK)
    plt.tight_layout(); plt.savefig(FIG / "shap_bar.png", bbox_inches="tight")
    plt.close()
    shap_rank = pd.Series(np.abs(sv.values).mean(0), index=X.columns
                          ).sort_values(ascending=False)
    print("  top-5 SHAP:", ", ".join(f"{k}({v:.4f})" for k, v in shap_rank.head(5).items()))

    # ---------------- deployment models + figures + summary --------------
    for name in GRIDS:
        route = "A" if not name.startswith(("ridge_b", "gbdt_b", "mlp_b")) else "B"
        target = (y_a if route == "A" else y_b).loc[deploy_train]
        Xtr = X.loc[deploy_train]
        if name.startswith(("ridge", "mlp")):
            sc = StandardScaler().fit(Xtr)
            Xm = sc.transform(Xtr)
            if name.startswith("mlp"):
                Xm = np.clip(Xm, -5, 5)
            m = factory(name, best[name])().fit(Xm, target.to_numpy())
            joblib.dump({"model": m, "scaler": sc, "params": best[name]},
                        MODELS / f"final_{name}.joblib", compress=3)
        else:
            m = factory(name, best[name])().fit(Xtr.to_numpy(), target.to_numpy())
            joblib.dump({"model": m, "params": best[name]},
                        MODELS / f"final_{name}.joblib", compress=3)

    # figure: test price series
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    ax.plot(test, oracle_price.loc[test], color=GRAY, linewidth=1.6, label="oracle")
    ax.plot(test, panel.loc[test, "price_sigma_ewma"], color=INK, linewidth=1.0,
            linestyle="--", label="EWMA baseline")
    for name, c in (("ridge_anch", BLUE), ("gbdt_anch", PURPLE), ("lstm_anch", TEAL)):
        pr = test_prices[name]
        ax.plot(pr.index, pr.values, color=c, linewidth=1.2, label=LABELS[name])
    ax.set_title("Test set (opened once): walk-forward prices vs oracle",
                 fontsize=10, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "test_prices.png", bbox_inches="tight")
    plt.close(fig)

    # figure: composite bar
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=200)
    names = sorted(composites, key=composites.get)
    vals = [composites[n] for n in names]
    ax.barh([LABELS[n] for n in names], vals,
            color=[GREEN if v < 1 else BLUE for v in vals], height=0.6)
    ax.axvline(1.0, color=ORANGE, linewidth=1.4)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_title("Weighted composite on TEST (lower = better; <1 beats EWMA baseline)",
                 fontsize=10, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "test_composite.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "best_params": best,
        "tuning_log": tuning_log,
        "lstm_val_MAE": mae_lstm,
        "test_suites": suites,
        "ewma_reference_test": ref_suite,
        "composites_test": composites,
        "weight_sensitivity_test": {k: v["ranking"] for k, v in sens.items()},
        "shap_rank": shap_rank.round(5).to_dict(),
        "test_window": [str(test.min().date()), str(test.max().date()), len(test)],
    }
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(suites).T.assign(composite=pd.Series(composites)).to_csv(
        OUT / "final_comparison.csv")
    print(f"\nfigures -> {FIG}\nmodels -> models/final_*.joblib\nsummary -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
