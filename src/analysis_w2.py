"""Week 2 supplementary analysis (mentor feedback):
correlation heatmap + feature-importance charts for the 18 features.

Importance target = forward 21d realized volatility (the quantity the
Week 5 ML route A will predict). Two independent lenses:
  - Random Forest impurity importance (nonlinear, interaction-aware)
  - Mutual information (model-free dependence)

Usage:  python -m src.analysis_w2      (from project root)
Output: docs/week2/figures/*.png  + console stats used in the report.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from src import config
from src.preprocess import FEATURE_COLS

FIG_DIR = config.PROJECT_ROOT / "docs" / "week2" / "figures"
STUDY_END = "2024-12-31"

# palette: single sequential hue for magnitude, diverging blue/red for polarity
BLUE = "#2f6db3"
GRID = "#d9d9d9"
INK = "#333333"


def load() -> pd.DataFrame:
    df = pd.read_csv(
        config.DATA_PROCESSED / "features_daily_model_ready.csv",
        index_col="date", parse_dates=True,
    )
    return df.loc[:STUDY_END]


def corr_heatmap(X: pd.DataFrame) -> pd.DataFrame:
    corr = X.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(9.6, 8.4), dpi=200)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=7.5, color=INK)
    ax.set_yticks(range(n), corr.columns, fontsize=7.5, color=INK)
    # annotate only meaningful cells to keep the figure readable
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if abs(v) >= 0.25 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.2,
                        color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pearson correlation", fontsize=9, color=INK)
    ax.set_title("Feature correlation matrix (model-ready set, 2019-10 .. 2024-12)",
                 fontsize=11, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corr_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return corr


def importance_charts(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)

    mi = pd.Series(
        mutual_info_regression(X, y, random_state=42), index=X.columns
    )
    mi = mi / mi.sum()  # normalize for comparability

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), dpi=200, sharey=False)
    for ax, s, title in (
        (axes[0], rf_imp.sort_values(), "Random Forest importance"),
        (axes[1], mi.sort_values(), "Mutual information (normalized)"),
    ):
        ax.barh(s.index, s.values, color=BLUE, height=0.62)
        ax.set_title(title + "\ntarget: forward 21d realized volatility",
                     fontsize=10, color=INK)
        ax.tick_params(labelsize=8, colors=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # direct labels on the top-3 bars only (selective labeling)
        for name in s.index[-3:]:
            ax.text(s[name], name, f" {s[name]:.3f}", va="center",
                    fontsize=7.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame({"rf": rf_imp, "mi": mi}).sort_values("rf", ascending=False)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    X = df[FEATURE_COLS].copy()
    y = df["realized_vol_21d"].shift(-21).rename("fwd_vol_21d")
    keep = y.notna()
    X, y = X[keep], y[keep]
    print(f"sample: {len(X)} rows ({X.index.min().date()} .. {X.index.max().date()})")

    corr = corr_heatmap(X)
    au = corr.abs().where(~np.eye(len(corr), dtype=bool))
    pairs = au.unstack().dropna().sort_values(ascending=False)
    print("\nTop |corr| pairs (dedup):")
    seen = set()
    for (a, b), v in pairs.items():
        if frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        print(f"  {a:>20s} ~ {b:<20s} {v:+.3f}")
        if len(seen) >= 8:
            break

    imp = importance_charts(X, y)
    print("\nImportance (RF | MI):")
    for name, row in imp.iterrows():
        print(f"  {name:>20s}  {row['rf']:.3f} | {row['mi']:.3f}")

    print("\ncorr(feature, target) top5:")
    tc = X.corrwith(y).abs().sort_values(ascending=False).head(5)
    for k, v in tc.items():
        print(f"  {k:>20s}  |r|={v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
