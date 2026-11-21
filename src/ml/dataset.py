"""Week 5 ML dataset: features, targets, purged 70/15/15 time-series split.

Targets (both defined on the Week 4 pricing panel, so evaluation is
same-panel/same-ruler as the baseline benchmark):
  route A: sigma_oracle  — vol actually realized over the option's 1y life
  route B: price_rel     — oracle ATM chooser price / spot (dimensionless)

Split design (chronological, no shuffling):
  70/15/15 by row count -> train / val / test.
  PURGE: the last 252 train rows are dropped so that no training label's
  forward window crosses the validation boundary (labels look 252 trading
  days ahead). The test set is naturally clean: its labels use 2025 prices
  that never appear in any model input (features end 2024-12).
  Documented caveat: validation labels overlap the test feature period —
  affects hyperparameter selection only, not test-set integrity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.analysis_w4 import build_panel, price_panel
from src.preprocess import FEATURE_COLS

HORIZON = 252          # label look-ahead in trading days
PURGE = HORIZON        # rows removed from the train tail

# for linear models: drop one member of each |rho|>0.95 pair (W2 finding)
LINEAR_DROP = ["r_3m", "parkinson_vol_21d"]
LINEAR_COLS = [c for c in FEATURE_COLS if c not in LINEAR_DROP]


@dataclass
class MLData:
    panel: pd.DataFrame          # full priced panel (with oracle price)
    X: pd.DataFrame              # 18 raw features
    y_sigma: pd.Series           # route A target (vol level)
    y_price_rel: pd.Series      # route B target
    y_sigma_ratio: pd.Series    # route A anchored target: log(oracle/EWMA)
    idx_train: pd.DatetimeIndex  # purged train
    idx_val: pd.DatetimeIndex
    idx_test: pd.DatetimeIndex
    scaler: StandardScaler       # fitted on (purged) train X only


def build() -> MLData:
    panel = price_panel(build_panel())

    X = panel[FEATURE_COLS].copy()
    y_sigma = panel["sigma_oracle"].rename("y_sigma")
    y_price_rel = (panel["price_sigma_oracle"] / panel["jpm_close"]).rename("y_price_rel")
    # anchored target (W5 design fix): predict the log-adjustment to the EWMA
    # anchor instead of the vol level. A model that outputs 0 reproduces the
    # EWMA baseline exactly, so it cannot be pushed outside the current
    # regime by a train/val level shift.
    y_sigma_ratio = np.log(panel["sigma_oracle"] / panel["sigma_ewma"]).rename("y_sigma_ratio")

    n = len(panel)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    idx_train_full = panel.index[:n_train]
    idx_val = panel.index[n_train:n_train + n_val]
    idx_test = panel.index[n_train + n_val:]
    idx_train = idx_train_full[:-PURGE]          # purged

    scaler = StandardScaler().fit(X.loc[idx_train])
    return MLData(panel, X, y_sigma, y_price_rel, y_sigma_ratio,
                  idx_train, idx_val, idx_test, scaler)


def scaled(data: MLData, idx: pd.DatetimeIndex,
           cols: list[str] | None = None) -> np.ndarray:
    """Standardized feature matrix (scaler fitted on train only)."""
    Xs = pd.DataFrame(data.scaler.transform(data.X),
                      index=data.X.index, columns=data.X.columns)
    return Xs.loc[idx, cols if cols is not None else list(data.X.columns)].to_numpy()


def make_sequences(data: MLData, idx: pd.DatetimeIndex,
                   lookback: int = 63) -> tuple[np.ndarray, np.ndarray]:
    """LSTM input: for each date in idx, the standardized feature rows of the
    previous `lookback` trading days (inclusive). Sequences may reach back
    into earlier calendar periods — legitimate, features are lagged
    observables. Dates whose lookback precedes the panel start are dropped."""
    Xs = pd.DataFrame(data.scaler.transform(data.X),
                      index=data.X.index, columns=data.X.columns)
    pos = {d: i for i, d in enumerate(data.X.index)}
    seqs, kept = [], []
    for d in idx:
        i = pos[d]
        if i - lookback + 1 < 0:
            continue
        seqs.append(Xs.iloc[i - lookback + 1:i + 1].to_numpy())
        kept.append(d)
    return np.stack(seqs), np.asarray(kept)
