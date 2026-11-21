"""Week 6 walk-forward (expanding-window) retraining with purge.

Motivation (W5 finding): one-shot training on a COVID-dominated window fails
under regime shift. Walk-forward retrains quarterly on ALL data available at
prediction time, mirroring real deployment.

Purge rule per block: a block predicting panel positions [p, p+63) may train
only on rows at positions <= p - HORIZON - 1, so no training label's forward
window overlaps the prediction block. The scaler is refit on each block's
training slice (no future statistics leak).

Feature set (W6 fix): rate LEVEL features replaced by deviations from their
own trailing 252d mean (r_1y_dev, r_3m dropped), eliminating the
out-of-distribution leaf problem that broke tree models in Week 5.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.dataset import HORIZON, MLData

BLOCK = 63  # quarterly refit

# W6 ML feature set: stationary-ized rates
W6_DROP = ["r_3m", "r_1y", "parkinson_vol_21d"]
W6_ADD = ["r_1y_dev"]


def ml_features(data: MLData) -> pd.DataFrame:
    """Transformed feature matrix for W6 models (rate levels -> deviations)."""
    X = data.X.copy()
    # deviation from trailing 1y mean, computed on the full panel history
    X["r_1y_dev"] = X["r_1y"] - X["r_1y"].rolling(252, min_periods=1).mean()
    return X.drop(columns=[c for c in W6_DROP if c in X.columns])


@dataclass
class WFResult:
    preds: pd.Series          # walk-forward predictions over eval_idx
    n_refits: int


def walk_forward(data: MLData, X: pd.DataFrame, y: pd.Series,
                 eval_idx: pd.DatetimeIndex, make_model,
                 standardize: bool, clip: float | None = None) -> WFResult:
    """Generic walk-forward for sklearn-style models.

    make_model: zero-arg factory returning a fresh unfitted model.
    standardize: refit a StandardScaler on each block's train slice.
    clip: optional post-standardization clipping at +/- clip (MLP robustness).
    """
    panel_pos = {d: i for i, d in enumerate(data.panel.index)}
    eval_pos = np.array([panel_pos[d] for d in eval_idx])

    preds = pd.Series(index=eval_idx, dtype=float)
    n_refits = 0
    for start in range(eval_pos.min(), eval_pos.max() + 1, BLOCK):
        block_mask = (eval_pos >= start) & (eval_pos < start + BLOCK)
        if not block_mask.any():
            continue
        block_dates = eval_idx[block_mask]
        train_dates = data.panel.index[: max(start - HORIZON, 0)]
        train_dates = train_dates[train_dates.isin(y.dropna().index)]

        Xtr, ytr = X.loc[train_dates], y.loc[train_dates]
        Xbl = X.loc[block_dates]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr = pd.DataFrame(sc.transform(Xtr), index=Xtr.index, columns=Xtr.columns)
            Xbl = pd.DataFrame(sc.transform(Xbl), index=Xbl.index, columns=Xbl.columns)
            if clip is not None:
                Xtr, Xbl = Xtr.clip(-clip, clip), Xbl.clip(-clip, clip)

        model = make_model()
        model.fit(Xtr.to_numpy(), ytr.to_numpy())
        preds.loc[block_dates] = model.predict(Xbl.to_numpy())
        n_refits += 1
    return WFResult(preds=preds, n_refits=n_refits)


def walk_forward_lstm(data: MLData, X: pd.DataFrame, y: pd.Series,
                      eval_idx: pd.DatetimeIndex, lookback: int = 63,
                      **lstm_kwargs) -> WFResult:
    """Walk-forward for the LSTM (sequences built after per-block scaling)."""
    from src.ml.lstm import train_lstm

    panel_pos = {d: i for i, d in enumerate(data.panel.index)}
    eval_pos = np.array([panel_pos[d] for d in eval_idx])
    all_idx = data.panel.index

    preds = pd.Series(index=eval_idx, dtype=float)
    n_refits = 0

    def sequences(Xs: pd.DataFrame, dates) -> tuple[np.ndarray, list]:
        seqs, kept = [], []
        for d in dates:
            i = panel_pos[d]
            if i - lookback + 1 < 0:
                continue
            seqs.append(Xs.iloc[i - lookback + 1: i + 1].to_numpy())
            kept.append(d)
        return (np.stack(seqs) if seqs else np.empty((0, lookback, X.shape[1]))), kept

    for start in range(eval_pos.min(), eval_pos.max() + 1, BLOCK):
        block_mask = (eval_pos >= start) & (eval_pos < start + BLOCK)
        if not block_mask.any():
            continue
        block_dates = eval_idx[block_mask]
        train_dates = all_idx[: max(start - HORIZON, 0)]
        train_dates = train_dates[train_dates.isin(y.dropna().index)]

        sc = StandardScaler().fit(X.loc[train_dates])
        Xs = pd.DataFrame(sc.transform(X), index=X.index, columns=X.columns)

        seq_tr, kept_tr = sequences(Xs, train_dates)
        seq_bl, kept_bl = sequences(Xs, block_dates)
        out = train_lstm(seq_tr, y.loc[kept_tr].to_numpy(), {"block": seq_bl},
                         **lstm_kwargs)
        preds.loc[kept_bl] = out["block"]
        n_refits += 1
    return WFResult(preds=preds, n_refits=n_refits)
