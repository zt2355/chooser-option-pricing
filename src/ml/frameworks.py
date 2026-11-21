"""Week 5 initial model frameworks (default-leaning hyperparameters).

Route A (volatility prediction -> BSM pricing):
  ridge_a        Ridge on pruned feature set (collinearity handled per W2)
  rf_a           RandomForest, 300 trees
  gbdt_a         HistGradientBoosting (sklearn; XGBoost-equivalent GBDT —
                 XGBoost itself needs the libomp system library, unavailable
                 on this machine; substitution documented in the design doc)
  lstm_a         small LSTM over 63-day feature sequences (src/ml/lstm.py)

Route B (end-to-end pricing of price/spot):
  ridge_b, gbdt_b, mlp_b (sklearn MLPRegressor)

All models expose fit/predict through `train_and_predict`, which returns
predictions for train/val/test so Week 6 can reuse the same entry point.
Hyperparameter tuning is deliberately deferred to Week 6.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from src.ml.dataset import LINEAR_COLS, MLData, make_sequences, scaled
from src.ml.lstm import train_lstm

SEED = 42


@dataclass
class ModelResult:
    name: str
    route: str
    preds: dict = field(default_factory=dict)   # split -> pd.Series


def _sk_models_route_a():
    return {
        "ridge_a": (Ridge(alpha=1.0, random_state=SEED), LINEAR_COLS, True),
        "rf_a": (RandomForestRegressor(n_estimators=300, max_depth=12,
                                       random_state=SEED, n_jobs=1), None, False),
        "gbdt_a": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 random_state=SEED), None, False),
    }


def _sk_models_route_b():
    return {
        "ridge_b": (Ridge(alpha=1.0, random_state=SEED), LINEAR_COLS, True),
        "gbdt_b": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                 random_state=SEED), None, False),
        "mlp_b": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                               early_stopping=True, random_state=SEED), None, True),
    }


def _fit_predict_sklearn(data: MLData, model, cols, use_scaled, target: pd.Series,
                         name: str, route: str) -> ModelResult:
    def matrix(idx):
        if use_scaled:
            return scaled(data, idx, cols)
        return data.X.loc[idx, cols if cols is not None else list(data.X.columns)].to_numpy()

    model.fit(matrix(data.idx_train), target.loc[data.idx_train].to_numpy())
    res = ModelResult(name, route)
    for split, idx in (("train", data.idx_train), ("val", data.idx_val),
                       ("test", data.idx_test)):
        res.preds[split] = pd.Series(model.predict(matrix(idx)), index=idx, name=name)
    res.fitted = model
    return res


def _sk_models_anchored():
    return {
        "ridge_anch": (Ridge(alpha=1.0, random_state=SEED), LINEAR_COLS, True),
        "rf_anch": (RandomForestRegressor(n_estimators=300, max_depth=12,
                                          random_state=SEED, n_jobs=1), None, False),
        "gbdt_anch": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                    random_state=SEED), None, False),
    }


def train_and_predict(data: MLData) -> dict[str, ModelResult]:
    out: dict[str, ModelResult] = {}
    for name, (m, cols, sc) in _sk_models_route_a().items():
        out[name] = _fit_predict_sklearn(data, m, cols, sc, data.y_sigma, name, "A")
    # anchored variants (route A design fix): predict log(sigma/EWMA);
    # route marker "A_anch" tells the evaluator to decode via EWMA*exp(pred)
    for name, (m, cols, sc) in _sk_models_anchored().items():
        out[name] = _fit_predict_sklearn(data, m, cols, sc, data.y_sigma_ratio,
                                         name, "A_anch")
    for name, (m, cols, sc) in _sk_models_route_b().items():
        out[name] = _fit_predict_sklearn(data, m, cols, sc, data.y_price_rel, name, "B")

    # LSTM (route A)
    seq_train, d_train = make_sequences(data, data.idx_train)
    seq_val, d_val = make_sequences(data, data.idx_val)
    seq_test, d_test = make_sequences(data, data.idx_test)
    y_tr = data.y_sigma.loc[d_train].to_numpy()
    preds = train_lstm(seq_train, y_tr, {"val": seq_val, "test": seq_test},
                       seed=SEED)
    res = ModelResult("lstm_a", "A")
    res.preds["train"] = pd.Series(preds["train"], index=d_train, name="lstm_a")
    res.preds["val"] = pd.Series(preds["val"], index=d_val, name="lstm_a")
    res.preds["test"] = pd.Series(preds["test"], index=d_test, name="lstm_a")
    out["lstm_a"] = res
    return out
