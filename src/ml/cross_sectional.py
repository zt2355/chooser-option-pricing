"""Cross-sectional ML reference model (Week 12) — the model whose increment was
tested in docs/week12/significance.json (H2, pre-registered).

Model      GARCH-1d anchor + HistGradientBoosting on the CROSS-SECTIONAL part of
           y = log(sigma_oracle / sigma_garch_1d): target demeaned by date,
           features z-scored within each date (clip +-5), NO sector dummies
           (they are look-ahead: sectors come from today's constituent table).
           Hyper-parameters exactly as Week 11 / Week 12 (max_iter 200,
           learning_rate 0.05, max_depth 4, min_samples_leaf 200, seed 0).
Training   every row of the S&P 500 point-in-time research panel with a GARCH
           anchor (2020-12..2024-12, labels to 2025-12). No row is held out: the
           out-of-sample evidence is the walk-forward test in analysis_w12; this
           file only freezes the deployed model.
Serving    sigma_ml = sigma_garch_1d * exp(clip(prediction, +-1.5)), with the
           features z-scored over the SAME DATE's cross-section of point-in-time
           S&P 500 members that have a GARCH anchor and complete features.
           Scope = those members; everything else (S&P 400/600 scale-test names,
           EWMA-fallback names, file mode without a database) gets NaN.

Usage: python -m src.ml.cross_sectional --train      (writes models/w12_gbdt_cs_garch1d.joblib)
"""

import hashlib
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config

MODEL_PATH = config.PROJECT_ROOT / "models" / "w12_gbdt_cs_garch1d.joblib"
MODEL_VERSION = "w12-gbdt_cs_garch1d"
RATIO_CLIP, Z_CLIP = 1.5, 5.0


def zscore_within_date(df: pd.DataFrame, features: list[str], date_col: str = "date") -> pd.DataFrame:
    """Serving-side copy of analysis_w11.cross_sectional's feature transform
    (identity of the two is asserted in analysis/week12)."""
    X = df[features].astype(float)
    mu = X.groupby(df[date_col]).transform("mean")
    sd = X.groupby(df[date_col]).transform("std").replace(0, np.nan)
    return ((X - mu) / sd).fillna(0.0).clip(-Z_CLIP, Z_CLIP)


def train_final() -> dict:
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    import src.analysis_w11 as w
    p = w.panel_round2().dropna(subset=["sigma_garch_1d"]).reset_index(drop=True)
    X, y_cs, _ = w.cross_sectional(p, "sigma_garch_1d", sector_dummies=False)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=200, random_state=0).fit(X.to_numpy(float), y_cs)
    bundle = {"model": model, "features": list(X.columns), "ratio_clip": RATIO_CLIP, "z_clip": Z_CLIP,
              "train_rows": int(len(p)), "train_tickers": int(p["ticker"].nunique()),
              "train_start": str(p["date"].min().date()), "train_end": str(p["date"].max().date()),
              "target": "log(sigma_oracle / sigma_garch_1d) demeaned by date", "version": MODEL_VERSION}
    joblib.dump(bundle, MODEL_PATH)
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    try:                                            # register when a database is configured
        from src.db.engine import db_available
        if db_available():
            import sqlalchemy as sa
            from src.db import schema
            from src.db.engine import get_engine
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            row = {"name": MODEL_VERSION, "path": str(MODEL_PATH.relative_to(config.PROJECT_ROOT)), "sha256": sha, "role": "reference"}
            with get_engine().begin() as c:
                stmt = pg_insert(schema.model_registry).values(row)
                c.execute(stmt.on_conflict_do_update(index_elements=["name"], set_={k: stmt.excluded[k] for k in ("path", "sha256", "role")}))
    except Exception as e:                          # registration is bookkeeping, not part of the model
        print("model_registry not updated:", e)
    return {k: v for k, v in bundle.items() if k != "model"} | {"sha256": sha}


@lru_cache(maxsize=1)
def bundle():
    import joblib
    return joblib.load(MODEL_PATH)


def cross_section(date=None) -> pd.DataFrame:
    """ML reference for the in-scope cross-section on `date` (latest stored date if None).
    Returns index=ticker, columns ratio, sigma_ml; empty without a database."""
    from src.db.engine import database_url, db_available
    if not (database_url() and db_available()):
        return pd.DataFrame(columns=["ratio", "sigma_ml"])
    from src.db import repository as repo
    from src.universe.sp500 import universe_on
    b = bundle()
    mem = repo.read_membership("sp500")
    rows = repo.read_latest_features(None, None)
    if date is not None and len(rows) and pd.Timestamp(rows["date"].max()) != pd.Timestamp(date):
        rows = repo.read_latest_features(None, date)              # historical as-of: scan path
    if not len(rows):
        return pd.DataFrame(columns=["ratio", "sigma_ml"])
    d = pd.Timestamp(date) if date is not None else pd.Timestamp(rows["date"].max())
    members = universe_on(mem, d)
    rows = rows[(pd.to_datetime(rows["date"]) == d) & rows.index.isin(members)]
    rows = rows[np.isfinite(rows[b["features"] + ["sigma_garch_1d"]].to_numpy(float)).all(axis=1)]
    if not len(rows):
        return pd.DataFrame(columns=["ratio", "sigma_ml"])
    Xz = zscore_within_date(rows.assign(date=d), b["features"])
    ratio = np.clip(b["model"].predict(Xz.to_numpy(float)), -b["ratio_clip"], b["ratio_clip"])
    out = pd.DataFrame({"ratio": ratio, "sigma_ml": np.maximum(rows["sigma_garch_1d"].to_numpy(float) * np.exp(ratio), 0.01)},
                       index=rows.index)
    out.attrs["date"] = d
    return out


if __name__ == "__main__":
    if "--train" in sys.argv:
        print(train_final())
