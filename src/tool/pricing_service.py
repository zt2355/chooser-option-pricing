"""Pricing service backing the Streamlit tool (Week 7-9).

Pure functions, no UI: the Streamlit app is a thin wrapper so that every
piece of tool logic is importable and auditable from analysis scripts.

Pricing policy (Week 12: primary anchor switched to GARCH-1d):
  PRIMARY   GARCH(1,1) 1-day-anchored BSM chooser (PRIMARY_ANCHOR, default
            "garch1d"); a ticker with < 750 returns has no GARCH state yet and
            falls back to EWMA row by row - every quote carries `anchor`
            Week 7-11: EWMA-anchored (PRIMARY_ANCHOR=ewma gives that primary price;
            the ML reference is the Week 12 model in both modes)
  REFERENCE Week 12: cross-sectional GBDT on the GARCH-1d anchor
            (models/w12_gbdt_cs_garch1d.joblib, src/ml/cross_sectional.py), shown for
            point-in-time S&P 500 members with a GARCH anchor; its increment over the
            anchor is significant on the 2023-24 out-of-sample panel (cross-sectionally,
            conditional on that period; docs/week12/significance.json).
            Weeks 7-11: the W6 EWMA-anchored Ridge, JPM only.
  BAND      +/- regime-conditional MAE of the quote's anchor (Week 4 definition;
            GARCH-1d bands from docs/week12/bands.json, window 2020-12..2024-12)

Week 9 changes (online stack):
  * `load_market_state(ticker, as_of, source)` reads the feature history from
    PostgreSQL when DATABASE_URL is reachable, else from the Week 2 files.
    Both paths run the same arithmetic; analysis/week9 audit 2 asserts the
    quotes agree to 1e-10.
  * `quote` / `scenario_table` are memoised in Redis (1 h TTL, invalidated by
    the daily pipeline). `cached=False` bypasses the cache for audits.
  * The reference model bundle is loaded once per process (lru_cache). The
    Week 7 code re-read the joblib on every quote; the Week 9 baseline
    measured that at only ~0.4 ms per call (S4 vs S5), so this is hygiene,
    not the bottleneck — the panel-pricing loop (S8) and the full feature
    rebuild in the daily job (S9) are.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import features as F
from src.cache import redis_cache as cache
from src.metrics import WEIGHTS  # noqa: F401  (re-exported for the app's metrics tab)
from src.ml import cross_sectional as xs
from src.models.chooser import chooser_closed_form
from src.models.bsm import bs_call, bs_put

EWMA_LAMBDA = 0.94
SIGMA_FLOOR, RATIO_CLIP = 0.01, 1.5
# Week 10 (universe): a trailing dividend yield above Q_CAP is a special or
# liquidating distribution (e.g. AIV 2026: 228%), not a run-rate yield. The
# pricing input is capped; the raw value stays in features_daily.
Q_CAP = 0.25
# Week 12: primary anchor. "garch1d" (default) or "ewma" (the Week 7-11 tool).
PRIMARY_ANCHOR = os.environ.get("PRIMARY_ANCHOR", "garch1d")
if PRIMARY_ANCHOR not in ("garch1d", "ewma"):
    raise ValueError(f"PRIMARY_ANCHOR must be garch1d or ewma, got {PRIMARY_ANCHOR!r}")
# Pricing MAE vs oracle by VIX regime on the JPM oracle panel (USD), Week 4 definition.
#   ewma    : Week 4 benchmark, 1310 days 2019-10..2024-12 (reproduced in docs/week12/bands.json)
#   garch1d : docs/week12/bands.json, 1011 days 2020-12..2024-12 (GARCH needs 750 obs). This
#             window does NOT contain the 2020 crash, so these bands are likely too narrow in a
#             crisis; on the same window EWMA's bands are 6.66 / 4.89.
BAND_MAE_BY_ANCHOR = {"ewma": {"calm": 9.01, "fear": 10.83},
                      "garch1d": {"calm": 5.48, "fear": 5.86}}
BAND_MAE = BAND_MAE_BY_ANCHOR[PRIMARY_ANCHOR]          # bands of the configured primary anchor
MODEL_VERSION = xs.MODEL_VERSION
# The ML reference is shown ONLY inside the validated scope (mentor, 2026-09-23): Weeks 10-11
# that was {JPM}; Week 12 (increment established, pre-registered rule) it is the point-in-time
# S&P 500 cross-section of the quote date with a GARCH anchor - decided per quote by
# xs.cross_section(); everything else gets NaN instead of an out-of-distribution extrapolation.
ML_SCOPE = "point-in-time S&P 500 members with a GARCH-1d anchor (cross-section of the quote date)"

ML_FEATURES = ["log_return", "realized_vol_21d", "realized_vol_63d",
               "realized_vol_252d", "div_yield_ttm", "div_growth_yoy",
               "curve_slope", "rate_momentum_21d", "spx_log_return",
               "beta_63d", "corr_jpm_vix_63d", "vix_level",
               "vix_sentiment_pct", "vix_momentum_5d", "fear_regime", "r_1y_dev"]


@dataclass
class MarketState:
    ticker: str
    date: pd.Timestamp
    S: float
    sigma_ewma: float
    r_1y: float
    q: float
    vix: float
    vix_pct: float
    regime: str          # "calm" / "fear"
    features: pd.Series  # W6 ML feature row (for the reference model)
    source: str = "file"  # "db" / "file"
    sigma_garch_1d: float = float("nan")   # Week 12 primary anchor (NaN: < 750 returns)

    @property
    def anchor(self) -> str:
        return "garch1d" if PRIMARY_ANCHOR == "garch1d" and math.isfinite(self.sigma_garch_1d) else "ewma"

    @property
    def sigma_anchor(self) -> float:
        return self.sigma_garch_1d if self.anchor == "garch1d" else self.sigma_ewma


# ------------------------------------------------------------ data access

def _history_from_file(ticker: str) -> pd.DataFrame:
    if ticker != config.TICKER:
        raise ValueError(f"file source only holds {config.TICKER}; DB needed for {ticker}")
    return pd.read_csv(config.DATA_PROCESSED / "features_daily.csv",
                       index_col="date", parse_dates=True)


def _history_from_db(ticker: str) -> pd.DataFrame:
    from src.db import repository as repo
    return F.to_file_names(repo.read_features(ticker))


def resolve_source(source: str | None = None) -> str:
    """'db' if reachable (or forced), else 'file'."""
    if source in ("db", "file"):
        return source
    from src.db.engine import database_url, db_available
    return "db" if database_url() and db_available() else "file"


def load_history(ticker: str = config.TICKER, source: str | None = None) -> tuple[pd.DataFrame, str]:
    src = resolve_source(source)
    df = _history_from_db(ticker) if src == "db" else _history_from_file(ticker)
    return df, src


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """EWMA anchor and W6 rate-deviation feature (single definition, both sources)."""
    df = df.copy()
    r2 = df["log_return"] ** 2
    df["sigma_ewma"] = np.sqrt(r2.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean() * 252)
    df["r_1y_dev"] = df["r_1y"] - df["r_1y"].rolling(252, min_periods=1).mean()
    return df


def derive_garch(df: pd.DataFrame) -> pd.DataFrame:
    """Week 12 (file source): the stored-state GARCH-1d path over the whole history,
    the same function the database rebuild uses (src/garch_state.py)."""
    from src import garch_state as gs
    df = df.drop(columns=[c for c in gs.OUT_COLS if c in df.columns])     # DB history already carries them
    return df.join(gs.path(df["log_return"])[["sigma_garch_1d"]], how="left")


def _state_from_row(ticker: str, row: pd.Series, source: str) -> MarketState:
    regime = "fear" if (row["vix_sentiment_pct"] > 0.75 or row["vix_level"] > 30) else "calm"
    g = row.get("sigma_garch_1d", float("nan"))
    return MarketState(
        sigma_garch_1d=float(g) if pd.notna(g) else float("nan"),
        ticker=ticker, date=row.name, S=float(row["jpm_close"]),
        sigma_ewma=float(row["sigma_ewma"]), r_1y=float(row["r_1y"]),
        q=min(float(row["div_yield_ttm"]), Q_CAP), vix=float(row["vix_level"]),
        vix_pct=float(row["vix_sentiment_pct"]), regime=regime,
        features=row[ML_FEATURES], source=source,
    )


REQUIRED = ["jpm_close", "sigma_ewma", "r_1y", "div_yield_ttm", "vix_level", "vix_sentiment_pct"]


def load_market_state(as_of: str | None = None, ticker: str = config.TICKER,
                      source: str | None = None) -> MarketState:
    src = resolve_source(source)
    if src == "db":
        # Week 10: one-row read of the stored state (sigma_ewma, r_1y_dev live in
        # features_daily); falls back to the full-history derivation only when
        # a ticker has no stored state yet (pre-Week-10 rows).
        from src.db import repository as repo
        latest = repo.read_latest_features([ticker], as_of)
        if len(latest) and pd.notna(latest.loc[ticker, "sigma_ewma"]):
            row = F.to_file_names(latest).loc[ticker].copy()
            row.name = pd.Timestamp(latest.loc[ticker, "date"])
            return _state_from_row(ticker, row, "db")
    hist, src = load_history(ticker, src)
    df = derive_garch(derive(hist))
    ok = (df.loc[:as_of] if as_of else df).dropna(subset=REQUIRED + ML_FEATURES)
    if not len(ok):
        raise ValueError(f"{ticker}: no complete feature row (history shorter than the 252-day windows)")
    return _state_from_row(ticker, ok.iloc[-1], src)


# ------------------------------------------------------------ pricing

def price_chooser(S, K, r, q, sigma, t1, T2) -> float:
    return float(chooser_closed_form(S0=S, K=K, r=r, q=q,
                                     sigma=max(sigma, SIGMA_FLOOR), t1=t1, T2=T2))


def price_legs(S, K, r, q, sigma, T2) -> tuple[float, float]:
    sigma = max(sigma, SIGMA_FLOOR)
    return (float(bs_call(S, K, r, q, sigma, T2)),
            float(bs_put(S, K, r, q, sigma, T2)))


_reference_bundle = xs.bundle            # kept name (Week 9 perf script clears its cache)


def _cross_section(state: MarketState) -> pd.DataFrame:
    if state.source != "db":                # file mode: no cross-section, no ML reference
        return pd.DataFrame(columns=["ratio", "sigma_ml"])
    return xs.cross_section(state.date)


def ml_reference_sigma(state: MarketState) -> tuple[float, float]:
    """Week 12: (sigma_ml, ratio) of the cross-sectional GBDT for this state, NaN out of scope.
    sigma_ml = sigma_garch_1d * exp(ratio)."""
    cs = _cross_section(state)
    if state.ticker not in cs.index:
        return float("nan"), float("nan")
    return float(cs.loc[state.ticker, "sigma_ml"]), float(cs.loc[state.ticker, "ratio"])


def _compute_quote(state: MarketState, K: float, t1: float, T2: float) -> dict:
    sig = state.sigma_anchor
    base = price_chooser(state.S, K, state.r_1y, state.q, sig, t1, T2)
    sigma_ml, ratio = ml_reference_sigma(state)
    in_scope = math.isfinite(sigma_ml)
    ml = price_chooser(state.S, K, state.r_1y, state.q, sigma_ml, t1, T2) if in_scope else float("nan")
    call, put = price_legs(state.S, K, state.r_1y, state.q, sig, T2)
    band = BAND_MAE_BY_ANCHOR[state.anchor][state.regime]
    return {
        "ticker": state.ticker, "as_of": str(state.date.date()), "S": state.S, "K": K,
        "anchor": state.anchor, "sigma_anchor": sig, "sigma_garch_1d": state.sigma_garch_1d,
        "sigma_ewma": state.sigma_ewma, "sigma_ml": sigma_ml, "ml_ratio": ratio,
        "r": state.r_1y, "q": state.q, "regime": state.regime,
        "primary_price": base, "primary_band": (base - band, base + band),
        "ml_reference_price": ml,
        "call_leg": call, "put_leg": put, "straddle": call + put,
        "source": state.source, "model_version": MODEL_VERSION, "ml_in_scope": in_scope,
    }


def quote(state: MarketState, K: float | None = None,
          t1: float = 0.5, T2: float = 1.0, cached: bool = True) -> dict:
    """Full dual quote used by the app and the audits (Redis-memoised)."""
    K = state.S if K is None else K
    k = cache.key("quote", state.ticker, state.date.date(), round(K, 6), t1, T2, MODEL_VERSION, PRIMARY_ANCHOR)
    if cached:
        hit = cache.get_json(k)
        if hit is not None:
            hit["primary_band"] = tuple(hit["primary_band"])
            return hit
    q = _compute_quote(state, K, t1, T2)
    if cached:
        cache.set_json(k, q)
    return q


def scenario_table(state: MarketState, t1: float = 0.5, T2: float = 1.0,
                   cached: bool = True) -> pd.DataFrame:
    """Extreme-scenario grid required by the Week 7 spec."""
    k = cache.key("scenario", state.ticker, state.date.date(), t1, T2, PRIMARY_ANCHOR)
    if cached:
        hit = cache.get_json(k)
        if hit is not None:
            return pd.DataFrame(hit)
    base_p = price_chooser(state.S, state.S, state.r_1y, state.q,
                           state.sigma_anchor, t1, T2)
    scenarios = {
        "基准(当前市场)": dict(),
        "波动率 +50%": dict(sigma=state.sigma_anchor * 1.5),
        "波动率 -30%": dict(sigma=state.sigma_anchor * 0.7),
        "利率 +200bp": dict(r=state.r_1y + 0.02),
        "利率 -100bp": dict(r=max(state.r_1y - 0.01, 0.0)),
        "股价 -15%(K 不变)": dict(S=state.S * 0.85),
        "股价 +15%(K 不变)": dict(S=state.S * 1.15),
        "组合压力: 波动率+50% 且 利率+200bp": dict(sigma=state.sigma_anchor * 1.5,
                                             r=state.r_1y + 0.02),
    }
    rows = []
    for name, ov in scenarios.items():
        p = price_chooser(ov.get("S", state.S), state.S, ov.get("r", state.r_1y),
                          state.q, ov.get("sigma", state.sigma_anchor), t1, T2)
        rows.append({"情景": name, "价格": round(p, 4),
                     "变化($)": round(p - base_p, 4),
                     "变化(%)": round((p / base_p - 1) * 100, 2)})
    tbl = pd.DataFrame(rows)
    if cached:
        cache.set_json(k, tbl.to_dict("records"))
    return tbl


# ------------------------------------------------------------ universe (Week 10)

SCENARIOS = {
    "基准(当前市场)": {},
    "波动率 +50%": {"sig": 1.5}, "波动率 -30%": {"sig": 0.7},
    "利率 +200bp": {"dr": 0.02}, "利率 -100bp": {"dr": -0.01},
    "股价 -15%(K 不变)": {"S": 0.85}, "股价 +15%(K 不变)": {"S": 1.15},
    "组合压力: 波动率+50% 且 利率+200bp": {"sig": 1.5, "dr": 0.02},
}


def quote_universe(tickers: list[str] | None = None, as_of=None,
                   t1: float = 0.5, T2: float = 1.0) -> pd.DataFrame:
    """Vectorised dual quote for every ticker in one pass (ATM strike).
    Reads one row per ticker, prices the whole universe with the array
    closed form, runs the reference model on the feature matrix. Row-for-row
    identical to `quote()` (audit 4 in analysis/week10)."""
    from src.db import repository as repo
    latest = repo.read_latest_features(tickers, as_of)
    if not len(latest):
        return pd.DataFrame()
    rows = F.to_file_names(latest)
    # robustness: a ticker with a non-finite PRICING input must not sink the whole batch
    # (Week 12: only the required inputs - the optional GARCH state may be NaN, see anchor fallback)
    need = [c for c in REQUIRED + ML_FEATURES if c in rows.columns]
    bad = ~np.isfinite(rows[need].to_numpy(float)).all(axis=1)
    if bad.any():
        rows = rows[~bad]
        quote_universe.dropped = list(latest.index[bad])
    else:
        quote_universe.dropped = []
    S = rows["jpm_close"].to_numpy(float); r = rows["r_1y"].to_numpy(float)
    qd = np.minimum(rows["div_yield_ttm"].to_numpy(float), Q_CAP)
    sig = np.maximum(rows["sigma_ewma"].to_numpy(float), SIGMA_FLOOR)       # EWMA (ML reference base)
    g1d = rows["sigma_garch_1d"].to_numpy(float) if "sigma_garch_1d" in rows else np.full(len(rows), np.nan)
    use_g = (PRIMARY_ANCHOR == "garch1d") & np.isfinite(g1d)
    anchor = np.where(use_g, "garch1d", "ewma")
    sig_a = np.maximum(np.where(use_g, g1d, rows["sigma_ewma"].to_numpy(float)), SIGMA_FLOOR)
    regime = np.where((rows["vix_sentiment_pct"] > 0.75) | (rows["vix_level"] > 30), "fear", "calm")
    dates = pd.to_datetime(rows["date"])
    cs = xs.cross_section(as_of if as_of is not None else dates.max())
    in_scope = rows.index.isin(cs.index) & (dates == cs.attrs.get("date", pd.NaT)).to_numpy()
    ratio = np.where(in_scope, cs["ratio"].reindex(rows.index).to_numpy(float), np.nan)
    sig_ml = np.where(in_scope, cs["sigma_ml"].reindex(rows.index).to_numpy(float), np.nan)
    prim = np.asarray(chooser_closed_form(S0=S, K=S, r=r, q=qd, sigma=sig_a, t1=t1, T2=T2), float)
    ml = np.asarray(chooser_closed_form(S0=S, K=S, r=r, q=qd, sigma=sig_ml, t1=t1, T2=T2), float)
    call = np.asarray(bs_call(S, S, r, qd, sig_a, T2), float)
    put = np.asarray(bs_put(S, S, r, qd, sig_a, T2), float)
    band = np.array([BAND_MAE_BY_ANCHOR[a][g] for a, g in zip(anchor, regime)], float)
    out = pd.DataFrame({
        "date": rows["date"].values, "S": S, "K": S, "anchor": anchor, "sigma_anchor": sig_a,
        "sigma_garch_1d": g1d, "sigma_ewma": sig, "sigma_ml": sig_ml,
        "ml_ratio": ratio, "r": r, "q": qd, "regime": regime, "primary_price": prim,
        "band_lo": prim - band, "band_hi": prim + band, "ml_reference_price": ml,
        "call_leg": call, "put_leg": put, "straddle": call + put,
        "t1": t1, "t2": T2, "model_version": MODEL_VERSION, "ml_in_scope": in_scope}, index=rows.index)
    out.index.name = "ticker"
    return out


def scenario_universe(uq: pd.DataFrame, t1: float = 0.5, T2: float = 1.0) -> pd.DataFrame:
    """Eight scenarios for every ticker, vectorised; long format
    (ticker, date, scenario, price, change_usd, change_pct)."""
    S = uq["S"].to_numpy(float); r = uq["r"].to_numpy(float)
    qd = uq["q"].to_numpy(float); sig = uq["sigma_anchor"].to_numpy(float)
    base = uq["primary_price"].to_numpy(float)
    parts = []
    for name, ov in SCENARIOS.items():
        p = np.asarray(chooser_closed_form(S0=S * ov.get("S", 1.0), K=S,
                                           r=np.maximum(r + ov.get("dr", 0.0), 0.0), q=qd,
                                           sigma=np.maximum(sig * ov.get("sig", 1.0), SIGMA_FLOOR),
                                           t1=t1, T2=T2), float)
        parts.append(pd.DataFrame({"ticker": uq.index, "date": uq["date"].values, "scenario": name,
                                   "price": np.round(p, 4), "change_usd": np.round(p - base, 4),
                                   "change_pct": np.round((p / base - 1) * 100, 2)}))
    return pd.concat(parts, ignore_index=True)


def refresh_data(ticker: str = config.TICKER) -> str:
    """Manual refresh button: Week 9 routes it through the daily pipeline
    (collect -> DB -> features -> quote -> cache invalidation). Falls back to
    the Week 7 file-only refresh when no database is configured."""
    from src.service import pipeline
    return pipeline.run_daily([ticker]).summary()
