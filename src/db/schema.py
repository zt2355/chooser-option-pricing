"""Table definitions (SQLAlchemy Core).

Design notes
  * Every market table carries a `ticker` column so the same schema serves
    JPM today and the S&P 500 / Russell 3000 universe in Weeks 10-11.
  * `features_daily` stores the 18 Week-2 features with generic column names
    (close/adj_close instead of jpm_close/...). The file pipeline keeps the
    historical jpm_* names; `src.features` maps between the two, and the
    Week 9 audit asserts the two representations are bit-identical for JPM.
  * `quotes_daily` is the tool's output table: one row per (ticker, date)
    with the primary EWMA-anchored price, the ML reference price, the
    regime band and the model version that produced it.
  * `pipeline_runs` is the operational log the scheduler writes to.
"""

import sqlalchemy as sa

metadata = sa.MetaData()

securities = sa.Table(
    "securities", metadata,
    sa.Column("ticker", sa.String(16), primary_key=True),
    sa.Column("name", sa.String(128)),
    sa.Column("sector", sa.String(64)),
    sa.Column("universe", sa.String(32), nullable=False, server_default="custom"),
    sa.Column("note", sa.String(128)),          # e.g. no_price_history
    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
)

prices_daily = sa.Table(
    "prices_daily", metadata,
    sa.Column("ticker", sa.String(16), sa.ForeignKey("securities.ticker"), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("open", sa.Float), sa.Column("high", sa.Float),
    sa.Column("low", sa.Float), sa.Column("close", sa.Float),
    sa.Column("adj_close", sa.Float), sa.Column("volume", sa.BigInteger),
    sa.Column("dividend", sa.Float, server_default="0"),
    sa.Column("split", sa.Float, server_default="0"),
    sa.Column("source", sa.String(32)),
    sa.Index("ix_prices_date", "date"),
)

index_daily = sa.Table(
    "index_daily", metadata,
    sa.Column("symbol", sa.String(16), primary_key=True),   # ^VIX ^GSPC ^TNX ^IRX
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("open", sa.Float), sa.Column("high", sa.Float),
    sa.Column("low", sa.Float), sa.Column("close", sa.Float),
)

rates_daily = sa.Table(
    "rates_daily", metadata,
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("dgs3mo", sa.Float), sa.Column("dgs1", sa.Float),
    sa.Column("dgs10", sa.Float), sa.Column("fedfunds", sa.Float),
    sa.Column("t10y2y", sa.Float),
)

FEATURE_COLUMNS = [
    "log_return", "realized_vol_21d", "realized_vol_63d", "realized_vol_252d",
    "parkinson_vol_21d", "div_yield_ttm", "div_growth_yoy",
    "r_3m", "r_1y", "curve_slope", "rate_momentum_21d",
    "spx_log_return", "beta_63d", "corr_vix_63d",
    "vix_level", "vix_sentiment_pct", "vix_momentum_5d", "fear_regime",
]

# Week 12: GARCH(1,1) 1-day anchor (the tool's primary anchor) and the state needed to
# continue its recursion incrementally (see src/garch_state.py)
GARCH_COLUMNS = ["sigma_garch_1d", "sigma_garch_1y", "garch_i", "garch_mu", "garch_omega",
                 "garch_alpha", "garch_beta", "garch_h_next"]

features_daily = sa.Table(
    "features_daily", metadata,
    sa.Column("ticker", sa.String(16), sa.ForeignKey("securities.ticker"), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("close", sa.Float), sa.Column("adj_close", sa.Float),
    sa.Column("vix_close", sa.Float),
    *[sa.Column(c, sa.Float) for c in FEATURE_COLUMNS],
    # Week 10: derived state stored so a quote reads ONE row instead of the
    # whole history (EWMA anchor and W6 rate-deviation feature)
    sa.Column("sigma_ewma", sa.Float), sa.Column("r_1y_dev", sa.Float),
    *[sa.Column(c, sa.Float) for c in GARCH_COLUMNS],
    sa.Index("ix_features_date", "date"),
)

# Week 11 (perf): one row per ticker = its latest COMPLETE feature row.
# Maintained by src.db.features_db on every rebuild so a universe quote is an
# O(tickers) read instead of a DISTINCT ON over the whole history.
features_latest = sa.Table(
    "features_latest", metadata,
    sa.Column("ticker", sa.String(16), sa.ForeignKey("securities.ticker"), primary_key=True),
    sa.Column("date", sa.Date, nullable=False),
    sa.Column("close", sa.Float), sa.Column("adj_close", sa.Float),
    sa.Column("vix_close", sa.Float),
    *[sa.Column(c, sa.Float) for c in FEATURE_COLUMNS],
    sa.Column("sigma_ewma", sa.Float), sa.Column("r_1y_dev", sa.Float),
    *[sa.Column(c, sa.Float) for c in GARCH_COLUMNS],
)

index_membership = sa.Table(
    "index_membership", metadata,
    sa.Column("index", sa.String(16), primary_key=True),       # sp500
    sa.Column("ticker", sa.String(16), primary_key=True),
    sa.Column("start", sa.Date, primary_key=True),
    sa.Column("end", sa.Date),                                 # NULL = still a member
    sa.Column("start_known", sa.Boolean),
)

quotes_daily = sa.Table(
    "quotes_daily", metadata,
    sa.Column("ticker", sa.String(16), sa.ForeignKey("securities.ticker"), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("spot", sa.Float), sa.Column("strike", sa.Float),
    sa.Column("t1", sa.Float), sa.Column("t2", sa.Float),
    sa.Column("sigma_ewma", sa.Float), sa.Column("sigma_ml", sa.Float),
    sa.Column("ml_ratio", sa.Float),
    sa.Column("anchor", sa.String(16)), sa.Column("sigma_anchor", sa.Float),     # Week 12
    sa.Column("sigma_garch_1d", sa.Float),
    sa.Column("r", sa.Float), sa.Column("q", sa.Float),
    sa.Column("regime", sa.String(8)),
    sa.Column("primary_price", sa.Float),
    sa.Column("band_lo", sa.Float), sa.Column("band_hi", sa.Float),
    sa.Column("ml_reference_price", sa.Float),
    sa.Column("call_leg", sa.Float), sa.Column("put_leg", sa.Float),
    sa.Column("model_version", sa.String(64)),
    sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
)

scenarios_daily = sa.Table(
    "scenarios_daily", metadata,
    sa.Column("ticker", sa.String(16), primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("scenario", sa.String(64), primary_key=True),
    sa.Column("price", sa.Float), sa.Column("change_usd", sa.Float),
    sa.Column("change_pct", sa.Float),
)

model_registry = sa.Table(
    "model_registry", metadata,
    sa.Column("name", sa.String(64), primary_key=True),
    sa.Column("path", sa.String(256)), sa.Column("sha256", sa.String(64)),
    sa.Column("role", sa.String(32)),          # primary / reference
    sa.Column("registered_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    sa.Column("notes", sa.Text),
)

pipeline_runs = sa.Table(
    "pipeline_runs", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("job", sa.String(32)),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("finished_at", sa.DateTime(timezone=True)),
    sa.Column("status", sa.String(16)),
    sa.Column("tickers", sa.Integer), sa.Column("rows_prices", sa.Integer),
    sa.Column("rows_features", sa.Integer), sa.Column("rows_quotes", sa.Integer),
    sa.Column("data_version", sa.String(64)),   # e.g. "prices@2026-09-11 n=2185" (reproducibility tag)
    sa.Column("timings", sa.Text),               # Week 10: JSON of per-stage seconds
    sa.Column("message", sa.Text),
)
