"""Create all tables (idempotent).  Usage: python -m src.db.migrate [--drop]"""

import sys

from src.db import schema
from src.db.engine import get_engine


def migrate(drop: bool = False) -> list[str]:
    eng = get_engine()
    if drop:
        schema.metadata.drop_all(eng)
    schema.metadata.create_all(eng)
    # additive column migrations for databases created by an earlier schema
    import sqlalchemy as sa
    with eng.begin() as c:
        for stmt in ("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS data_version VARCHAR(64)",
                     "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS timings TEXT",
                     "ALTER TABLE securities ADD COLUMN IF NOT EXISTS sector VARCHAR(64)",
                     "ALTER TABLE securities ADD COLUMN IF NOT EXISTS note VARCHAR(128)",
                     "ALTER TABLE features_daily ADD COLUMN IF NOT EXISTS sigma_ewma DOUBLE PRECISION",
                     "ALTER TABLE features_daily ADD COLUMN IF NOT EXISTS r_1y_dev DOUBLE PRECISION",
                     # Week 11: covering path for per-ticker tail reads / DISTINCT ON fallback
                     "CREATE INDEX IF NOT EXISTS ix_features_ticker_date_desc ON features_daily (ticker, date DESC)",
                     "CREATE INDEX IF NOT EXISTS ix_prices_ticker_date_desc ON prices_daily (ticker, date DESC)",
                     # Week 12: GARCH-1d primary anchor + recursion state; anchor recorded on each quote
                     *[f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS {c} DOUBLE PRECISION"
                       for t in ("features_daily", "features_latest") for c in schema.GARCH_COLUMNS],
                     "ALTER TABLE quotes_daily ADD COLUMN IF NOT EXISTS anchor VARCHAR(16)",
                     "ALTER TABLE quotes_daily ADD COLUMN IF NOT EXISTS sigma_anchor DOUBLE PRECISION",
                     "ALTER TABLE quotes_daily ADD COLUMN IF NOT EXISTS sigma_garch_1d DOUBLE PRECISION"):
            c.execute(sa.text(stmt))
        c.execute(sa.text("ANALYZE"))     # refresh planner stats so estimated counts are usable on a fresh DB
    return sorted(schema.metadata.tables)


if __name__ == "__main__":
    tables = migrate(drop="--drop" in sys.argv)
    print("tables:", ", ".join(tables))
