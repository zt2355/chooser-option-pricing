"""SQLAlchemy engine factory.

DATABASE_URL comes from the environment (docker-compose sets it) or from .env
(loaded by src.config). When it is unset the service layer falls back to the
file-based data path, so every Week 1-8 script keeps working without a DB.
"""

import os
from functools import lru_cache

import sqlalchemy as sa

from src import config  # noqa: F401  (loads .env)

DEFAULT_URL = "postgresql+psycopg2://chooser:chooser@localhost:5432/chooser"


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL")
    return url if url else None


@lru_cache(maxsize=None)
def get_engine(url: str | None = None) -> sa.Engine:
    url = url or database_url() or DEFAULT_URL
    return sa.create_engine(url, pool_pre_ping=True, future=True)


def db_available(url: str | None = None, timeout: float = 2.0) -> bool:
    """Cheap liveness probe; never raises."""
    try:
        eng = get_engine(url)
        with eng.connect() as c:
            c.execute(sa.text("SELECT 1"))
        return True
    except Exception:
        return False
