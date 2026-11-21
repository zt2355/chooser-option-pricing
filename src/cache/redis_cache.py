"""Redis cache for quotes / scenarios / market snapshots.

Policy
  * Keys are namespaced `chooser:<kind>:<ticker>:<...>`; values are JSON.
  * TTL defaults to 1 hour for live quotes; the daily pipeline invalidates
    a ticker's keys after writing new rows, so stale reads cannot outlive
    a data refresh.
  * If REDIS_URL is unset or the server is unreachable, every call degrades
    to a no-op (miss) and the service computes directly. `stats()` reports
    hits/misses so the performance report can quantify the cache.
"""

import json
import os
from functools import lru_cache

from src import config  # noqa: F401  (loads .env)

NS = "chooser"
DEFAULT_TTL = 3600


def redis_url() -> str | None:
    return os.environ.get("REDIS_URL") or None


@lru_cache(maxsize=None)
def _client():
    url = redis_url()
    if not url:
        return None
    try:
        import redis
        r = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1,
                                 decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


_stats = {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0}


def available() -> bool:
    return _client() is not None


def key(kind: str, *parts) -> str:
    return ":".join([NS, kind, *[str(p) for p in parts]])


def get_json(k: str):
    r = _client()
    if r is None:
        _stats["misses"] += 1
        return None
    try:
        v = r.get(k)
    except Exception:
        v = None
    if v is None:
        _stats["misses"] += 1
        return None
    _stats["hits"] += 1
    return json.loads(v)


def set_json(k: str, value, ttl: int = DEFAULT_TTL) -> None:
    r = _client()
    if r is None:
        return
    try:
        r.set(k, json.dumps(value, default=_default), ex=ttl)
        _stats["sets"] += 1
    except Exception:
        pass


def invalidate(ticker: str) -> int:
    """Drop every key for one ticker (called after a data refresh)."""
    r = _client()
    if r is None:
        return 0
    n = 0
    try:
        for k in r.scan_iter(match=f"{NS}:*:{ticker}:*"):
            r.delete(k); n += 1
        for k in r.scan_iter(match=f"{NS}:*:{ticker}"):
            r.delete(k); n += 1
    except Exception:
        pass
    _stats["invalidations"] += n
    return n


def stats() -> dict:
    out = dict(_stats)
    out["available"] = available()
    r = _client()
    if r is not None:
        try:
            out["keys"] = sum(1 for _ in r.scan_iter(match=f"{NS}:*"))
            out["used_memory"] = r.info("memory").get("used_memory_human")
        except Exception:
            pass
    return out


def _default(o):
    import datetime as dt
    import numpy as np
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
