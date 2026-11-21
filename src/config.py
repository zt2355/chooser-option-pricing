"""Central configuration: loads API keys from .env and defines data-collection constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Collection window (paper replication period per project spec).
# END_DATE can be overridden via env var (used by the scheduled GitHub
# Actions run to keep data current); modeling code slices to <= 2024-12-31.
START_DATE = "2018-01-01"
END_DATE = os.environ.get("END_DATE", "2024-12-31")

TICKER = "JPM"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids external python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
