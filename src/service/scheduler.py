"""Scheduler process: runs the daily pipeline after the US close.

    python -m src.service.scheduler           blocking scheduler (container entrypoint)
    python -m src.service.scheduler --once    run the job now and exit

Due time: Mon-Fri 17:30 America/New_York (PIPELINE_CRON_HOUR / PIPELINE_CRON_MINUTE /
PIPELINE_TZ). The GitHub Actions weekly job from Week 2 is kept as an independent
backup path.

Week 12 fix (catch-up heartbeat). Until Week 11 a cron trigger fired the job at the
due time. On a laptop host the Docker VM is suspended while the Mac sleeps; the
scheduler's timed wait runs on the VM's monotonic clock, which stops during the
suspension, so a wait computed for "51 hours from now" kept being postponed and the
job silently stopped firing (no run from 2026-09-26 to 2026-10-02, not even a
"missed" warning). Now a heartbeat wakes every HEARTBEAT_MIN minutes, compares WALL
time with the most recent due time and runs the pipeline if no successful daily run
started after it. A wait is at most HEARTBEAT_MIN, so a suspension can delay a run by
at most that much after the host wakes. Each heartbeat touches HEARTBEAT_FILE, which
the container healthcheck reads (the image's default healthcheck probes Streamlit,
which this container does not run).
"""

import datetime as dt
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from src.service.pipeline import run_daily

log = logging.getLogger("scheduler")
HEARTBEAT_MIN = 10
HEARTBEAT_FILE = Path(os.environ.get("SCHEDULER_HEARTBEAT_FILE", "/tmp/scheduler_heartbeat"))


def _due_settings():
    tz = ZoneInfo(os.environ.get("PIPELINE_TZ", "America/New_York"))
    return tz, int(os.environ.get("PIPELINE_CRON_HOUR", "17")), int(os.environ.get("PIPELINE_CRON_MINUTE", "30"))


def last_due(now: dt.datetime, tz: ZoneInfo, hour: int, minute: int) -> dt.datetime:
    """Most recent Mon-Fri hour:minute (in tz) that is <= now."""
    local = now.astimezone(tz)
    cand = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand > local:
        cand -= dt.timedelta(days=1)
    while cand.weekday() >= 5:                       # Sat/Sun -> previous Friday
        cand -= dt.timedelta(days=1)
    return cand


def last_daily_start() -> dt.datetime | None:
    """Start time of the most recent daily run that finished ok/partial (None if none / no DB)."""
    try:
        import sqlalchemy as sa
        from src.db.engine import get_engine
        with get_engine().connect() as c:
            return c.execute(sa.text("SELECT max(started_at) FROM pipeline_runs "
                                     "WHERE job = 'daily' AND status IN ('ok', 'partial')")).scalar()
    except Exception as e:                           # DB down: do not crash the scheduler
        log.warning("cannot read pipeline_runs: %s", e)
        return None


def needs_run(now: dt.datetime, last_start: dt.datetime | None) -> bool:
    tz, hour, minute = _due_settings()
    due = last_due(now, tz, hour, minute)
    return last_start is None or last_start < due


def job():
    rep = run_daily()
    log.info(rep.summary())


def heartbeat():
    HEARTBEAT_FILE.write_text(dt.datetime.now(dt.timezone.utc).isoformat())
    if needs_run(dt.datetime.now(dt.timezone.utc), last_daily_start()):
        log.info("daily run due and not done yet -> running now")
        job()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if "--once" in sys.argv:
        job()
        return 0
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    tz, hour, minute = _due_settings()
    sched = BlockingScheduler(timezone=tz)
    sched.add_job(heartbeat, IntervalTrigger(minutes=HEARTBEAT_MIN), id="heartbeat",
                  max_instances=1, coalesce=True, misfire_grace_time=None,
                  next_run_time=dt.datetime.now(tz))
    log.info("daily pipeline due Mon-Fri %02d:%02d %s; catch-up heartbeat every %d min",
             hour, minute, tz.key, HEARTBEAT_MIN)
    if os.environ.get("PIPELINE_RUN_ON_START", "0") == "1":
        job()
    sched.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
