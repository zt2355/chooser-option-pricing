#!/bin/bash
# 本机开发用:在没有 Docker 的机器上启动/停止 PostgreSQL 与 Redis(Week 9)。
# 交付形态仍以 docker-compose.yml 为准;本脚本只是让开发机复现同一套服务。
#
# 用法: bash scripts/dev_services.sh start | stop | status | reset
# 可用环境变量覆盖二进制位置与数据目录:
#   PG_BIN    PostgreSQL bin 目录   (默认 ~/local/pg/bin,源码编译安装)
#   REDIS_BIN redis-server 所在目录 (默认 ~/local/src/redis-stable/src)
#   DEV_DATA  数据目录             (默认 ~/local/chooser-dev)
set -e
PG_BIN=${PG_BIN:-$HOME/local/pg/bin}
REDIS_BIN=${REDIS_BIN:-$HOME/local/src/redis-stable/src}
DEV_DATA=${DEV_DATA:-$HOME/local/chooser-dev}
PGDATA="$DEV_DATA/pgdata"; PGLOG="$DEV_DATA/postgres.log"
REDIS_DIR="$DEV_DATA/redis"; REDIS_PID="$REDIS_DIR/redis.pid"
DB_USER=chooser; DB_PASS=chooser; DB_NAME=chooser; DB_PORT=${DB_PORT:-5432}; REDIS_PORT=${REDIS_PORT:-6379}

start() {
  mkdir -p "$DEV_DATA" "$REDIS_DIR"
  if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "[pg] initdb -> $PGDATA"
    "$PG_BIN/initdb" -D "$PGDATA" -U postgres --auth=trust -E UTF8 >/dev/null
  fi
  if ! "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" -l "$PGLOG" -o "-p $DB_PORT -c listen_addresses=localhost" -w start >/dev/null
    echo "[pg] started on :$DB_PORT"
  else
    echo "[pg] already running"
  fi
  "$PG_BIN/psql" -p "$DB_PORT" -U postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 \
    || "$PG_BIN/psql" -p "$DB_PORT" -U postgres -c "CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASS'" >/dev/null
  "$PG_BIN/psql" -p "$DB_PORT" -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
    || "$PG_BIN/psql" -p "$DB_PORT" -U postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER" >/dev/null
  if [ -f "$REDIS_PID" ] && kill -0 "$(cat "$REDIS_PID")" 2>/dev/null; then
    echo "[redis] already running"
  else
    "$REDIS_BIN/redis-server" --port "$REDIS_PORT" --bind 127.0.0.1 --daemonize yes \
      --dir "$REDIS_DIR" --pidfile "$REDIS_PID" --logfile "$REDIS_DIR/redis.log" \
      --save "" --appendonly no >/dev/null
    echo "[redis] started on :$REDIS_PORT"
  fi
  echo "DATABASE_URL=postgresql+psycopg2://$DB_USER:$DB_PASS@localhost:$DB_PORT/$DB_NAME"
  echo "REDIS_URL=redis://localhost:$REDIS_PORT/0"
}

stop() {
  "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast stop >/dev/null 2>&1 && echo "[pg] stopped" || echo "[pg] not running"
  if [ -f "$REDIS_PID" ] && kill -0 "$(cat "$REDIS_PID")" 2>/dev/null; then
    "$REDIS_BIN/redis-cli" -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true; echo "[redis] stopped"
  else echo "[redis] not running"; fi
}

status() {
  "$PG_BIN/pg_ctl" -D "$PGDATA" status 2>/dev/null | head -1 || echo "[pg] not running"
  "$REDIS_BIN/redis-cli" -p "$REDIS_PORT" ping 2>/dev/null | sed 's/^/[redis] /' || echo "[redis] not running"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  reset) stop; rm -rf "$PGDATA" "$REDIS_DIR"; echo "data dirs removed"; start ;;
  *) echo "用法: bash scripts/dev_services.sh start|stop|status|reset"; exit 1 ;;
esac
