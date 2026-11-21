"""Persistence layer (Week 9): PostgreSQL via SQLAlchemy.

Modules
  engine      connection factory from DATABASE_URL (env / .env)
  schema      table definitions (multi-ticker from day one)
  repository  read/write helpers used by the pipeline and the pricing service
  migrate     `python -m src.db.migrate`  create tables (idempotent)
  ingest      `python -m src.db.ingest`   load data/raw + data/processed into PG
"""
