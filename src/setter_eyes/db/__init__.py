"""Event database (SQLite via stdlib). The source of truth for all stats."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# Bump when schema.sql changes in a way CREATE ... IF NOT EXISTS can't apply
# to an existing file. Stored in PRAGMA user_version.
SCHEMA_VERSION = 2


class SchemaVersionError(RuntimeError):
    pass


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the event DB, applying the schema."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    has_tables = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    if has_tables and version != SCHEMA_VERSION:
        conn.close()
        raise SchemaVersionError(
            f"{db_path} uses schema v{version}, this code needs v{SCHEMA_VERSION}. "
            "The labels DB is rebuilt from label files: delete it and re-run "
            "scripts/ingest_labels.py.")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn
