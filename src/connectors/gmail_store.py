"""SQLite DAO for raw Gmail interaction rows (metadata only).

One row per directed message edge (a message to N recipients => N rows). No
subject/body/snippet columns exist — metadata-only is enforced by schema.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Set

DEFAULT_DB_PATH = "data/database/networks.db"

_COLUMNS = [
    "src_email", "dst_email", "recipient_kind", "ts_utc",
    "thread_id", "size_bytes", "src_orgunit", "dst_orgunit",
]


class GmailInteractionStore:
    """Read/write access to the gmail_interactions table."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gmail_interactions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_domain   TEXT    NOT NULL,
                    sync_run_id  TEXT    NOT NULL,
                    src_email    TEXT    NOT NULL,
                    dst_email    TEXT    NOT NULL,
                    recipient_kind TEXT  NOT NULL,
                    ts_utc       INTEGER NOT NULL,
                    thread_id    TEXT,
                    size_bytes   INTEGER,
                    src_orgunit  TEXT,
                    dst_orgunit  TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_gmail_org_ts "
                "ON gmail_interactions(org_domain, ts_utc)"
            )

    def insert_rows(self, org_domain: str, sync_run_id: str,
                    rows: Iterable[Dict]) -> int:
        payload = [
            (org_domain, sync_run_id,
             r["src_email"], r["dst_email"], r["recipient_kind"], int(r["ts_utc"]),
             r.get("thread_id"), r.get("size_bytes"),
             r.get("src_orgunit"), r.get("dst_orgunit"))
            for r in rows
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO gmail_interactions "
                "(org_domain, sync_run_id, src_email, dst_email, recipient_kind, "
                " ts_utc, thread_id, size_bytes, src_orgunit, dst_orgunit) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        return len(payload)

    def query_window(self, org_domain: str, start_ts: int,
                     end_ts: int) -> List[Dict]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM gmail_interactions "
                "WHERE org_domain = ? AND ts_utc >= ? AND ts_utc <= ? "
                "ORDER BY ts_utc",
                (org_domain, start_ts, end_ts),
            )
            return [dict(row) for row in cur.fetchall()]

    def column_names(self) -> Set[str]:
        with self._connect() as conn:
            cur = conn.execute("PRAGMA table_info(gmail_interactions)")
            return {row["name"] for row in cur.fetchall()}
